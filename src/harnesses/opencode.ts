import { createOpencodeServer, createOpencodeClient } from '@opencode-ai/sdk';

import { PROJECT_ROOT, OPENCODE_SERVER_PORT, OPENCODE_HOST } from '../config.js';
import { logger } from '../logger.js';
import type { AgentHarness, AgentOptions, AgentResult } from './index.js';
import type { UsageInfo } from '../agent.js';

// ── Singleton server + client ─────────────────────────────────────────────────
// One OpenCode server per ccclaw process, on a dedicated port (default 4097)
// so it never conflicts with the user's own opencode TUI (which uses 4096).
// The server is spawned lazily on first use and reused for every subsequent query.

type OcClient = Awaited<ReturnType<typeof createOpencodeClient>>;
type OcServer = { url: string; close(): void };

let _client: OcClient | null = null;
let _ownedServer: OcServer | null = null;
let _connecting: Promise<OcClient> | null = null;
let _cleanupRegistered = false;

function registerCleanup(): void {
  if (_cleanupRegistered) return;
  _cleanupRegistered = true;
  const cleanup = () => {
    if (_ownedServer) {
      try { _ownedServer.close(); } catch { /* ignore */ }
      _ownedServer = null;
    }
  };
  process.on('exit', cleanup);
  process.on('SIGTERM', () => { cleanup(); });
  process.on('SIGINT',  () => { cleanup(); });
}

async function getClient(): Promise<OcClient> {
  if (_client) return _client;

  // Deduplicate concurrent first calls (e.g. two messages arriving simultaneously)
  if (_connecting) return _connecting;

  _connecting = (async (): Promise<OcClient> => {
    const baseUrl = `http://${OPENCODE_HOST}:${OPENCODE_SERVER_PORT}`;

    // 1. Try connecting to an already-running server on the dedicated port
    //    (user may have pre-started `opencode serve --port 4097`)
    try {
      const candidate = createOpencodeClient({ baseUrl });
      await (candidate as unknown as { config: { get(): Promise<unknown> } }).config.get();
      logger.info({ port: OPENCODE_SERVER_PORT }, 'Connected to existing OpenCode server');
      _client = candidate;
      registerCleanup();
      return _client;
    } catch (err: unknown) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code !== 'ECONNREFUSED') throw err; // unexpected — surface it
    }

    // 2. Nothing listening — spawn our own server on the dedicated port
    logger.info({ host: OPENCODE_HOST, port: OPENCODE_SERVER_PORT }, 'Spawning OpenCode server');
    _ownedServer = await createOpencodeServer({
      hostname: OPENCODE_HOST,
      port: OPENCODE_SERVER_PORT,
      timeout: 15000, // first startup can be slow on some systems
    });
    logger.info({ url: _ownedServer.url }, 'OpenCode server ready');
    _client = createOpencodeClient({ baseUrl: _ownedServer.url });
    registerCleanup();
    return _client;
  })();

  try {
    const result = await _connecting;
    _connecting = null;
    return result;
  } catch (err) {
    _connecting = null;
    throw err;
  }
}

// Reset the singleton (called when the server dies mid-session)
function resetClient(): void {
  _client = null;
  // Don't close _ownedServer here — it may already be dead
  _ownedServer = null;
}

// ── Harness ───────────────────────────────────────────────────────────────────

interface ParsedModel {
  providerID: string;
  modelID: string;
}

export class OpenCodeHarness implements AgentHarness {
  readonly type = 'opencode' as const;
  readonly supportsStructuredOutput = true;
  readonly supportsMultiProvider = true;

  async run(options: AgentOptions): Promise<AgentResult> {
    let sessionId: string | undefined;
    let streamedText = '';

    logger.info(
      { sessionId: options.sessionId ?? 'new', messageLen: options.message.length },
      'Starting OpenCode query',
    );

    let client: OcClient;
    try {
      client = await getClient();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.error({ err, host: OPENCODE_HOST, port: OPENCODE_SERVER_PORT }, 'Failed to start/connect OpenCode server');
      return {
        text: `Could not start OpenCode server on port ${OPENCODE_SERVER_PORT}.\n\nMake sure the opencode binary is installed:\n  npm install -g opencode-ai\n\nError: ${msg}`,
        newSessionId: undefined,
        usage: null,
      };
    }

    try {
      const directory = options.cwd || PROJECT_ROOT;

      if (options.sessionId) {
        sessionId = options.sessionId;
      } else {
        const sessionResp = await client.session.create({
          body: { title: `ccclaw-${Date.now()}` },
          query: { directory },
        });

        if (sessionResp.error) {
          throw new Error(`Failed to create OpenCode session: ${JSON.stringify(sessionResp.error)}`);
        }
        sessionId = sessionResp.data!.id;

        if (options.systemPrompt) {
          await client.session.prompt({
            path: { id: sessionId },
            body: {
              noReply: true,
              parts: [{ type: 'text' as const, text: options.systemPrompt }],
            },
            query: { directory },
          });
        }
      }

      if (options.abortController) {
        const sid = sessionId;
        options.abortController.signal.addEventListener('abort', async () => {
          try {
            await client.session.abort({ path: { id: sid } });
          } catch (err) {
            logger.warn({ err }, 'Failed to abort OpenCode session');
          }
        });
      }

      const typingInterval = options.onTyping ? setInterval(options.onTyping, 4000) : null;

      const eventSource = await client.event.subscribe({ query: { directory } });

      const streamProcessing = (async () => {
        try {
          for await (const event of eventSource.stream) {
            if (event.type === 'message.part.updated') {
              const part = event.properties.part;
              if (part.type === 'text' && options.onStreamText && event.properties.delta) {
                streamedText += event.properties.delta;
                options.onStreamText(streamedText);
              }
              if (part.type === 'tool' && options.onProgress) {
                const toolPart = part as { tool: string };
                options.onProgress({
                  type: 'tool_active',
                  description: toolPart.tool,
                });
              }
            }
            if (event.type === 'session.idle') {
              break;
            }
          }
        } catch (err) {
          logger.debug({ err }, 'Event stream ended');
        }
      })();

      const parsedModel = options.model ? this.parseModel(options.model) : undefined;

      const promptResult = await client.session.prompt({
        path: { id: sessionId },
        body: {
          ...(parsedModel ? { model: parsedModel } : {}),
          parts: [{ type: 'text' as const, text: options.message }],
        },
        query: { directory },
      });

      await streamProcessing;

      if (typingInterval) clearInterval(typingInterval);

      let resultText: string | null = null;
      let usage: UsageInfo | null = null;

      if (promptResult.data) {
        resultText = this.extractText(promptResult.data.parts);
        usage = this.extractUsage(promptResult.data.info);
      } else {
        logger.warn({ promptResult: JSON.stringify(promptResult) }, 'No promptResult.data');
      }

      logger.info(
        { inputTokens: usage?.inputTokens ?? 0, costUsd: usage?.totalCostUsd ?? 0, resultText: resultText?.slice(0, 100) },
        'OpenCode turn usage',
      );

      return { text: resultText, newSessionId: sessionId, usage };

    } catch (err) {
      if (options.abortController?.signal.aborted) {
        logger.info('OpenCode query aborted by user');
        return { text: null, newSessionId: sessionId, usage: null, aborted: true };
      }

      const error = err as NodeJS.ErrnoException;
      const msg = err instanceof Error ? err.message : String(err);

      // Server died mid-session — reset so next call triggers a respawn
      if (error.code === 'ECONNREFUSED' || error.code === 'EPIPE' ||
          msg.includes('ECONNREFUSED') || msg.includes('EPIPE')) {
        logger.error({ err }, 'OpenCode server connection lost — will respawn on next query');
        resetClient();
        return {
          text: 'OpenCode server connection lost. The server will restart automatically on your next message.',
          newSessionId: undefined,
          usage: null,
        };
      }

      logger.error({ err }, 'OpenCode query failed');
      return { text: `Error: ${msg}`, newSessionId: sessionId, usage: null };
    }
  }

  private parseModel(model: string): ParsedModel {
    if (model.includes('/')) {
      const slash = model.indexOf('/');
      return { providerID: model.slice(0, slash), modelID: model.slice(slash + 1) };
    }
    const lower = model.toLowerCase();
    if (lower.includes('claude')) return { providerID: 'anthropic', modelID: model };
    if (lower.includes('gpt'))    return { providerID: 'openai',    modelID: model };
    if (lower.includes('gemini')) return { providerID: 'google',    modelID: model };
    if (lower.includes('llama') || lower.includes('mixtral') || lower.includes('gemma')) {
      return { providerID: 'groq', modelID: model };
    }
    return { providerID: 'anthropic', modelID: model };
  }

  private extractText(parts: Array<{ type: string; text?: string }> | undefined): string | null {
    if (!parts) return null;
    const texts = parts.filter(p => p.type === 'text' && p.text).map(p => p.text!);
    return texts.length > 0 ? texts.join('\n') : null;
  }

  private extractUsage(info: {
    tokens?: { input?: number; output?: number; cache?: { read?: number } };
    cost?: number;
  } | undefined): UsageInfo | null {
    if (!info?.tokens) return null;
    return {
      inputTokens: info.tokens.input ?? 0,
      outputTokens: info.tokens.output ?? 0,
      cacheReadInputTokens: info.tokens.cache?.read ?? 0,
      totalCostUsd: info.cost ?? 0,
      didCompact: false,
      preCompactTokens: null,
      lastCallCacheRead: 0,
      lastCallInputTokens: info.tokens.input ?? 0,
    };
  }
}
