import { createOpencode, createOpencodeClient } from '@opencode-ai/sdk';

import { PROJECT_ROOT } from '../config.js';
import { logger } from '../logger.js';
import type { AgentHarness, AgentOptions, AgentResult } from './index.js';
import type { UsageInfo } from '../agent.js';

const OPENCODE_PORT = parseInt(process.env.OPENCODE_PORT ?? '4096', 10);
const OPENCODE_HOST = process.env.OPENCODE_HOST ?? '127.0.0.1';

interface ParsedModel {
  providerID: string;
  modelID: string;
}

export class OpenCodeHarness implements AgentHarness {
  readonly type = 'opencode' as const;
  readonly supportsStructuredOutput = true;
  readonly supportsMultiProvider = true;

  async run(options: AgentOptions): Promise<AgentResult> {
    let client: Awaited<ReturnType<typeof createOpencodeClient>> | null = null;
    let sessionId: string | undefined;
    let streamedText = '';

    logger.info(
      { sessionId: options.sessionId ?? 'new', messageLen: options.message.length },
      'Starting OpenCode query',
    );

    try {
      // Connect to existing OpenCode server
      logger.info({ host: OPENCODE_HOST, port: OPENCODE_PORT }, 'Connecting to OpenCode server');
      client = await createOpencodeClient({
        baseUrl: `http://${OPENCODE_HOST}:${OPENCODE_PORT}`,
      });

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
            await client!.session.abort({ path: { id: sid } });
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
      }

      logger.info(
        { inputTokens: usage?.inputTokens ?? 0, costUsd: usage?.totalCostUsd ?? 0 },
        'OpenCode turn usage',
      );

      return { text: resultText, newSessionId: sessionId, usage };
    } catch (err) {
      const error = err as Error & { code?: string };
      const msg = err instanceof Error ? err.message : String(err);
      if (error.code === 'ECONNREFUSED' || error.code === 'EPIPE' || msg.includes('EPIPE') || msg.includes('ECONNREFUSED')) {
        logger.error({ err, host: OPENCODE_HOST, port: OPENCODE_PORT }, 'Cannot reach OpenCode server');
        return {
          text: `Cannot connect to OpenCode server at ${OPENCODE_HOST}:${OPENCODE_PORT}.\n\nStart it first:\n  opencode serve --hostname ${OPENCODE_HOST} --port ${OPENCODE_PORT}`,
          newSessionId: undefined,
          usage: null,
        };
      }
      if (options.abortController?.signal.aborted) {
        logger.info('OpenCode query aborted by user');
        return { text: null, newSessionId: sessionId, usage: null, aborted: true };
      }
      logger.error({ err }, 'OpenCode query failed');
      return { text: `Error: ${err instanceof Error ? err.message : String(err)}`, newSessionId: sessionId, usage: null };
    } finally {
      // Client connection stays open for reuse
    }
  }

  private parseModel(model: string): ParsedModel {
    if (model.includes('/')) {
      const [providerID, modelID] = model.split('/');
      return { providerID, modelID };
    }
    const lower = model.toLowerCase();
    if (lower.includes('claude')) return { providerID: 'anthropic', modelID: model };
    if (lower.includes('gpt')) return { providerID: 'openai', modelID: model };
    if (lower.includes('gemini')) return { providerID: 'google', modelID: model };
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
    tokens?: { input?: number; output?: number; reasoning?: number; cache?: { read?: number; write?: number } };
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

  async connectToExistingServer(): Promise<ReturnType<typeof createOpencodeClient>> {
    return createOpencodeClient({
      baseUrl: `http://${OPENCODE_HOST}:${OPENCODE_PORT}`,
    });
  }
}
