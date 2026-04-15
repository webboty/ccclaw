import { createOpencode, createOpencodeClient } from '@opencode-ai/sdk';

import { PROJECT_ROOT } from '../config.js';
import { logger } from '../logger.js';
import type { AgentHarness, AgentOptions, AgentResult } from './index.js';
import type { UsageInfo } from '../agent.js';

const OPENCODE_PORT = parseInt(process.env.OPENCODE_PORT ?? '4096', 10);
const OPENCODE_HOST = process.env.OPENCODE_HOST ?? '127.0.0.1';

export interface OpenCodeProviderConfig {
  provider: string;
  model?: string;
}

const DEFAULT_PROVIDERS: Record<string, string> = {
  anthropic: 'claude-3-5-sonnet-20241022',
  openai: 'gpt-4o',
  google: 'gemini-2-5-flash',
  'models.dev': 'anthropic/claude-3-5-sonnet-20241022',
};

export class OpenCodeHarness implements AgentHarness {
  readonly type = 'opencode' as const;
  readonly supportsStructuredOutput = true;
  readonly supportsMultiProvider = true;

  async run(options: AgentOptions): Promise<AgentResult> {
    let opencode: Awaited<ReturnType<typeof createOpencode>> | null = null;
    let sessionId: string | undefined;
    let streamedText = '';

    logger.info(
      { sessionId: options.sessionId ?? 'new', messageLen: options.message.length },
      'Starting OpenCode query',
    );

    try {
      opencode = await createOpencode({
        hostname: OPENCODE_HOST,
        port: OPENCODE_PORT,
        timeout: 5000,
        config: {
          cwd: options.cwd || PROJECT_ROOT,
          ...(options.model ? { model: this.parseModel(options.model) } : {}),
        },
      });

      sessionId = options.sessionId;

      if (!sessionId) {
        const session = await opencode.client.session.create({
          body: {
            title: `ccclaw-${Date.now()}`,
          },
        });
        sessionId = session.id;

        if (options.systemPrompt) {
          await opencode.client.session.prompt({
            path: { id: sessionId },
            body: {
              noReply: true,
              parts: [{ type: 'text', text: options.systemPrompt }],
            },
          });
        }
      }

      if (options.abortController) {
        options.abortController.signal.addEventListener('abort', async () => {
          try {
            await opencode!.client.session.abort({ path: { id: sessionId! } });
          } catch (err) {
            logger.warn({ err }, 'Failed to abort OpenCode session');
          }
        });
      }

      const typingInterval = options.onTyping ? setInterval(options.onTyping, 4000) : null;

      const events = opencode.client.event.subscribe();

      const promptPromise = opencode.client.session.prompt({
        path: { id: sessionId },
        body: {
          parts: [{ type: 'text', text: options.message }],
        },
      });

      let resultText: string | null = null;
      let usage: UsageInfo | null = null;

      const streamProcessing = (async () => {
        try {
          for await (const event of events.stream) {
            if (event.type === 'message_start' && options.onStreamText) {
              streamedText = '';
            }
            if (event.type === 'content_block_delta' && options.onStreamText) {
              if (event.properties?.delta?.text) {
                streamedText += event.properties.delta.text;
                options.onStreamText(streamedText);
              }
            }
            if (event.type === 'tool_use' && options.onProgress) {
              options.onProgress({
                type: 'tool_active',
                description: event.properties?.name ?? 'Using tool',
              });
            }
            if (event.type === 'message_stop') {
              break;
            }
          }
        } catch (err) {
          logger.debug({ err }, 'Event stream ended');
        }
      })();

      const result = await promptPromise;

      await streamProcessing;

      if (typingInterval) clearInterval(typingInterval);

      const messageData = await opencode.client.session.message({
        path: { id: sessionId },
      });

      resultText = this.extractTextFromMessage(messageData);

      usage = this.extractUsage(messageData);

      logger.info(
        { inputTokens: usage?.inputTokens ?? 0, costUsd: usage?.totalCostUsd ?? 0 },
        'OpenCode turn usage',
      );

      return { text: resultText, newSessionId: sessionId, usage };
    } catch (err) {
      if (options.abortController?.signal.aborted) {
        logger.info('OpenCode query aborted by user');
        return { text: null, newSessionId: sessionId, usage: null, aborted: true };
      }
      throw err;
    } finally {
      if (opencode) {
        opencode.server.close();
      }
    }
  }

  private parseModel(model: string): OpenCodeProviderConfig {
    if (model.includes('/')) {
      const [provider, modelId] = model.split('/');
      return { provider, model: modelId };
    }
    const knownProviders = ['anthropic', 'openai', 'google', 'models.dev', 'groq', 'ollama'];
    const lowerModel = model.toLowerCase();
    if (lowerModel.includes('claude')) return { provider: 'anthropic', model };
    if (lowerModel.includes('gpt')) return { provider: 'openai', model };
    if (lowerModel.includes('gemini')) return { provider: 'google', model };
    return { provider: 'anthropic', model };
  }

  private extractTextFromMessage(message: { info: unknown; parts: unknown[] }): string | null {
    const parts = message.parts as Array<{ type: string; text?: string }> | undefined;
    if (!parts) return null;

    const textParts = parts.filter((p) => p.type === 'text' && p.text);
    return textParts.map((p) => p.text).join('\n') || null;
  }

  private extractUsage(message: { info: unknown }): UsageInfo | null {
    const info = message.info as {
      usage?: { input_tokens?: number; output_tokens?: number };
      cost_usd?: number;
    } | undefined;

    if (!info?.usage) return null;

    return {
      inputTokens: info.usage.input_tokens ?? 0,
      outputTokens: info.usage.output_tokens ?? 0,
      cacheReadInputTokens: 0,
      totalCostUsd: info.cost_usd ?? 0,
      didCompact: false,
      preCompactTokens: null,
      lastCallCacheRead: 0,
      lastCallInputTokens: info.usage.input_tokens ?? 0,
    };
  }

  async connectToExistingServer(): Promise<ReturnType<typeof createOpencodeClient>> {
    return createOpencodeClient({
      baseUrl: `http://${OPENCODE_HOST}:${OPENCODE_PORT}`,
    });
  }
}
