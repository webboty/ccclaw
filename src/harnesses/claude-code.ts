import fs from 'fs';
import path from 'path';

import { query } from '@anthropic-ai/claude-agent-sdk';

import { AGENT_MAX_TURNS, agentCwd, PROJECT_ROOT } from '../config.js';
import { readEnvFile } from '../env.js';
import { logger } from '../logger.js';
import type { AgentHarness, AgentOptions, AgentProgressEvent, AgentResult } from './index.js';
import type { UsageInfo } from '../agent.js';

const TOOL_LABELS: Record<string, string> = {
  Read: 'Reading file',
  Write: 'Writing file',
  Edit: 'Editing file',
  Bash: 'Running command',
  Grep: 'Searching code',
  Glob: 'Finding files',
  WebSearch: 'Web search',
  WebFetch: 'Fetching page',
  Agent: 'Sub-agent',
  NotebookEdit: 'Editing notebook',
  AskUserQuestion: 'User question',
};

export interface McpStdioConfig {
  command: string;
  args?: string[];
  env?: Record<string, string>;
}

function loadMcpServers(allowlist?: string[]): Record<string, McpStdioConfig> {
  const merged: Record<string, McpStdioConfig> = {};

  const projectSettings = path.join(agentCwd ?? PROJECT_ROOT, '.claude', 'settings.json');
  const userSettings = path.join(process.env.HOME ?? '/tmp', '.claude', 'settings.json');

  for (const file of [userSettings, projectSettings]) {
    try {
      const raw = JSON.parse(fs.readFileSync(file, 'utf-8'));
      const servers = raw?.mcpServers;
      if (servers && typeof servers === 'object') {
        for (const [name, config] of Object.entries(servers)) {
          const cfg = config as Record<string, unknown>;
          if (cfg.command && typeof cfg.command === 'string') {
            merged[name] = {
              command: cfg.command,
              ...(cfg.args ? { args: cfg.args as string[] } : {}),
              ...(cfg.env ? { env: cfg.env as Record<string, string> } : {}),
            };
          }
        }
      }
    } catch {}
  }

  if (allowlist) {
    const allowed = new Set(allowlist);
    for (const name of Object.keys(merged)) {
      if (!allowed.has(name)) delete merged[name];
    }
  }

  return merged;
}

async function* singleTurn(text: string): AsyncGenerator<{
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}> {
  yield {
    type: 'user',
    message: { role: 'user', content: text },
    parent_tool_use_id: null,
    session_id: '',
  };
}

export class ClaudeCodeHarness implements AgentHarness {
  readonly type = 'claude-code' as const;
  readonly supportsStructuredOutput = false;
  readonly supportsMultiProvider = true;

  async run(options: AgentOptions): Promise<AgentResult> {
    const secrets = readEnvFile(['CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_API_KEY']);

    const sdkEnv: Record<string, string | undefined> = { ...process.env };
    if (secrets.CLAUDE_CODE_OAUTH_TOKEN) {
      sdkEnv.CLAUDE_CODE_OAUTH_TOKEN = secrets.CLAUDE_CODE_OAUTH_TOKEN;
    }
    if (secrets.ANTHROPIC_API_KEY) {
      sdkEnv.ANTHROPIC_API_KEY = secrets.ANTHROPIC_API_KEY;
    }

    let newSessionId: string | undefined;
    let resultText: string | null = null;
    let usage: UsageInfo | null = null;
    let didCompact = false;
    let preCompactTokens: number | null = null;
    let lastCallCacheRead = 0;
    let lastCallInputTokens = 0;
    let streamedText = '';

    const typingInterval = options.onTyping ? setInterval(options.onTyping, 4000) : null;

    try {
      const mcpServers = loadMcpServers(options.mcpAllowlist);
      const mcpServerNames = Object.keys(mcpServers);
      logger.info(
        { sessionId: options.sessionId ?? 'new', messageLen: options.message.length, mcpServers: mcpServerNames },
        'Starting Claude Code query',
      );

      const mcpServerSpecs = mcpServerNames.length > 0 ? mcpServers : undefined;

      for await (const event of query({
        prompt: singleTurn(options.message),
        options: {
          cwd: options.cwd,
          resume: options.sessionId,
          settingSources: ['project', 'user'],
          permissionMode: 'bypassPermissions',
          allowDangerouslySkipPermissions: true,
          ...(AGENT_MAX_TURNS > 0 ? { maxTurns: AGENT_MAX_TURNS } : {}),
          env: sdkEnv,
          ...(mcpServerSpecs ? { mcpServers: mcpServerSpecs } : {}),
          includePartialMessages: !!options.onStreamText,
          ...(options.model ? { model: options.model } : {}),
          ...(options.abortController ? { abortController: options.abortController } : {}),
        },
      })) {
        const ev = event as Record<string, unknown>;

        if (ev['type'] === 'system' && ev['subtype'] === 'init') {
          newSessionId = ev['session_id'] as string;
          logger.info({ newSessionId }, 'Session initialized');
        }

        if (ev['type'] === 'system' && ev['subtype'] === 'compact_boundary') {
          didCompact = true;
          const meta = ev['compact_metadata'] as { trigger: string; pre_tokens: number } | undefined;
          preCompactTokens = meta?.pre_tokens ?? null;
          logger.warn({ trigger: meta?.trigger, preCompactTokens }, 'Context window compacted');
        }

        if (ev['type'] === 'assistant') {
          const msg = ev['message'] as Record<string, unknown> | undefined;
          const msgUsage = msg?.['usage'] as Record<string, number> | undefined;
          const callCacheRead = msgUsage?.['cache_read_input_tokens'] ?? 0;
          const callInputTokens = msgUsage?.['input_tokens'] ?? 0;
          if (callCacheRead > 0) lastCallCacheRead = callCacheRead;
          if (callInputTokens > 0) lastCallInputTokens = callInputTokens;

          if (options.onProgress) {
            const content = msg?.['content'] as Array<{ type: string; name?: string }> | undefined;
            if (Array.isArray(content)) {
              for (const block of content) {
                if (block.type === 'tool_use' && block.name) {
                  const label = TOOL_LABELS[block.name] ?? block.name;
                  options.onProgress({ type: 'tool_active', description: label });
                }
              }
            }
          }
        }

        if (ev['type'] === 'system' && ev['subtype'] === 'task_started' && options.onProgress) {
          options.onProgress({ type: 'task_started', description: (ev['description'] as string) ?? 'Sub-agent started' });
        }
        if (ev['type'] === 'system' && ev['subtype'] === 'task_notification' && options.onProgress) {
          const summary = (ev['summary'] as string) ?? 'Sub-agent finished';
          const status = (ev['status'] as string) ?? 'completed';
          options.onProgress({
            type: 'task_completed',
            description: status === 'failed' ? `Failed: ${summary}` : summary,
          });
        }

        if (ev['type'] === 'stream_event' && options.onStreamText && ev['parent_tool_use_id'] === null) {
          const streamEvent = ev['event'] as Record<string, unknown> | undefined;
          if (streamEvent?.['type'] === 'content_block_delta') {
            const delta = streamEvent['delta'] as Record<string, unknown> | undefined;
            if (delta?.['type'] === 'text_delta' && typeof delta['text'] === 'string') {
              streamedText += delta['text'];
              options.onStreamText(streamedText);
            }
          }
          if (streamEvent?.['type'] === 'message_start') {
            streamedText = '';
          }
        }

        if (ev['type'] === 'result') {
          resultText = (ev['result'] as string | null | undefined) ?? null;
          const evUsage = ev['usage'] as Record<string, number> | undefined;
          if (evUsage) {
            usage = {
              inputTokens: evUsage['input_tokens'] ?? 0,
              outputTokens: evUsage['output_tokens'] ?? 0,
              cacheReadInputTokens: evUsage['cache_read_input_tokens'] ?? 0,
              totalCostUsd: (ev['total_cost_usd'] as number) ?? 0,
              didCompact,
              preCompactTokens,
              lastCallCacheRead,
              lastCallInputTokens,
            };
            logger.info(
              { inputTokens: usage.inputTokens, costUsd: usage.totalCostUsd, didCompact },
              'Turn usage',
            );
          }
        }
      }
    } catch (err) {
      if (options.abortController?.signal.aborted) {
        logger.info('Agent query aborted by user');
        return { text: null, newSessionId, usage, aborted: true };
      }
      throw err;
    } finally {
      if (typingInterval) clearInterval(typingInterval);
    }

    return { text: resultText, newSessionId, usage };
  }
}
