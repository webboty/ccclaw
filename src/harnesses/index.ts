import type { UsageInfo } from '../agent.js';
import { ClaudeCodeHarness } from './claude-code.js';
import { OpenCodeHarness } from './opencode.js';

export interface AgentProgressEvent {
  type: 'task_started' | 'task_completed' | 'tool_active';
  description: string;
}

export interface AgentResult {
  text: string | null;
  newSessionId?: string;
  usage: UsageInfo | null;
  aborted?: boolean;
}

export interface AgentOptions {
  message: string;
  sessionId?: string;
  cwd: string;
  model?: string;
  systemPrompt?: string;
  mcpAllowlist?: string[];
  abortController?: AbortController;
  onTyping?: () => void;
  onProgress?: (event: AgentProgressEvent) => void;
  onStreamText?: (accumulatedText: string) => void;
}

export type HarnessType = 'claude-code' | 'opencode';

export interface AgentHarness {
  type: HarnessType;
  run(options: AgentOptions): Promise<AgentResult>;
  supportsStructuredOutput?: boolean;
  supportsMultiProvider?: boolean;
}

export { ClaudeCodeHarness, OpenCodeHarness };

export function createHarness(type: HarnessType): AgentHarness {
  switch (type) {
    case 'claude-code':
      return new ClaudeCodeHarness();
    case 'opencode':
      return new OpenCodeHarness();
    default:
      throw new Error(`Unknown harness type: ${type}`);
  }
}