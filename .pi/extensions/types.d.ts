// Type declarations for pi extension runtime modules.
// These are provided by the pi runtime at execution time — no npm install needed.

declare module "node:child_process" {
	export function execSync(
		command: string,
		options?: { encoding?: string; cwd?: string },
	): string;
}

declare module "node:fs" {
	export function existsSync(path: string): boolean;
}

declare var process: {
	cwd(): string;
};

// Inline types for the pi runtime API (provided at execution time).
// Full declarations in @earendil-works/pi-coding-agent.
declare module "@earendil-works/pi-coding-agent" {
	interface ExecResult {
		code: number;
		stdout: string;
		stderr: string;
	}
	interface MessageOptions {
		deliverAs?: "nextTurn";
	}
	interface ToolResult {
		content: Array<{ type: string; text: string }>;
		details?: Record<string, unknown>;
	}
	interface RegisteredCommand {
		description: string;
		handler: (
			args: unknown,
			ctx: { ui: { notify: (msg: string, level: string) => void } },
		) => Promise<void>;
	}
	interface RegisteredTool {
		name: string;
		label: string;
		description: string;
		promptSnippet: string;
		promptGuidelines?: string[];
		parameters: unknown;
		execute: (...args: unknown[]) => Promise<ToolResult>;
	}

	export interface ExtensionAPI {
		exec(prog: string, args?: string[]): Promise<ExecResult>;
		on(event: string, handler: (event: unknown, ctx: unknown) => void): void;
		sendMessage(
			msg: { customType: string; content: string; display?: boolean },
			opts?: MessageOptions,
		): void;
		registerCommand(name: string, cmd: RegisteredCommand): void;
		registerTool(tool: RegisteredTool): void;
	}
	export interface ExtensionContext {}
}

declare module "typebox" {
	const Type: {
		Object(properties: Record<string, unknown>): unknown;
	};
	export { Type };
}
