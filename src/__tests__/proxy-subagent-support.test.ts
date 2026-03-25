/**
 * Phase 3: Subagent and Multi-turn Support Tests
 *
 * Tests for:
 * 1. Concurrent requests (subagents must not be blocked by parent)
 * 2. Error recovery (Claude should be able to retry after tool errors)
 * 3. tool_result messages properly converted in follow-up requests
 * 4. Multiple tool calls in a single response
 */

import { describe, it, expect, mock, beforeEach } from "bun:test"
import {
  messageStart,
  textBlockStart,
  toolUseBlockStart,
  textDelta,
  inputJsonDelta,
  blockStop,
  messageDelta,
  messageStop,
  assistantMessage,
  makeRequest,
  makeToolResultRequest,
  parseSSE,
} from "./helpers"

// --- Capture SDK calls ---
let mockMessages: any[] = []
let capturedQueryParams: any = null
let queryCallCount = 0

mock.module("@anthropic-ai/claude-agent-sdk", () => ({
  query: (params: any) => {
    capturedQueryParams = params
    queryCallCount++
    return (async function* () {
      for (const msg of mockMessages) {
        yield msg
      }
    })()
  },
  createSdkMcpServer: () => ({
    type: "sdk",
    name: "test",
    instance: {},
  }),
}))

mock.module("../logger", () => ({
  claudeLog: () => {},
  withClaudeLogContext: (_ctx: any, fn: any) => fn(),
}))

mock.module("../mcpTools", () => ({
  opencodeMcpServer: { type: "sdk", name: "opencode", instance: {} },
}))

const { createProxyServer } = await import("../proxy/server")

function createTestApp() {
  const { app } = createProxyServer({ port: 0, host: "127.0.0.1" })
  return app
}

async function postMessages(app: any, body: Record<string, unknown>) {
  const req = new Request("http://localhost/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  return app.fetch(req)
}

async function readStreamFull(response: Response): Promise<string> {
  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let result = ""
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    result += decoder.decode(value, { stream: true })
  }
  return result
}

// ============================================================
// CONCURRENT REQUESTS
// ============================================================

describe("Phase 3: Concurrent request support", () => {
  beforeEach(() => {
    mockMessages = []
    capturedQueryParams = null
    queryCallCount = 0
  })

  it("should handle concurrent streaming requests without blocking", async () => {
    mockMessages = [
      messageStart(),
      textBlockStart(0),
      textDelta(0, "Response"),
      blockStop(0),
      messageDelta("end_turn"),
      messageStop(),
    ]

    const app = createTestApp()

    // Fire two requests simultaneously (like OpenCode does with main + title gen)
    const [r1, r2] = await Promise.all([
      postMessages(app, makeRequest({ stream: true })),
      postMessages(app, makeRequest({ stream: true })),
    ])

    expect(r1.status).toBe(200)
    expect(r2.status).toBe(200)

    // Both should complete (not block each other)
    const [text1, text2] = await Promise.all([
      readStreamFull(r1),
      readStreamFull(r2),
    ])

    expect(text1).toContain("Response")
    expect(text2).toContain("Response")
  })

  it("should handle concurrent non-streaming requests", async () => {
    mockMessages = [
      assistantMessage([{ type: "text", text: "OK" }]),
    ]

    const app = createTestApp()

    const [r1, r2] = await Promise.all([
      postMessages(app, makeRequest({ stream: false })),
      postMessages(app, makeRequest({ stream: false })),
    ])

    const [b1, b2] = await Promise.all([
      r1.json() as Promise<any>,
      r2.json() as Promise<any>,
    ])

    expect(b1.content[0].text).toBe("OK")
    expect(b2.content[0].text).toBe("OK")
  })

  it("should strip internal orchestration and transcript wrappers from streamed text deltas", async () => {
    mockMessages = [
      messageStart(),
      textBlockStart(0),
      textDelta(0, "Before <system-reminder>ALL BACKGROUND TASKS COMPLETE</system-reminder> [SYSTEM DIRECTIVE: OH-MY-OPENCODE - TODO CONTINUATION] <task_metadata>session_id: ses_x</task_metadata> <thinking>hidden debug</thinking> H: <tool_output for=\"toolu_1\">internal</tool_output> A: <tool_exec name=\"bash\" /> ⚙ background_output [task_id=bg_123] After <!-- OMO_INTERNAL_INITIATOR -->"),
      blockStop(0),
      messageDelta("end_turn"),
      messageStop(),
    ]

    const app = createTestApp()
    const response = await postMessages(app, makeRequest({ stream: true }))
    const text = await readStreamFull(response)
    const events = parseSSE(text)
    const deltas = events
      .filter((e) => e.event === "content_block_delta")
      .map((e) => (e.data as any).delta?.text)
      .filter(Boolean)
      .join("\n")

    expect(deltas).toContain("Before")
    expect(deltas).toContain("After")
    expect(deltas).not.toContain("<system-reminder>")
    expect(deltas).not.toContain("ALL BACKGROUND TASKS COMPLETE")
    expect(deltas).not.toContain("SYSTEM DIRECTIVE: OH-MY-OPENCODE")
    expect(deltas).not.toContain("<task_metadata>")
    expect(deltas).not.toContain("<thinking>")
    expect(deltas).not.toContain("<tool_output")
    expect(deltas).not.toContain("<tool_exec")
    expect(deltas).not.toContain("background_output [task_id=")
    expect(deltas).not.toContain("H:")
    expect(deltas).not.toContain("A:")
    expect(deltas).not.toContain("OMO_INTERNAL_INITIATOR")
  })
})

// ============================================================
// TOOL RESULT HANDLING
// ============================================================

describe("Phase 3: Tool result in follow-up requests", () => {
  beforeEach(() => {
    mockMessages = []
    capturedQueryParams = null
    queryCallCount = 0
  })

  it("should include tool_use blocks from assistant messages in prompt", async () => {
    const unique = crypto.randomUUID()
    mockMessages = [
      assistantMessage([{ type: "text", text: "Here are the contents." }]),
    ]

    const app = createTestApp()
    const response = await postMessages(app, makeRequest({
      stream: false,
      messages: [
        { role: "user", content: `Read test.ts ${unique}` },
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "toolu_1", name: "Read", input: { file_path: "test.ts" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_1", content: "console.log('hello')" },
          ],
        },
      ],
    }))
    await response.json()

    const prompt = capturedQueryParams.prompt
    // Should contain the tool use context
    expect(prompt).toContain("Read")
    expect(prompt).toContain("test.ts")
    // Should contain the tool result
    expect(prompt).toContain("console.log('hello')")
  })

  it("should handle multiple tool results in a single message", async () => {
    mockMessages = [
      assistantMessage([{ type: "text", text: "Both files read." }]),
    ]

    const app = createTestApp()
    const response = await postMessages(app, makeRequest({
      stream: false,
      messages: [
        { role: "user", content: "Read both files" },
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "toolu_a", name: "Read", input: { file_path: "a.ts" } },
            { type: "tool_use", id: "toolu_b", name: "Read", input: { file_path: "b.ts" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_a", content: "file a contents" },
            { type: "tool_result", tool_use_id: "toolu_b", content: "file b contents" },
          ],
        },
      ],
    }))
    await response.json()

    const prompt = capturedQueryParams.prompt
    expect(prompt).toContain("file a contents")
    expect(prompt).toContain("file b contents")
  })

  it("should handle error tool results", async () => {
    mockMessages = [
      assistantMessage([{ type: "text", text: "I see the error, let me try differently." }]),
    ]

    const app = createTestApp()
    const response = await postMessages(app, makeRequest({
      stream: false,
      messages: [
        { role: "user", content: "Do something" },
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "toolu_err", name: "Task", input: { agent_type: "general-purpose", prompt: "read file" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_err", content: "Error: Unknown agent type: general-purpose is not a valid agent type", is_error: true },
          ],
        },
      ],
    }))
    await response.json()

    const prompt = capturedQueryParams.prompt
    expect(prompt).toContain("Unknown agent type")
    expect(prompt).toContain("general-purpose")
  })

  it("should preserve long tool_result context in passthrough mode", async () => {
    const previousPassthrough = process.env.CLAUDE_PROXY_PASSTHROUGH
    const previousSummaryChars = process.env.CLAUDE_PROXY_TOOL_RESULT_SUMMARY_CHARS
    process.env.CLAUDE_PROXY_PASSTHROUGH = "1"
    delete process.env.CLAUDE_PROXY_TOOL_RESULT_SUMMARY_CHARS

    try {
      mockMessages = [
        assistantMessage([{ type: "text", text: "Done reading." }]),
      ]

      const longResult = `${"A".repeat(70_000)}__TAIL_MARKER__`
      const app = createTestApp()
      const response = await postMessages(app, makeRequest({
        stream: false,
        messages: [
          { role: "user", content: "Read long file" },
          {
            role: "assistant",
            content: [
              { type: "tool_use", id: "toolu_long", name: "Read", input: { file_path: "big.ts" } },
            ],
          },
          {
            role: "user",
            content: [
              { type: "tool_result", tool_use_id: "toolu_long", content: longResult },
            ],
          },
        ],
      }))

      await response.json()
      const prompt = capturedQueryParams.prompt
      expect(prompt).toContain("__TAIL_MARKER__")
    } finally {
      if (previousPassthrough === undefined) {
        delete process.env.CLAUDE_PROXY_PASSTHROUGH
      } else {
        process.env.CLAUDE_PROXY_PASSTHROUGH = previousPassthrough
      }

      if (previousSummaryChars === undefined) {
        delete process.env.CLAUDE_PROXY_TOOL_RESULT_SUMMARY_CHARS
      } else {
        process.env.CLAUDE_PROXY_TOOL_RESULT_SUMMARY_CHARS = previousSummaryChars
      }
    }
  })

  it("should strip internal orchestration and transcript wrappers from reconstructed prompt", async () => {
    mockMessages = [
      assistantMessage([{ type: "text", text: "Acknowledged." }]),
    ]

    const app = createTestApp()
    const response = await postMessages(app, makeRequest({
      stream: false,
      messages: [
        {
          role: "assistant",
          content: "Waiting. <system-reminder>ALL BACKGROUND TASKS COMPLETE\nUse background_output</system-reminder> [SYSTEM DIRECTIVE: OH-MY-OPENCODE - TODO CONTINUATION] <task_metadata>session_id: ses_x</task_metadata> <thinking>chain-of-thought</thinking> H: <tool_output for=\"toolu_1\">internal</tool_output> A: <tool_exec name=\"bash\" /> ⚙ background_output [task_id=bg_123] Continue. <!-- OMO_INTERNAL_INITIATOR -->",
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu_bg",
              content: "H: <system-reminder>ALL BACKGROUND TASKS COMPLETE</system-reminder> [SYSTEM DIRECTIVE: OH-MY-OPENCODE - ULTRAWORK LOOP VERIFICATION 1/1] <task_metadata>task_id: ses_x</task_metadata> <thinking>loop</thinking> H: <tool_output for=\"toolu_2\">trace</tool_output> A: <tool_exec name=\"task\" /> A: ⚙ background_output [task_id=bg_123] <!-- OMO_INTERNAL_INITIATOR -->",
            },
          ],
        },
      ],
    }))
    await response.json()

    const prompt = String(capturedQueryParams.prompt)
    expect(prompt).not.toContain("<system-reminder>")
    expect(prompt).not.toContain("ALL BACKGROUND TASKS COMPLETE")
    expect(prompt).not.toContain("SYSTEM DIRECTIVE: OH-MY-OPENCODE")
    expect(prompt).not.toContain("<task_metadata>")
    expect(prompt).not.toContain("<thinking>")
    expect(prompt).not.toContain("<tool_output for=\"toolu_2\"")
    expect(prompt).not.toContain("<tool_exec")
    expect(prompt).not.toContain("background_output [task_id=")
    expect(prompt).not.toContain("H:")
    expect(prompt).not.toContain("A:")
    expect(prompt).not.toContain("OMO_INTERNAL_INITIATOR")
    expect(prompt).toContain("Waiting.")
    expect(prompt).toContain("Continue.")
  })

  it("should preserve OpenCode-style tool markers in passthrough mode", async () => {
    mockMessages = [
      assistantMessage([{ type: "text", text: "Done." }]),
    ]

    const app = createTestApp()
    const response = await postMessages(app, makeRequest({
      stream: false,
      messages: [
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "toolu_1", name: "mcp__oc__todowrite", input: { todos: [] } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_1", content: "ok" },
          ],
        },
      ],
    }))
    await response.json()

    const prompt = String(capturedQueryParams.prompt)
    expect(prompt).toContain("mcp__oc__todowrite")
  })
})

// ============================================================
// MULTIPLE TOOL CALLS IN SINGLE RESPONSE
// ============================================================

describe("Phase 3: Multiple tool calls in response", () => {
  beforeEach(() => {
    mockMessages = []
    capturedQueryParams = null
  })

  it("should forward multiple tool_use blocks in streaming", async () => {
    mockMessages = [
      messageStart(),
      // First tool call
      toolUseBlockStart(0, "Read", "toolu_r1"),
      inputJsonDelta(0, '{"file_path":"a.ts"}'),
      blockStop(0),
      // Second tool call
      toolUseBlockStart(1, "Read", "toolu_r2"),
      inputJsonDelta(1, '{"file_path":"b.ts"}'),
      blockStop(1),
      messageDelta("tool_use"),
      messageStop(),
    ]

    const app = createTestApp()
    const response = await postMessages(app, makeRequest({ stream: true }))
    const text = await readStreamFull(response)
    const events = parseSSE(text)

    const blockStarts = events.filter((e) => e.event === "content_block_start")
    const toolStarts = blockStarts.filter((e) => (e.data as any).content_block?.type === "tool_use")
    expect(toolStarts.length).toBe(2)
    expect((toolStarts[0]?.data as any).content_block.name).toBe("Read")
    expect((toolStarts[1]?.data as any).content_block.name).toBe("Read")
  })

  it("should include multiple tool_use blocks in non-streaming", async () => {
    mockMessages = [
      assistantMessage([
        { type: "tool_use", id: "toolu_m1", name: "Read", input: { file_path: "a.ts" } },
        { type: "tool_use", id: "toolu_m2", name: "Bash", input: { command: "ls" } },
      ]),
    ]

    const app = createTestApp()
    const response = await postMessages(app, makeRequest({ stream: false }))
    const body = await response.json() as any

    expect(body.content.length).toBe(2)
    expect(body.content[0].type).toBe("tool_use")
    expect(body.content[0].name).toBe("Read")
    expect(body.content[1].type).toBe("tool_use")
    expect(body.content[1].name).toBe("Bash")
    expect(body.stop_reason).toBe("tool_use")
  })
})
