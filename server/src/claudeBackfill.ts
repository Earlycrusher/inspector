import { Anthropic } from "@anthropic-ai/sdk";
import {
  CancelledNotification,
  isInitializeRequest,
  isJSONRPCRequest,
  ElicitRequestSchema,
  CreateMessageRequest,
  CreateMessageRequestSchema,
  CreateMessageResult,
  JSONRPCResponse,
  LoggingMessageNotification,
  JSONRPCNotification,
  isInitializedNotification,
  JSONRPCMessage,
} from "@modelcontextprotocol/sdk/types.js";
import {
  BetaBase64ImageSource,
  BetaContentBlock,
  BetaContentBlockParam,
  BetaMessageParam,
} from "@anthropic-ai/sdk/resources/beta.js";
import { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import { CancelledNotificationSchema } from "@modelcontextprotocol/sdk/types.js";

const isCancelledNotification: (value: unknown) => value is CancelledNotification =
  ((value: any) => CancelledNotificationSchema.safeParse(value).success) as any;

const isElicitRequest: (value: unknown) => value is any =
  ((value: any) => ElicitRequestSchema.safeParse(value).success) as any;

const isCreateMessageRequest: (value: unknown) => value is CreateMessageRequest =
  ((value: any) => CreateMessageRequestSchema.safeParse(value).success) as any;

function contentToMcp(content: BetaContentBlock): CreateMessageResult["content"][number] {
  switch (content.type) {
    case "text":
      return { type: "text", text: content.text };
    default:
      throw new Error(`Unsupported content type: ${content.type}`);
  }
}

function contentFromMcp(
  content: CreateMessageRequest["params"]["messages"][number]["content"]
): BetaContentBlockParam {
  switch (content.type) {
    case "text":
      return { type: "text", text: content.text };
    case "image":
      return {
        type: "image",
        source: {
          data: content.data,
          media_type: content.mimeType as BetaBase64ImageSource["media_type"],
          type: "base64",
        },
      };
    case "audio":
    default:
      throw new Error(`Unsupported content type: ${content.type}`);
  }
}

export interface ClaudeBackfillOptions {
  apiKey?: string;
  enabled?: boolean;
}

export class ClaudeBackfillService {
  private api: Anthropic | null = null;
  private samplingMeta: {
    sampling_models: string[];
    sampling_default_model: string;
  } | null = null;
  private enabled: boolean;

  constructor(options: ClaudeBackfillOptions = {}) {
    this.enabled = options.enabled ?? true;

    if (this.enabled && (options.apiKey || process.env.ANTHROPIC_API_KEY)) {
      try {
        this.api = new Anthropic({
          apiKey: options.apiKey || process.env.ANTHROPIC_API_KEY,
        });
        this.initializeMeta();
      } catch (error) {
        console.warn("Failed to initialize Claude API:", error);
        this.enabled = false;
      }
    } else {
      this.enabled = false;
    }
  }

  private async initializeMeta() {
    if (!this.api) return;

    try {
      const models = new Set<string>();
      let defaultModel: string | undefined;

      for await (const info of this.api.beta.models.list()) {
        models.add(info.id);
        if (info.id.indexOf("sonnet") >= 0 && defaultModel === undefined) {
          defaultModel = info.id;
        }
      }

      if (defaultModel === undefined) {
        if (models.size === 0) {
          throw new Error("No models available from the API");
        }
        defaultModel = models.values().next().value!;
      }

      this.samplingMeta = {
        sampling_models: Array.from(models),
        sampling_default_model: defaultModel,
      };
    } catch (error) {
      console.warn("Failed to initialize Claude model metadata:", error);
      this.enabled = false;
    }
  }

  private pickModel(
    preferences: CreateMessageRequest["params"]["modelPreferences"] | undefined
  ): string {
    if (!this.samplingMeta) {
      throw new Error("Claude backfill service not properly initialized");
    }

    if (preferences?.hints) {
      for (const hint of Object.values(preferences.hints)) {
        if (hint.name !== undefined && this.samplingMeta.sampling_models.includes(hint.name)) {
          return hint.name;
        }
      }
    }
    // TODO: linear model on preferences?.{intelligencePriority, speedPriority, costPriority} to pick between haiku, sonnet, opus.
    return this.samplingMeta.sampling_default_model!;
  }

  public isEnabled(): boolean {
    return this.enabled && this.api !== null && this.samplingMeta !== null;
  }

  public getSamplingMeta() {
    return this.samplingMeta;
  }

  public async handleCreateMessageRequest(
    request: CreateMessageRequest
  ): Promise<CreateMessageResult> {
    if (!this.api || !this.samplingMeta) {
      throw new Error("Claude backfill service not available");
    }

    if (request.params.includeContext !== "none") {
      throw new Error("includeContext != none not supported by MCP sampling backfill");
    }

    const msg = await this.api.beta.messages.create({
      model: this.pickModel(request.params.modelPreferences),
      system:
        request.params.systemPrompt === undefined
          ? undefined
          : [
              {
                type: "text",
                text: request.params.systemPrompt,
              },
            ],
      messages: request.params.messages.map(({ role, content }) => ({
        role,
        content: [contentFromMcp(content)],
      } as BetaMessageParam)),
      max_tokens: request.params.maxTokens,
      temperature: request.params.temperature,
      stop_sequences: request.params.stopSequences,
    });

    if (msg.content.length !== 1) {
      throw new Error(`Expected exactly one content item in the response, got ${msg.content.length}`);
    }

    return {
      model: msg.model,
      stopReason: msg.stop_reason ?? undefined,
      role: msg.role,
      content: contentToMcp(msg.content[0]) as any,
    };
  }
}

export function createClaudeBackfillProxy(
  transportToClient: Transport,
  transportToServer: Transport,
  backfillService: ClaudeBackfillService
) {
  let clientSupportsSampling: boolean | undefined;

  // Intercept messages from client to server
  transportToClient.onmessage = async (message, extra) => {
    if (isJSONRPCRequest(message)) {
      if (isInitializeRequest(message)) {
        // Check if client already supports sampling
        clientSupportsSampling = !!message.params.capabilities.sampling;

        if (!clientSupportsSampling && backfillService.isEnabled()) {
          // Add sampling capabilities and metadata if backfill is enabled
          message.params.capabilities.sampling = {};
          const meta = backfillService.getSamplingMeta();
          if (meta) {
            message.params._meta = { ...(message.params._meta ?? {}), ...meta };
          }
        }
      } else if (isCreateMessageRequest(message) && !clientSupportsSampling && backfillService.isEnabled()) {
        // Intercept and handle CreateMessageRequest with Claude API
        try {
          const result = await backfillService.handleCreateMessageRequest(message);

          const response: JSONRPCResponse = {
            jsonrpc: "2.0",
            id: message.id,
            result,
          };

          transportToClient.send(response, { relatedRequestId: message.id });
          return; // Don't forward to server
        } catch (error) {
          const errorResponse = {
            jsonrpc: "2.0" as const,
            id: message.id,
            error: {
              code: -32001,
              message: `Error processing message: ${(error as Error).message}`,
            },
          };
          transportToClient.send(errorResponse, { relatedRequestId: message.id });
          return; // Don't forward to server
        }
      }
    }

    // Forward message to server
    try {
      await transportToServer.send(message);
    } catch (error) {
      // Send error response back to client if it was a request (has id)
      if (isJSONRPCRequest(message)) {
        const errorResponse = {
          jsonrpc: "2.0" as const,
          id: message.id,
          error: {
            code: -32001,
            message: (error as Error).message,
            data: error,
          },
        };
        transportToClient.send(errorResponse).catch(console.error);
      }
    }
  };

  // Intercept messages from server to client
  transportToServer.onmessage = async (message, extra) => {
    if (isJSONRPCRequest(message) && isInitializedNotification(message)) {
      if (!clientSupportsSampling && backfillService.isEnabled()) {
        // Add metadata to initialized notification
        const meta = backfillService.getSamplingMeta();
        if (meta) {
          message.params = { ...(message.params ?? {}), _meta: { ...(message.params?._meta ?? {}), ...meta } };
        }
      }
    }

    // Forward message to client
    try {
      await transportToClient.send(message);
    } catch (error) {
      console.error("Error forwarding message to client:", error);
    }
  };

  return {
    transportToClient,
    transportToServer,
  };
}