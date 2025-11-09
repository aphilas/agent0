package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/sashabaranov/go-openai"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprint(os.Stderr, "error:", err)
	}
}

func run() error {
	ctx := context.Background()

	config := openai.DefaultConfig(os.Getenv("OPENROUTER_API_KEY"))
	config.BaseURL = "https://openrouter.ai/api/v1"
	client := openai.NewClientWithConfig(config)

	var messages = []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a concise assistant that answers in one line.",
		},
	}

	for {
		fmt.Print("> ")

		in := bufio.NewReader(os.Stdin)
		line, err := in.ReadString('\n')
		if err != nil {
			return fmt.Errorf("reading message: %w", err)
		}

		line = strings.TrimSpace(line)
		if line == "exit" {
			return nil
		}

		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: line,
		})

		assistant, err := chatCompletion(ctx, client, messages)
		if err != nil {
			return err
		}

		fmt.Println(assistant)
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: assistant,
		})
	}
}

// chatCompletion calls the chat completion api with the given messages and
// returns the assistant's reply.
func chatCompletion(
	ctx context.Context,
	client *openai.Client,
	messages []openai.ChatCompletionMessage,
) (string, error) {
	const (
		modelDeepSeek = "deepseek/deepseek-v3.2-exp"
		toolBash      = "bash"

		toolSchema = `{
	"type": "object",
	"properties": {
		"command": {
			"type": "string",
			"description": "The bash command to execute."
		}
	},
	"required": [
		"command"
	]
}`
	)

	type Parameters struct {
		Command string `json:"command"`
	}

	t := openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        toolBash,
			Description: "Execute bash commands on a linux shell.",
			Parameters:  json.RawMessage(toolSchema),
		},
	}

	params := openai.ChatCompletionRequest{
		Messages: messages,
		Model:    modelDeepSeek,
		Tools:    []openai.Tool{t},
	}

	resp, err := client.CreateChatCompletion(ctx, params)
	if err != nil {
		return "", fmt.Errorf("running chat completion: %w", err)
	}

	// Append the assistant's message and tool call outputs to the conversation.
	params.Messages = append(params.Messages, openai.ChatCompletionMessage{
		Role:      openai.ChatMessageRoleAssistant,
		ToolCalls: resp.Choices[0].Message.ToolCalls,
	})

	toolCalls := resp.Choices[0].Message.ToolCalls
	for _, toolCall := range toolCalls {
		if toolCall.Function.Name == toolBash {
			var parameters Parameters
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &parameters); err != nil {
				return "", fmt.Errorf("unmarshaling bash tool arguments: %w", err)
			}

			output, err := bash(parameters.Command)
			if err != nil {
				return "", fmt.Errorf("executing bash tool: %w", err)
			}

			params.Messages = append(params.Messages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    output,
				Name:       toolCall.Function.Name,
				ToolCallID: toolCall.ID,
			})
		}
	}

	if len(toolCalls) > 0 {
		// Re-run the chat completion with the tool outputs.
		resp, err = client.CreateChatCompletion(ctx, params)
		if err != nil {
			return "", fmt.Errorf("running chat completion with tool outputs: %w", err)
		}
	}

	assistant := resp.Choices[0].Message.Content

	return assistant, nil
}

// bash executes the given command in a bash shell and returns the output.
func bash(command string) (string, error) {
	cmd := exec.Command("bash", "-c", command)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("executing bash command: %w", err)
	}

	return string(output), nil
}
