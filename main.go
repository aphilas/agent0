package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprint(os.Stderr, "error:", err)
	}
}

func run() error {
	ctx := context.Background()

	client := openai.NewClient(
		option.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
		option.WithBaseURL("https://openrouter.ai/api/v1"),
	)

	var messages = []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("You are a concise assistant that answers in one line."),
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

		messages = append(messages, openai.UserMessage(line))

		assistant, err := chatCompletion(ctx, &client, messages)
		if err != nil {
			return err
		}

		fmt.Println(assistant)
		messages = append(messages, openai.AssistantMessage(assistant))
	}
}

// chatCompletion calls the chat completion api with the given messages and
// returns the assistant's reply.
func chatCompletion(
	ctx context.Context,
	client *openai.Client,
	messages []openai.ChatCompletionMessageParamUnion,
) (string, error) {
	const modelDeepSeek = "deepseek/deepseek-v3.2-exp"

	chatCompletion, err := client.Chat.Completions.New(
		ctx,
		openai.ChatCompletionNewParams{
			Messages: messages,
			Model:    modelDeepSeek,
		})
	if err != nil {
		return "", fmt.Errorf("running chat completion: %w", err)
	}

	assistant := chatCompletion.Choices[0].Message.Content

	return assistant, nil
}
