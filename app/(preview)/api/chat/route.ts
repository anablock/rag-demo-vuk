import { createResource } from "@/lib/actions/resources";
import { findRelevantContent } from "@/lib/ai/embedding";
import {
  convertToModelMessages,
  stepCountIs,
  streamText,
  tool,
  UIMessage,
} from "ai";
import { z } from "zod";

// Allow streaming responses up to 30 seconds
export const maxDuration = 60;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: "openai/gpt-4o",
    messages: convertToModelMessages(messages),
    system: `You are a financial analyst assistant with deep expertise in SEC filings (10-K annual reports and 10-Q quarterly reports) from major US public companies spanning 2023–2025.

Your knowledge base contains extracted text from EDGAR filings. Always call getInformation before answering any question.

## How to use getInformation
When you call getInformation, pass:
- "question": the user's original question verbatim
- "similarQuestions": 3–5 search queries that will help retrieve relevant chunks. Think about:
  - Company names and ticker symbols mentioned
  - Financial metrics (revenue, earnings, guidance, growth)
  - Risk categories (regulatory, competitive, macroeconomic, operational)
  - Time periods (fiscal year, quarter)
  - Filing type keywords (10-K, 10-Q, annual report, quarterly report)

## Answering guidelines
- ONLY use information retrieved from tool calls. Do not hallucinate filing content.
- If no relevant information is found, say: "I couldn't find relevant information in the indexed filings for that question."
- For multi-company questions, organize your answer by company with clear headings.
- Cite the source where possible (e.g. "per Apple's FY2024 10-K…").
- For comparisons, highlight both similarities and key differences.
- For financial metrics, include specific numbers when available in the retrieved context.
- Structure longer answers with bullet points or sections for readability.
- Be thorough — do not truncate important information just to be brief.
`,
    stopWhen: stepCountIs(5),
    tools: {
      addResource: tool({
        description: `Add a resource to the knowledge base. Use if the user explicitly provides new information to store.`,
        inputSchema: z.object({
          content: z
            .string()
            .describe("the content or resource to add to the knowledge base"),
        }),
        execute: async ({ content }) => createResource({ content }),
      }),
      getInformation: tool({
        description: `Retrieve relevant passages from the SEC filings knowledge base. Always call this before answering any financial or company-related question. Pass the original question and several targeted search queries to maximize recall across companies and topics.`,
        inputSchema: z.object({
          question: z.string().describe("the user's original question"),
          similarQuestions: z
            .array(z.string())
            .describe(
              "3–5 search queries targeting different aspects of the question: company names, metrics, risk types, time periods",
            ),
        }),
        execute: async ({ similarQuestions }) => {
          const results = await Promise.all(
            similarQuestions.map((q) => findRelevantContent(q)),
          );
          // Flatten and deduplicate by content
          const uniqueResults = Array.from(
            new Map(results.flat().map((item) => [item?.name, item])).values(),
          );
          return uniqueResults;
        },
      }),
    },
  });

  return result.toUIMessageStreamResponse();
}
