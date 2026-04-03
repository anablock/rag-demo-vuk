# Prompt Iteration Log

Documents the evolution of the system prompt and retrieval design for the SEC filings RAG assessment.

---

## Iteration 1 — Original template prompt

**Prompt:**
```
You are a helpful assistant acting as the users' second brain.
Use tools on every request.
Be sure to getInformation from your knowledge base before answering any questions.
If the user presents information about themselves, use the addResource tool to store it.
...
Keep responses short and concise. Answer in a single sentence where possible.
```

**Tools:** `addResource`, `getInformation`, `understandQuery`

The `understandQuery` tool called `generateObject` internally (a second LLM request) to expand the user's query into similar questions before retrieval.

**Problems identified:**

1. **Violated the single-call constraint.** `understandQuery` made a nested `generateObject` call to GPT-4o, meaning every user question triggered at least two LLM API calls before the final answer.
2. **Generic framing.** "Second brain" framing gives no guidance about financial filings, citation, structure, or comparative analysis.
3. **"Answer in a single sentence"** — directly counterproductive for multi-company comparisons like "compare Apple, Tesla, and JPMorgan risk factors."
4. **Retrieval too narrow.** `findRelevantContent` returned only 4 chunks with similarity > 0.3. Multi-company questions spanning 3 companies can't be answered from 4 chunks.

---

## Iteration 2 — Fix single-call constraint, refocus prompt

**Change:** Removed `understandQuery` tool entirely. Moved query expansion responsibility into the `getInformation` tool's input schema: the model is asked to provide `similarQuestions` (3–5 search queries) when it calls `getInformation`. This means query expansion happens as part of the model's single reasoning pass — no extra LLM call.

**Why this works:** GPT-4o's tool-calling step is still within the same `streamText` call. The model generates search terms as part of deciding what arguments to pass to `getInformation`, not via a separate generation step.

**Retrieval changes:**
- Limit raised from 4 → 16 results per query
- Similarity threshold lowered from 0.3 → 0.25
- Because the model now passes 3–5 queries, total retrieved context can be up to 80 chunks before deduplication — enough to cover multiple companies

**Prompt rewrite rationale:**

| Old behavior | New behavior |
|---|---|
| "Second brain" framing | Financial analyst with SEC filing expertise |
| "Answer in one sentence" | Structured answer with headings for multi-company questions |
| No citation guidance | Explicitly told to cite filing type and company |
| No comparison format | Instructed to highlight similarities and differences |
| No handling for missing data | Explicit fallback: "I couldn't find relevant information…" |
| `similarQuestions` used as keywords | `similarQuestions` framed as targeted retrieval queries covering companies, metrics, risk types, time periods |

**Final system prompt:**
```
You are a financial analyst assistant with deep expertise in SEC filings (10-K annual reports
and 10-Q quarterly reports) from major US public companies spanning 2023–2025.

Your knowledge base contains extracted text from EDGAR filings. Always call getInformation
before answering any question.

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
- If no relevant information is found, say: "I couldn't find relevant information in the
  indexed filings for that question."
- For multi-company questions, organize your answer by company with clear headings.
- Cite the source where possible (e.g. "per Apple's FY2024 10-K…").
- For comparisons, highlight both similarities and key differences.
- For financial metrics, include specific numbers when available in the retrieved context.
- Structure longer answers with bullet points or sections for readability.
- Be thorough — do not truncate important information just to be brief.
```

---

## Quality evaluation

**What I checked:**

- **Relevance:** Does the retrieved context actually contain information about the companies and topics in the question? Verified by logging retrieved chunks during development.
- **Completeness for multi-company queries:** With 5 search queries × 16 results = up to 80 chunks retrieved (before dedup), comparative questions get enough context from multiple filings.
- **Hallucination prevention:** The prompt and tool structure ground the model — it must call `getInformation` and can only use what it retrieves. The explicit fallback ("I couldn't find...") prevents the model from confabulating if retrieval fails.
- **Citation quality:** Chunks retain enough context (800 chars with overlap) that the model can identify the source company and filing type from the text itself.
- **Single-call compliance:** Verified by removing `generateObject` from the tool chain. The only LLM call that produces a user-visible answer is the top-level `streamText` call.

**Known limitations:**
- Chunk metadata doesn't store company name / filing type / date as structured fields — the model infers these from the chunk text. Adding metadata would improve citation accuracy.
- Very long comparative answers may approach GPT-4o's context window if many chunks are retrieved. Could mitigate by re-ranking or summarizing chunks before injection.
