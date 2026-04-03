# RAG System Technical Design

A reusable design reference for building retrieval-augmented generation systems over document corpora.

---

## Core constraint: single LLM call for answer generation

The final answer must come from one LLM API request. Retrieval and preprocessing happen before that call; query expansion happens inside it via tool-calling.

**Why it matters:**
- Each additional LLM call adds 500ms–2s of sequential latency
- Two-call chains compound prompt drift — errors in call 1 corrupt call 2
- A single call is easier to audit, reproduce, and iterate on
- GPT-4o can reason about what to search for as part of its tool-calling step; a separate expansion call adds cost without adding capability

**Pattern: query expansion via tool input, not a separate call**

Instead of calling the LLM twice (once to expand the query, once to answer), define a `getInformation` tool whose input schema asks the model to provide multiple search queries:

```ts
getInformation: tool({
  inputSchema: z.object({
    question: z.string(),
    similarQuestions: z.array(z.string()).describe(
      "3–5 search queries targeting different aspects: entity names, metrics, categories, time periods"
    ),
  }),
  execute: async ({ similarQuestions }) => {
    const results = await Promise.all(similarQuestions.map(q => vectorSearch(q)));
    return deduplicate(results.flat());
  },
})
```

The model reasons about what to search for as part of generating its tool call arguments — within the same API request.


this query uses approximate nearest neighbor (ANN) via the HNSW
   index, but the SQL itself is doing an exhaustive similarity scan with a
  threshold filter, not a pure k-NN query.

  Here's the distinction:

  What the query actually does

  1. Computes cosine similarity between the query embedding and every row in
  the table
  2. Filters to rows where similarity > 0.25
  3. Orders by similarity descending
  4. Returns the top 16

  The WHERE gt(similarity, 0.25) is a post-computation filter, not a
  nearest-neighbor lookup. It doesn't say "give me the 16 closest vectors" — it
   says "give me all vectors above a threshold, then take the top 16."

  Where HNSW comes in

  The HNSW index (vector_cosine_ops) accelerates the distance computation step.
   Without it, Postgres would do a full sequential scan comparing the query
  vector against every row. With it, Postgres uses the approximate graph
  traversal to find candidates quickly. So the algorithm running underneath is
  ANN — but the query semantics are threshold-filtered top-K, not strict k-NN.

  The practical difference

  ┌──────────────────────────────────────┬─────────────────────────────────┐
  │  Pure k-NN (ORDER BY ... LIMIT 16)   │   Threshold + top-K (current)   │
  ├──────────────────────────────────────┼─────────────────────────────────┤
  │ Always returns exactly 16 results    │ May return fewer than 16 if     │
  │                                      │ corpus is sparse                │
  ├──────────────────────────────────────┼─────────────────────────────────┤
  │ Can return irrelevant results if     │ Filters out noise below         │
  │ top-16 are all low-similarity        │ threshold                       │
  ├──────────────────────────────────────┼─────────────────────────────────┤
  │ Better when you always need N        │ Better when result quality      │
  │ results                              │ matters more than result count  │
  └──────────────────────────────────────┴─────────────────────────────────┘

  For a RAG system, the threshold filter is the right call — you don't want to
  inject low-relevance chunks into the prompt just to hit a count. A pure LIMIT
   16 without a threshold would return 16 results even if the best match has
  similarity 0.1, which would pollute the context sent to GPT-4o.

  If you wanted strict k-NN

  pgvector supports it with the <=> operator directly:

  SELECT content FROM embeddings
  ORDER BY embedding <=> '[query_vector]'
  LIMIT 16;

  This is pure ANN (no threshold), uses the HNSW index optimally, and always
  returns exactly 16 rows. You'd then filter by quality in application code
  after the fact.
---

## Ingestion pipeline

### Text extraction

Raw documents often contain noise (metadata headers, XBRL markup, boilerplate). Extract only the readable content before chunking:

```ts
function extractReadableText(raw: string): string {
  const startIdx = raw.indexOf("KNOWN_CONTENT_MARKER");
  if (startIdx !== -1) return raw.slice(startIdx);
  // fallback: skip header lines up to a known separator
  const lines = raw.split("\n");
  const headerEnd = lines.findIndex(l => l.startsWith("====="));
  return lines.slice(headerEnd + 1).join("\n");
}
```

Adapt `KNOWN_CONTENT_MARKER` to your document format.

### Chunking strategy

Chunk on natural boundaries (paragraphs > sentences > fixed size) with overlap:

```
CHUNK_SIZE    = 800 chars   — small enough for focused retrieval, large enough for context
CHUNK_OVERLAP = 100 chars   — prevents sentences from being cut across chunk boundaries
MIN_PARA_LEN  = 40 chars    — skip near-empty paragraphs
```

**Algorithm:**
1. Split on `\n\n+` (paragraph boundaries)
2. Accumulate paragraphs into a buffer until it exceeds `CHUNK_SIZE`
3. If a single paragraph exceeds `CHUNK_SIZE`, split by sentence
4. After building chunks, prepend the last `CHUNK_OVERLAP` chars of chunk N-1 to chunk N

This preserves more semantic context than fixed-width splitting.

### Embedding

Batch embed chunks to minimize API calls:

```ts
const EMBED_BATCH_SIZE = 96; // OpenAI max is 2048; keep batches conservative
const CONCURRENCY = 3;       // files processed in parallel

for (let i = 0; i < chunks.length; i += EMBED_BATCH_SIZE) {
  const batch = chunks.slice(i, i + EMBED_BATCH_SIZE);
  const { embeddings } = await embedMany({ model, values: batch });
  // store embeddings...
}
```

Use `embedMany` (not `embed` in a loop) — one API call per batch instead of one per chunk.

### Concurrency

Process files in parallel with a bounded worker pool:

```ts
async function runWithConcurrency<T>(
  items: T[],
  concurrency: number,
  fn: (item: T) => Promise<void>
): Promise<void> {
  let idx = 0;
  async function worker() {
    while (idx < items.length) await fn(items[idx++]);
  }
  await Promise.all(Array.from({ length: concurrency }, worker));
}
```

`CONCURRENCY = 3` balances throughput against API rate limits. Tune based on your embedding API tier.

---

## Vector storage schema

Minimum schema for a pgvector-backed RAG store:

```sql
-- resources: one row per source document
CREATE TABLE resources (
  id         VARCHAR(191) PRIMARY KEY,
  content    TEXT NOT NULL,         -- filename or summary; used for citation
  created_at TIMESTAMP DEFAULT NOW()
);

-- embeddings: one row per chunk
CREATE TABLE embeddings (
  id          VARCHAR(191) PRIMARY KEY,
  resource_id VARCHAR(191) REFERENCES resources(id) ON DELETE CASCADE,
  content     TEXT NOT NULL,        -- the actual chunk text injected into prompts
  embedding   VECTOR(1536) NOT NULL
);

-- HNSW index for fast approximate nearest-neighbor search
CREATE INDEX embedding_idx ON embeddings
  USING hnsw (embedding vector_cosine_ops);
```

**HNSW vs. IVFFlat:**
- HNSW: faster queries, higher memory, better recall — preferred for interactive use
- IVFFlat: lower memory, requires `ANALYZE` after bulk inserts — better for very large corpora

### Retrieval query

```ts
const similarity = sql<number>`1 - (${cosineDistance(embeddings.embedding, queryEmbedding)})`;

const results = await db
  .select({ content: embeddings.content, similarity })
  .from(embeddings)
  .where(gt(similarity, 0.25))   // threshold: filter out low-relevance noise
  .orderBy(desc(similarity))
  .limit(16);                     // per query; multiply by number of queries for total context
```

### Retrieval algorithm: ANN with threshold-filtered top-K

This query is **not** pure k-nearest-neighbor. It uses two mechanisms in combination:

**1. Approximate nearest neighbor (ANN) via HNSW index**

The HNSW index accelerates distance computation using graph traversal — it does not scan every row. Without the index, Postgres would do a full sequential scan comparing the query vector against every embedding. With it, candidates are found in O(log n) time. This is *approximate* because HNSW may occasionally miss a slightly better match in exchange for dramatically faster queries.

**2. Threshold-filtered top-K (the query semantics)**

The `WHERE gt(similarity, 0.25)` filter and `LIMIT 16` operate on top of the ANN candidates. This is *not* the same as pure k-NN:

| Approach | Behavior |
|---|---|
| Pure k-NN: `ORDER BY embedding <=> query LIMIT 16` | Always returns exactly 16 results, even if all have low similarity |
| Threshold + top-K (current) | Returns up to 16 results, but only those above the similarity threshold |

For RAG specifically, the threshold matters: you do not want to inject low-relevance chunks into the prompt just to hit a count. A pure `LIMIT 16` without a threshold would return 16 results even if the best match scores 0.1, polluting the context sent to the LLM.

**Pure k-NN syntax (pgvector `<=>` operator):**

```ts
// Strict ANN — always returns exactly N results, no threshold
const results = await db.execute(
  sql`SELECT content FROM embeddings ORDER BY embedding <=> ${queryEmbedding} LIMIT 16`
);
```

Use pure k-NN when you always need a fixed number of results (e.g., a recommendation system). Use threshold-filtered top-K when result quality matters more than result count (e.g., RAG context injection).

**Tuning thresholds:**

| Threshold | Effect |
|---|---|
| 0.4+ | High precision, may miss relevant chunks |
| 0.25–0.35 | Balanced; good default for factual corpora |
| <0.2 | High recall, more noise injected into context |

**Tuning limit per query:**

For single-entity questions, 4–8 chunks is sufficient. For multi-entity comparisons, use 12–16 per query across 3–5 queries. More queries × higher limit = more context, but watch GPT-4o's context window (~128k tokens for gpt-4o).

---

## System prompt design

### Structure

```
[Role + domain framing]
[How to use the retrieval tool — what queries to generate]
[Answer format rules — structure, citation, completeness]
[Fallback rule — what to say when retrieval returns nothing]
```

### Key principles

**Be explicit about query generation.** Tell the model what dimensions to search across — entity names, metric types, time periods, document types. Without this, the model generates generic queries and misses relevant chunks.

**Don't cap response length for analytical questions.** "Answer in one sentence" is appropriate for a personal assistant; it's wrong for a financial comparison. Match length guidance to the query type your system handles.

**Ground the model with a hard fallback.** Without an explicit fallback, models will hallucinate when retrieval fails. Use:
```
If no relevant information is found in the retrieved context, say:
"I couldn't find relevant information in the indexed documents for that question."
```

**Cite sources in the prompt, not just in post-processing.** Instruct the model to reference the source document in its answer (e.g., "per Apple's FY2024 10-K"). The chunk content usually contains enough context for the model to infer this.

### Template

```
You are a [domain expert role] with expertise in [document type].

Your knowledge base contains [description of corpus]. Always call getInformation before answering.

## How to use getInformation
Pass "similarQuestions" as 3–5 targeted queries covering:
- [entity dimension: company names, product names, etc.]
- [metric dimension: financial figures, KPIs, etc.]
- [category dimension: risk types, regulatory areas, etc.]
- [time dimension: fiscal years, quarters, etc.]

## Answer guidelines
- Only use information from tool call results. Do not hallucinate.
- If no relevant information is found, say: "[your fallback message]"
- [Format instructions tailored to your domain]
- [Citation format]
- [Length/structure guidance]
```

---

## API route pattern (Vercel AI SDK)

```ts
export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: "openai/gpt-4o",
    messages: convertToModelMessages(messages),
    system: SYSTEM_PROMPT,
    stopWhen: stepCountIs(5),   // prevent runaway tool loops
    tools: { getInformation },
  });

  return result.toUIMessageStreamResponse();
}
```

`stopWhen: stepCountIs(5)` is a safety valve — the model should only need 1–2 tool steps (retrieve, then answer), but this prevents infinite loops if tool outputs are ambiguous.

---

## Quality evaluation checklist

Before shipping or demoing a RAG system, verify:

- [ ] **Retrieval coverage** — For a multi-entity question, do retrieved chunks span all mentioned entities? Log raw retrieval results during development.
- [ ] **No hallucination on missing data** — Ask about a company/topic not in the corpus. The system should trigger the fallback, not invent an answer.
- [ ] **Single-call compliance** — Confirm the answer is produced in one LLM request. Check for any nested `generateObject` / `generateText` calls inside tool `execute` functions.
- [ ] **Context window headroom** — For your largest expected query (5 queries × 16 chunks × ~200 tokens/chunk ≈ 16k tokens), plus system prompt, you should be well under the model's limit.
- [ ] **Threshold calibration** — Run 10 representative queries and check whether retrieved chunks are relevant. Adjust similarity threshold and limit if needed.
- [ ] **Chunk boundary coherence** — Spot-check a few chunks to ensure sentences aren't cut mid-thought. Tune overlap if they are.

---

## Known limitations and mitigations

| Limitation | Mitigation |
|---|---|
| Chunk metadata not stored (company, date, filing type) | Add structured metadata columns to `embeddings` table; filter by metadata at retrieval time |
| Very long comparative answers approach context limits | Re-rank or summarize chunks before injection; reduce limit per query |
| Cosine similarity is lexically blind | Use hybrid search: combine vector similarity with BM25 full-text search |
| Ingestion re-embeds duplicate files | Track ingested file hashes in the `resources` table; skip on re-run |
| No source attribution in UI | Return `resourceId` alongside chunk content; display filing name in the response |
