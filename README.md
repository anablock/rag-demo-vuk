# SEC Filings RAG — AI Engineering Assessment

A retrieval-augmented generation system that answers business questions about SEC EDGAR filings (10-K and 10-Q) from major US public companies (2023–2025). Built with Next.js, Vercel AI SDK, PostgreSQL + pgvector, and OpenAI GPT-4o.

---

## How it works

1. **Ingestion** — `.txt` filings from EDGAR are chunked (800 chars, 100-char overlap), embedded with `text-embedding-ada-002`, and stored in PostgreSQL with a pgvector HNSW index.
2. **Retrieval** — At query time, the model generates 3–5 targeted search queries as part of its `getInformation` tool call. Each is embedded and run against the vector index (cosine similarity ≥ 0.25, top 16 results per query). Duplicates are removed.
3. **Generation** — All retrieved chunks are injected into a single GPT-4o call that produces a structured, cited answer. **One LLM API call produces the final answer.**

---

## Prerequisites

- Node.js 20+
- PostgreSQL 15+ with the `pgvector` extension enabled
- An OpenAI API key (or a Vercel AI Gateway key that routes to OpenAI)
- The EDGAR corpus `.txt` files unzipped on disk

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd vercel-ai-sdk-rag
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
DATABASE_URL=postgres://user:password@localhost:5432/edgar_rag
AI_GATEWAY_API_KEY=your_key_here
```

### 3. Enable pgvector and run migrations

Connect to your Postgres instance and enable the extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Then run the Drizzle migrations:

```bash
npm run db:migrate
```

### 4. Ingest the EDGAR corpus

**Download and unzip the corpus**

Download `edgar_corpus.zip` and unzip it. You should have a directory containing `.txt` filings and a `manifest.json`:

```
edgar_corpus/
  manifest.json
  aapl_10k_2024.txt
  tsla_10q_2024q2.txt
  ...
```

**Point the script at your corpus directory**

Open `scripts/ingest-edgar.ts` and update `CORPUS_DIR` to match where you unzipped the corpus:

```ts
const CORPUS_DIR = "/path/to/edgar_corpus"; // edit this line
```

By default it points to `~/Downloads/edgar_corpus`.

**Run ingestion**

```bash
npm run ingest:edgar
```

The script will:
1. Find all `.txt` files in the corpus directory
2. Extract readable text from each filing (strips metadata headers and XBRL markup)
3. Chunk each filing into ~800-character segments with 100-character overlap
4. Batch-embed all chunks using `text-embedding-ada-002` (96 chunks per API call)
5. Insert resources and embeddings into your Postgres database

Live progress is printed to the terminal:

```
[47/210] 83s elapsed — tsla_10k_2024.txt (34 chunks)
```

Any files that fail are collected and reported at the end without stopping the run.

**Expected duration:** 5–20 minutes depending on corpus size and your OpenAI API rate limit tier.

**Verify ingestion**

After the script completes, confirm rows were inserted:

```bash
npm run db:studio
```

Open the Drizzle Studio URL in your browser and check that the `embeddings` table has rows.

---

## Running the app

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Type a business question and press Enter.

---

## Example questions to try

```
What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?

How has NVIDIA's revenue and growth outlook changed over the last two years?

What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?

What does Amazon say about its capital expenditure plans in its most recent filings?

How do Microsoft and Google describe their AI strategy and competitive positioning?
```

---

## Project structure

```
app/(preview)/
  api/chat/route.ts     # RAG API: retrieval + single GPT-4o call
  page.tsx              # Streaming chat UI

lib/
  ai/embedding.ts       # Embedding generation + vector similarity search
  actions/resources.ts  # Server action: store content with embeddings
  db/
    schema/             # Drizzle schema: resources + embeddings (pgvector)
    migrations/         # Auto-generated SQL migrations

scripts/
  ingest-edgar.ts       # Bulk EDGAR corpus ingestion pipeline
```

---

## Design decisions

See [PROMPT_LOG.md](./PROMPT_LOG.md) for full prompt iteration history and rationale.

**Single-call constraint:** The final answer is produced in one `streamText` call to GPT-4o. Query expansion (generating multiple search terms) happens inside the `getInformation` tool's `similarQuestions` parameter — the model reasons about what to search for as part of its tool-calling step, not via a separate LLM call. An earlier version used an `understandQuery` tool that called `generateObject` internally; this was removed as it violated the single-call constraint.

**Chunking strategy:** 800-character chunks with 100-character overlap on paragraph boundaries. Overlap preserves sentence context across chunk boundaries. Short paragraphs (<40 chars) are skipped to avoid noise.

**Retrieval breadth:** For multi-company comparison questions, the model generates 3–5 distinct queries (e.g., one per company + one for the topic). Each query retrieves up to 16 chunks (similarity ≥ 0.25), deduplicated. This ensures adequate coverage across filings from different companies.

**Model:** GPT-4o for reasoning and generation. `text-embedding-ada-002` for embeddings (1536 dimensions, HNSW cosine index).
