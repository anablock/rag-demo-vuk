import "dotenv/config";
import fs from "fs";
import path from "path";
import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { embedMany } from "ai";
import { customAlphabet } from "nanoid";
import { resources } from "../lib/db/schema/resources";
import { embeddings as embeddingsTable } from "../lib/db/schema/embeddings";

const nanoid = customAlphabet("abcdefghijklmnopqrstuvwxyz0123456789");

const CORPUS_DIR = "/Users/vukdukic/Downloads/edgar_corpus";
const CHUNK_SIZE = 800;
const CHUNK_OVERLAP = 100;
const EMBED_BATCH_SIZE = 96; // OpenAI max is 2048 inputs, keep batches reasonable
const CONCURRENCY = 3; // files processed in parallel

// ── DB setup ─────────────────────────────────────────────────────────────────

const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) throw new Error("DATABASE_URL is not set");

const client = postgres(DATABASE_URL);
const db = drizzle(client);

// ── Chunking ──────────────────────────────────────────────────────────────────

function extractReadableText(raw: string): string {
  // Files start with metadata lines (Company:, Ticker:, etc.) then a separator
  // then dense XBRL, then the actual human-readable SEC filing text.
  // The readable text starts after the last occurrence of a long run of uppercase
  // form text ("UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION").
  const separatorIdx = raw.indexOf("UNITED STATES");
  if (separatorIdx !== -1) {
    return raw.slice(separatorIdx);
  }
  // Fallback: strip the header metadata block (first 10 lines) and return rest
  const lines = raw.split("\n");
  const headerEnd = lines.findIndex((l) => l.startsWith("====="));
  return lines.slice(headerEnd + 1).join("\n");
}

function chunkText(text: string): string[] {
  // Split on double newlines (paragraph boundaries) first, then enforce max size
  const paragraphs = text
    .split(/\n{2,}/)
    .map((p) => p.replace(/\s+/g, " ").trim())
    .filter((p) => p.length > 40); // skip near-empty paragraphs

  const chunks: string[] = [];
  let current = "";

  for (const para of paragraphs) {
    if (current.length + para.length + 1 <= CHUNK_SIZE) {
      current = current ? `${current} ${para}` : para;
    } else {
      if (current) chunks.push(current);
      // If single paragraph exceeds CHUNK_SIZE, split by sentence
      if (para.length > CHUNK_SIZE) {
        const sentences = para.match(/[^.!?]+[.!?]+/g) ?? [para];
        let sentBuf = "";
        for (const s of sentences) {
          if (sentBuf.length + s.length <= CHUNK_SIZE) {
            sentBuf = sentBuf ? `${sentBuf} ${s}` : s;
          } else {
            if (sentBuf) chunks.push(sentBuf);
            sentBuf = s.slice(0, CHUNK_SIZE);
          }
        }
        if (sentBuf) current = sentBuf;
        else current = "";
      } else {
        current = para;
      }
    }
  }
  if (current) chunks.push(current);

  // Add overlap: prepend tail of previous chunk to each chunk
  const withOverlap: string[] = [];
  for (let i = 0; i < chunks.length; i++) {
    if (i === 0) {
      withOverlap.push(chunks[i]);
    } else {
      const prev = chunks[i - 1];
      const overlap = prev.slice(-CHUNK_OVERLAP);
      withOverlap.push(`${overlap} ${chunks[i]}`);
    }
  }
  return withOverlap;
}

// ── Embedding ─────────────────────────────────────────────────────────────────

const embeddingModel = "openai/text-embedding-ada-002";

async function embedChunks(
  chunks: string[]
): Promise<Array<{ content: string; embedding: number[] }>> {
  const results: Array<{ content: string; embedding: number[] }> = [];
  for (let i = 0; i < chunks.length; i += EMBED_BATCH_SIZE) {
    const batch = chunks.slice(i, i + EMBED_BATCH_SIZE);
    const { embeddings } = await embedMany({
      model: embeddingModel,
      values: batch,
    });
    for (let j = 0; j < batch.length; j++) {
      results.push({ content: batch[j], embedding: embeddings[j] });
    }
  }
  return results;
}

// ── Per-file ingestion ────────────────────────────────────────────────────────

async function ingestFile(filePath: string): Promise<number> {
  const raw = fs.readFileSync(filePath, "utf-8");
  const readable = extractReadableText(raw);
  const chunks = chunkText(readable);

  if (chunks.length === 0) return 0;

  // Insert resource row (store filename as content summary)
  const fileName = path.basename(filePath);
  const [resource] = await db
    .insert(resources)
    .values({ content: fileName })
    .returning();

  // Embed and insert in batches
  const embedded = await embedChunks(chunks);
  await db.insert(embeddingsTable).values(
    embedded.map(({ content, embedding }) => ({
      id: nanoid(),
      resourceId: resource.id,
      content,
      embedding,
    }))
  );

  return chunks.length;
}

// ── Concurrency helper ────────────────────────────────────────────────────────

async function runWithConcurrency<T>(
  items: T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<void>
): Promise<void> {
  let idx = 0;
  async function worker() {
    while (idx < items.length) {
      const i = idx++;
      await fn(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: concurrency }, worker));
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const files = fs
    .readdirSync(CORPUS_DIR)
    .filter((f) => f.endsWith(".txt"))
    .map((f) => path.join(CORPUS_DIR, f));

  console.log(`Found ${files.length} files. Starting ingestion...\n`);

  let completed = 0;
  let totalChunks = 0;
  const errors: string[] = [];
  const startTime = Date.now();

  await runWithConcurrency(files, CONCURRENCY, async (filePath, i) => {
    const name = path.basename(filePath);
    try {
      const chunks = await ingestFile(filePath);
      totalChunks += chunks;
      completed++;
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
      process.stdout.write(
        `\r[${completed}/${files.length}] ${elapsed}s elapsed — ${name} (${chunks} chunks)`
      );
    } catch (err) {
      errors.push(`${name}: ${err instanceof Error ? err.message : err}`);
      completed++;
    }
  });

  console.log(`\n\n✅ Done. ${completed} files, ${totalChunks} total chunks.`);
  if (errors.length > 0) {
    console.log(`\n⚠️  ${errors.length} errors:`);
    errors.forEach((e) => console.log(`  - ${e}`));
  }

  await client.end();
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
