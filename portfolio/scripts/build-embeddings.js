// scripts/build-embeddings.js
//
// One-time script that pre-computes dense embeddings for the corpus.
// Run with: npm run build-embeddings
//
// What this does:
//   1. Loads data/corpus.json
//   2. Loads a small sentence-transformer model (Xenova/all-MiniLM-L6-v2, ~25MB)
//   3. Embeds each chunk's text to a 384-dimensional vector
//   4. Writes data/embeddings.json
//
// After running this, the chat API automatically switches from BM25-only
// to hybrid retrieval (BM25 + dense + RRF fusion).
//
// Why MiniLM-L6-v2:
//   - 384-dim vectors (small, fast cosine similarity)
//   - Strong on semantic similarity benchmarks despite size
//   - Standard baseline in the retrieval literature

import { pipeline } from '@xenova/transformers';
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CORPUS_PATH = resolve(__dirname, '../data/corpus.json');
const OUTPUT_PATH = resolve(__dirname, '../data/embeddings.json');
const MODEL = 'Xenova/all-MiniLM-L6-v2';

async function main() {
  console.log('━━━ Building dense embeddings ━━━');
  console.log(`Model: ${MODEL}`);

  const corpus = JSON.parse(await readFile(CORPUS_PATH, 'utf-8'));
  console.log(`Corpus chunks: ${corpus.chunks.length}`);

  console.log('Loading model (first run downloads ~25MB)...');
  const t0 = Date.now();
  const embedder = await pipeline('feature-extraction', MODEL);
  console.log(`Model loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  console.log('Embedding corpus...');
  const vectors = {};
  let i = 0;
  for (const chunk of corpus.chunks) {
    const out = await embedder(chunk.text, { pooling: 'mean', normalize: true });
    vectors[chunk.id] = Array.from(out.data);
    i++;
    if (i % 5 === 0 || i === corpus.chunks.length) {
      console.log(`  ${i}/${corpus.chunks.length} chunks embedded`);
    }
  }

  const dim = Object.values(vectors)[0].length;
  const payload = {
    model: MODEL,
    dim,
    builtAt: new Date().toISOString(),
    vectors
  };

  await writeFile(OUTPUT_PATH, JSON.stringify(payload, null, 2));
  console.log(`━━━ Done ━━━`);
  console.log(`Wrote ${Object.keys(vectors).length} embeddings × ${dim} dims to data/embeddings.json`);
  console.log('Dense retrieval is now enabled. Redeploy to Vercel.');
}

main().catch(err => {
  console.error('Build failed:', err);
  process.exit(1);
});
