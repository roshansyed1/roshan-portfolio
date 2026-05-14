// /api/chat.js — Hybrid RAG retrieval + Claude generation
//
// Architecture:
//   1. BM25 sparse retrieval (always on, pure JS, no setup required)
//   2. Dense retrieval via pre-computed embeddings (optional, enabled by running scripts/build-embeddings.js)
//   3. Reciprocal Rank Fusion (RRF) when both are available
//   4. Claude generation with citation enforcement
//
// Engineering decisions worth defending in an interview:
//   - Pure-JS BM25 keeps the cold path zero-dependency
//   - Browser-side query embedding (when enabled) eliminates server-side embedding cost
//   - RRF (k=60) is a parameter-free, well-established fusion technique
//   - Citation enforcement: every claim must reference a chunk id

import corpus from '../data/corpus.json' assert { type: 'json' };

// Embeddings load lazily — file may or may not exist.
let cachedEmbeddings = null;
let embeddingsAttempted = false;

async function tryLoadEmbeddings() {
  if (embeddingsAttempted) return cachedEmbeddings;
  embeddingsAttempted = true;
  try {
    const mod = await import('../data/embeddings.json', { assert: { type: 'json' } });
    cachedEmbeddings = mod.default;
    return cachedEmbeddings;
  } catch {
    return null; // Dense retrieval disabled, fall through to BM25-only
  }
}

// ─── TOKENIZATION ─────────────────────────────────────────────────
const STOPWORDS = new Set([
  'a','an','the','and','or','but','is','are','was','were','be','been','being',
  'has','have','had','do','does','did','will','would','could','should','may',
  'might','must','can','of','in','on','at','to','for','with','from','by','as',
  'this','that','these','those','i','you','he','she','it','we','they','what',
  'which','who','whom','about','into','through','during','how','why','when','where'
]);

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 1 && !STOPWORDS.has(w));
}

// ─── BM25 ─────────────────────────────────────────────────────────
const K1 = 1.5;
const B = 0.75;

// Pre-compute corpus statistics once at module load.
const corpusStats = (() => {
  const docs = corpus.chunks.map(c => ({ id: c.id, tokens: tokenize(c.text) }));
  const docLens = docs.map(d => d.tokens.length);
  const avgDocLen = docLens.reduce((a, b) => a + b, 0) / docs.length;
  const df = new Map(); // term -> doc count
  const tf = docs.map(d => {
    const counts = new Map();
    for (const t of d.tokens) counts.set(t, (counts.get(t) || 0) + 1);
    for (const t of counts.keys()) df.set(t, (df.get(t) || 0) + 1);
    return counts;
  });
  return { docs, docLens, avgDocLen, df, tf, N: docs.length };
})();

function bm25(query) {
  const qTokens = tokenize(query);
  const { docs, docLens, avgDocLen, df, tf, N } = corpusStats;
  const scores = new Array(N).fill(0);
  for (const qt of qTokens) {
    const nq = df.get(qt) || 0;
    if (nq === 0) continue;
    const idf = Math.log((N - nq + 0.5) / (nq + 0.5) + 1);
    for (let i = 0; i < N; i++) {
      const fqi = tf[i].get(qt) || 0;
      if (fqi === 0) continue;
      const norm = 1 - B + B * (docLens[i] / avgDocLen);
      scores[i] += idf * (fqi * (K1 + 1)) / (fqi + K1 * norm);
    }
  }
  return scores.map((s, i) => ({ id: docs[i].id, score: s }))
               .filter(x => x.score > 0)
               .sort((a, b) => b.score - a.score);
}

// ─── DENSE RETRIEVAL ──────────────────────────────────────────────
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

function denseRetrieve(queryEmbedding, embeddings) {
  return Object.entries(embeddings.vectors)
    .map(([id, vec]) => ({ id, score: cosine(queryEmbedding, vec) }))
    .sort((a, b) => b.score - a.score);
}

// ─── RECIPROCAL RANK FUSION ───────────────────────────────────────
function rrf(rankings, k = 60) {
  // rankings: array of { id, score } arrays
  const fused = new Map();
  for (const list of rankings) {
    list.forEach((item, rank) => {
      fused.set(item.id, (fused.get(item.id) || 0) + 1 / (k + rank + 1));
    });
  }
  return Array.from(fused.entries())
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score);
}

// ─── RETRIEVE TOP-K ───────────────────────────────────────────────
async function retrieve(query, queryEmbedding = null, topK = 5) {
  const bm25Results = bm25(query).slice(0, 20);

  let denseResults = null;
  if (queryEmbedding) {
    const embeddings = await tryLoadEmbeddings();
    if (embeddings) {
      denseResults = denseRetrieve(queryEmbedding, embeddings).slice(0, 20);
    }
  }

  const fused = denseResults ? rrf([bm25Results, denseResults]) : bm25Results;
  const top = fused.slice(0, topK);

  // Attach chunk content + metadata for the response
  const chunkById = Object.fromEntries(corpus.chunks.map(c => [c.id, c]));
  return top.map(({ id, score }) => {
    const chunk = chunkById[id];
    return {
      id,
      score: Number(score.toFixed(4)),
      source: chunk.source,
      text: chunk.text,
      tags: chunk.tags,
      mode: denseResults ? 'hybrid' : 'bm25'
    };
  });
}

// ─── CLAUDE GENERATION ────────────────────────────────────────────
function buildSystemPrompt(retrieved) {
  const evidence = retrieved
    .map((r, i) => `[${i + 1}] (id: ${r.id}) ${r.text}`)
    .join('\n\n');

  return `You are the AI Career Concierge for Roshan Syed's portfolio. A visitor — usually a recruiter, hiring manager, or engineer — is asking about Roshan.

ANSWER STYLE:
- Concise (2-4 sentences usually, 5-6 max for complex questions)
- Confident and specific, never vague
- Always grounded in the retrieved evidence below
- If the evidence doesn't cover the question, say so honestly. Never invent.

CITATION REQUIREMENT (critical):
- Every factual claim must reference its evidence chunk by id, using the format [chunk-id] inline.
- Example: "Roshan deployed a production ML pipeline processing 3000+ tickers daily on AWS [proj-siddhantha-overview]."
- If you can't cite a claim to retrieved evidence, don't make the claim.

RETRIEVED EVIDENCE:
${evidence}

Answer the visitor's question using only this evidence. Cite chunk ids inline.`;
}

// ─── HANDLER ──────────────────────────────────────────────────────
export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return res.status(500).json({
      error: 'ANTHROPIC_API_KEY not set in Vercel environment variables'
    });
  }

  try {
    const { messages, queryEmbedding } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'messages array required' });
    }

    const lastUser = [...messages].reverse().find(m => m.role === 'user');
    if (!lastUser) return res.status(400).json({ error: 'no user message' });

    // Retrieve top-K relevant chunks
    const retrieved = await retrieve(lastUser.content, queryEmbedding, 5);
    const retrievalMode = retrieved[0]?.mode || 'bm25';

    // Generate grounded response
    const systemPrompt = buildSystemPrompt(retrieved);
    const trimmedHistory = messages.slice(-8); // last 4 turns

    const t0 = Date.now();
    const r = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 500,
        system: systemPrompt,
        messages: trimmedHistory
      })
    });

    if (!r.ok) {
      const errText = await r.text();
      console.error('Anthropic API error:', errText);
      return res.status(500).json({ error: 'Generation API error' });
    }

    const data = await r.json();
    const reply = data?.content?.[0]?.text || "I couldn't generate a response.";
    const genLatency = Date.now() - t0;

    return res.status(200).json({
      reply,
      retrieved,
      stats: {
        mode: retrievalMode,
        retrievedCount: retrieved.length,
        topScore: retrieved[0]?.score || 0,
        generationLatencyMs: genLatency
      }
    });

  } catch (err) {
    console.error('Chat handler error:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
