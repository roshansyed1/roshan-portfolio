# Roshan Syed — Portfolio with Hybrid RAG

An AI-powered personal portfolio that combines an editorial design system with a working retrieval-augmented generation pipeline. The "Ask Roshan-AI" concierge runs a real retrieval system over a hand-curated corpus of resume content — not a chatbot wrapper.

---

## What's inside

**The portfolio itself is the AI project.** Architecture summary:

| Layer | What it does | Implementation |
|---|---|---|
| Corpus | 31 hand-curated retrievable chunks | `data/corpus.json` |
| Sparse retrieval | BM25 with proper tokenization, IDF, length norm | `api/chat.js` (pure JS, ~50 LOC) |
| Dense retrieval *(optional)* | MiniLM-L6-v2 embeddings, cosine similarity | `data/embeddings.json` (built once) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) | `api/chat.js` |
| Generation | Claude Haiku 4.5 with citation enforcement | `api/chat.js` |
| Frontend | SPA navigation, evidence display, latency | `index.html` |

**Default mode** (no setup): pure BM25 sparse retrieval. Works out of the box after deployment.

**Hybrid mode** (one command): BM25 + dense embeddings, fused with RRF. Run `npm run build-embeddings` once.

---

## Deploy in 5 minutes

### Step 1 — Get a Claude API key
1. Sign in at https://console.anthropic.com
2. Create an API key. Add credits ($5 covers thousands of queries).

### Step 2 — Deploy to Vercel
1. Push this folder to a GitHub repo.
2. Sign in to https://vercel.com with that GitHub account.
3. "Add New Project" → import your repo → Deploy.

### Step 3 — Add the API key
1. In your Vercel project: **Settings → Environment Variables**.
2. Add `ANTHROPIC_API_KEY` with your key as the value.
3. Redeploy (one click).

Done. The portfolio is live with BM25 retrieval + Claude generation + citation enforcement.

---

## Optional: enable hybrid retrieval (BM25 + dense + RRF)

This step adds dense embeddings on top of the default BM25, fused via Reciprocal Rank Fusion. The chat API auto-detects `data/embeddings.json` and switches modes — no other config needed.

```bash
npm install
npm run build-embeddings   # ~2 minutes, downloads MiniLM-L6-v2 (~25MB) once
git add data/embeddings.json
git commit -m "Enable hybrid retrieval"
git push                   # Vercel auto-redeploys
```

The chat header will now show `Hybrid RAG · hybrid` instead of `bm25`.

---

## File structure

```
roshan-portfolio/
├── index.html                  # SPA: 7 pages, editorial design, AI Concierge UI
├── api/
│   └── chat.js                 # Hybrid retrieval engine + Claude generation
├── data/
│   ├── corpus.json             # 31 hand-curated chunks of resume content
│   └── embeddings.json         # (built by script — enables hybrid mode)
├── scripts/
│   └── build-embeddings.js     # One-time embeddings build (MiniLM-L6-v2)
├── package.json
└── README.md
```

---

## Customizing

**Add or edit a chunk:**
1. Open `data/corpus.json`.
2. Add or modify an entry with `id`, `type`, `tags`, `source`, `text`.
3. If hybrid mode is enabled, re-run `npm run build-embeddings`.

**Update GitHub / live demo links:**
- `index.html` — search for `TODO` comments near the contact section to add your GitHub URL.

**Update the chat suggestions:**
- `index.html` — search for `.cin-sg` to find the suggestion buttons.

---

## What this gets you in an interview

A defensible answer to *"tell me about an AI project you built recently"*:

> "I built a hybrid retrieval system over my own resume corpus. It combines BM25 sparse retrieval — which I implemented from scratch with proper tokenization, IDF weighting, and length normalization — with dense retrieval using MiniLM-L6-v2 embeddings. The two rankings are fused via Reciprocal Rank Fusion with k=60. Generation runs through Claude with strict citation enforcement — every claim references a chunk ID, and the UI surfaces the retrieved evidence with similarity scores so visitors can verify there's no hallucination. The corpus is 31 hand-curated chunks with type and tag metadata. Default mode is BM25-only with zero setup; hybrid mode is an optional upgrade via a one-time embedding build script. The whole thing runs on Vercel serverless with the Claude API."

Five distinct technical talking points:
1. Hybrid retrieval (BM25 + dense + RRF) — published research-grade technique
2. BM25 implemented from scratch (not a library wrapper)
3. Citation enforcement as a hallucination-prevention mechanism
4. Graceful degradation (works without embeddings, upgrades cleanly)
5. UI surfaces retrieved evidence with scores — transparency built into the product

---

## Tech stack

- Vanilla HTML/CSS/JS (no framework — fast load, no build step for the frontend)
- Vercel serverless functions (Node 18+, ESM)
- Claude API (Haiku 4.5 for generation)
- `@xenova/transformers` (only for the optional embedding build script)
- Fonts: Fraunces (variable serif), Inter Tight, JetBrains Mono — all Google Fonts

## License

Personal portfolio code — feel free to reference the RAG architecture for learning; don't lift the corpus content.
