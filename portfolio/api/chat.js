// /api/chat.js — Hybrid RAG retrieval + Claude generation
//
// Architecture:
//   1. BM25 sparse retrieval (always on)
//   2. Dense retrieval via pre-computed embeddings (optional)
//   3. Reciprocal Rank Fusion (RRF) when both are available
//   4. Claude generation with citation enforcement
//
// Note: corpus is inlined to avoid JSON import attribute compatibility
// issues across Node versions (Node 22 removed the `assert` keyword).

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

// ─── CORPUS (inlined) ─────────────────────────────────────────────
const corpus = {
  version: "1.0.0",
  owner: "Roshan Syed",
  chunks: [
    { id: "about-summary", type: "about", tags: ["bio","summary","education"], source: "About",
      text: "Roshan Syed is a Data Science graduate student at the University of Maryland, College Park (MS Data Science, GPA 3.9/4.0, 2024-Present), with a background in Artificial Intelligence and Machine Learning. He builds end-to-end ML systems that go from raw data to deployed, scalable inference, with production experience spanning cloud orchestration, generative AI, and computer vision." },
    { id: "about-focus", type: "about", tags: ["specialization","production","mlops"], source: "About",
      text: "Roshan's work spans production ML pipelines (containerized on AWS, processing thousands of signals daily), generative AI (diffusion models, LoRA fine-tuning, Stable Diffusion), and computer vision (super-resolution, object detection, image enhancement). He cares about the full stack: feature engineering, model architecture, cloud orchestration, monitoring, and deployment." },
    { id: "about-current", type: "about", tags: ["job-search","roles","availability"], source: "About",
      text: "Roshan is currently looking for Machine Learning Engineer, Data Scientist, and AI Research roles where he can apply deep learning, cloud MLOps, and generative AI at scale. He is based in College Park, MD and willing to relocate. Contact: roshan.17.syed@gmail.com." },
    { id: "edu-umd", type: "education", tags: ["umd","masters","data-science","gpa"], source: "Education / UMD",
      text: "Master of Science in Data Science at the University of Maryland, College Park (2024-Present). Current GPA: 3.9 out of 4.0. Coursework spans machine learning, statistical inference, data engineering, and applied AI." },
    { id: "edu-nhce", type: "education", tags: ["bachelors","ai-ml","undergrad"], source: "Education / NHCE",
      text: "Bachelor of Engineering in Artificial Intelligence and Machine Learning at New Horizon College of Engineering (2020-2024). Final CGPA: 8.46 out of 10.0. Four-year program focused on AI/ML fundamentals, neural networks, computer vision, and applied data science." },
    { id: "exp-fyenn-overview", type: "experience", tags: ["internship","fyenn-labs","ml-engineering"], source: "Fyenn Labs Services / Internship",
      text: "Machine Learning Intern at Fyenn Labs Services from February 2024 to May 2024. Built production computer vision systems for surveillance and IoT, delivered market segmentation analysis, and produced quantitative demand reports across three vertical sectors." },
    { id: "exp-fyenn-ultrafocus", type: "experience", tags: ["super-resolution","computer-vision","production","cnn"], source: "Fyenn Labs / Ultra-Focus",
      text: "Architected and deployed Ultra-Focus, a low-latency image super-resolution inference pipeline using deep convolutional upsampling networks. Achieved 2x resolution enhancement on live surveillance feeds while eliminating hardware upgrade dependencies, reducing per-camera infrastructure cost by approximately 40 percent." },
    { id: "exp-fyenn-ultrafocus2", type: "experience", tags: ["optimization","iot","streaming","latency"], source: "Fyenn Labs / Ultra-Focus 2.0",
      text: "Engineered Ultra-Focus 2.0 with optimized batch and streaming inference pipelines for IoT camera ingestion. Implemented frame-level preprocessing, model serving optimization, and asynchronous I/O to reduce end-to-end frame processing latency by 35 percent across distributed deployments." },
    { id: "exp-fyenn-segmentation", type: "experience", tags: ["clustering","k-means","unsupervised","marketing"], source: "Fyenn Labs / Market Segmentation",
      text: "Built unsupervised market segmentation pipeline using K-Means clustering with silhouette-score-based k-selection on consumer behavioral features. Identified 4 actionable buyer segments that informed EV adoption targeting strategy under India's subsidy scheme." },
    { id: "exp-fyenn-reports", type: "experience", tags: ["analytics","business","market-research"], source: "Fyenn Labs / Demand Analysis",
      text: "Delivered quantitative demand-analysis reports across 3 vertical sectors: security, enterprise, and education. Synthesized competitive landscape data and total addressable market sizing for executive stakeholders." },
    { id: "proj-siddhantha-overview", type: "project", tags: ["finance","production","ml-pipeline","aws","lightgbm"], source: "Project / Siddhantha T20",
      text: "Siddhantha T20 is a production-grade stock intelligence platform that processes over 3,000 tickers daily across the Russell 2000, S&P 500, and S&P 600 universes. It trains per-universe LightGBM binary classifiers with time-series-aware train/validation splits targeting 8-20 percent forward-return thresholds. Status: live since 2024." },
    { id: "proj-siddhantha-features", type: "project", tags: ["feature-engineering","finance","technical-indicators"], source: "Project / Siddhantha T20 — Feature Engineering",
      text: "Engineered over 100 predictive features across multi-window technical indicators (momentum, volatility, OHLCV ratios, market-relative beta) and cross-sectional fundamental factors (value, growth, quality, leverage) with z-score normalization, computed over 11+ years of historical data serialized in Parquet format on Amazon S3." },
    { id: "proj-siddhantha-cloud", type: "project", tags: ["aws","ecs-fargate","step-functions","mlops","deployment"], source: "Project / Siddhantha T20 — Cloud Architecture",
      text: "Deployed a fully automated cloud-native pipeline on AWS: containerized ECS Fargate tasks orchestrated via Step Functions with EventBridge-scheduled daily and weekly cadences, SNS per-task failure alerting, RDS (MySQL) prediction persistence, and a Lambda-served dashboard with OAuth-gated access control." },
    { id: "proj-siddhantha-modeling", type: "project", tags: ["lightgbm","time-series","classification","modeling"], source: "Project / Siddhantha T20 — Modeling",
      text: "Modeling approach: LightGBM binary classifiers trained per universe with time-series-aware train/validation splits to prevent look-ahead leakage. Targets set at 8-20 percent forward-return thresholds. Predictions persisted to RDS MySQL for downstream consumption by the dashboard." },
    { id: "proj-clearshot-overview", type: "project", tags: ["generative-ai","diffusion","computer-vision","stable-diffusion"], source: "Project / ClearShot",
      text: "ClearShot is an AI-powered product photo enhancement system built in Spring 2025, deployed on Hugging Face Spaces. It is a 5-stage generative restoration pipeline combining background segmentation, structural conditioning, ControlNet-guided Stable Diffusion, studio compositing, and Real-ESRGAN super-resolution." },
    { id: "proj-clearshot-pipeline", type: "project", tags: ["pipeline","stable-diffusion","controlnet","u2net","real-esrgan"], source: "Project / ClearShot — Pipeline",
      text: "Pipeline stages: (1) U2-Net background segmentation, (2) Canny and HED structural conditioning, (3) ControlNet-guided Stable Diffusion 1.5 image-to-image enhancement, (4) studio-background compositing, (5) Real-ESRGAN 2x super-resolution. Each stage independently testable and tunable." },
    { id: "proj-clearshot-lora", type: "project", tags: ["lora","fine-tuning","diffusion","unet","training"], source: "Project / ClearShot — LoRA Fine-Tuning",
      text: "Authored a LoRA fine-tuning pipeline with rank=16, alpha=32, targeting UNet cross-attention projections. Trained for 5,000 optimizer steps on a T4 GPU with MSE loss on a curated 24,131-image Amazon Berkeley Objects subset across 7 product categories." },
    { id: "proj-clearshot-results", type: "project", tags: ["metrics","lpips","fid","evaluation"], source: "Project / ClearShot — Results",
      text: "Achieved best-in-class perceptual quality on a 457-image stratified test set: LPIPS 0.322 and FID 88.66, outperforming all 5 evaluated baselines on perceptual similarity. Evaluation methodology designed to test both fidelity to the original product and quality of the enhanced output." },
    { id: "proj-clearshot-deployment", type: "project", tags: ["deployment","huggingface","gradio","docker","production"], source: "Project / ClearShot — Deployment",
      text: "Implemented a DiffusionEnhancer wrapper with UniPC multistep scheduler, xformers memory-efficient attention, and fp16/fp32 hardware abstraction. Deployed as a Gradio web application on Hugging Face Spaces with Docker containerization." },
    { id: "proj-clearshot-bugfix", type: "project", tags: ["lora","peft","debugging","engineering"], source: "Project / ClearShot — LoRA Loading Bug Fix",
      text: "Diagnosed and resolved a silent compatibility issue in the LoRA loading path. The training script saved adapter checkpoints in peft format, but the diffusers default loader pipe.load_lora_weights() silently failed to parse those keys and degraded to without-LoRA at inference, making the entire fine-tuning effort invisible. Roshan wrote a custom load_lora_peft() helper using PeftModel.from_pretrained() to merge the adapter into the UNet cross-attention blocks before bundling it back into the pipeline. Without this fix the LoRA results LPIPS 0.322 would not have been reproducible." },
    { id: "proj-emojify-overview", type: "project", tags: ["computer-vision","emotion-recognition","cnn","real-time"], source: "Project / Emojify",
      text: "Emojify is a real-time facial emotion recognition system combining a custom 5-layer CNN with Haar Cascade face detection. The full pipeline runs end-to-end on CPU at under 150 milliseconds, demonstrating production-ready integration of computer vision models with interactive user interfaces." },
    { id: "proj-emojify-model", type: "project", tags: ["cnn","fer-2013","classification","training"], source: "Project / Emojify — Model",
      text: "Trained a custom 5-layer convolutional neural network on the FER-2013 dataset, which contains 35,887 labeled images spanning 7 emotion classes (anger, disgust, fear, happiness, sadness, surprise, neutral). Achieved 65 percent or higher test accuracy across all 7 emotion classes." },
    { id: "proj-emojify-pipeline", type: "project", tags: ["pipeline","real-time","opencv","haar-cascade"], source: "Project / Emojify — Pipeline",
      text: "Real-time pipeline: webcam capture, Haar Cascade face detection, CNN emotion classification, animated GIF response. Sub-100ms emotion inference on CPU, total end-to-end pipeline latency under 150ms. No GPU required for deployment." },
    { id: "proj-visualsearch-overview", type: "project", tags: ["object-detection","yolo","recommendation","computer-vision","graduate"], source: "Project / Visual Product Search",
      text: "Visual Product Search is a real-time visual product recommendation system that combines YOLO v3 object detection with a curated 10,000-entry Amazon product catalog spanning seven macro-categories. Users upload an image, the system detects objects with bounding-box transparency, and surfaces matching products with name, category, price, and direct purchase URL. Completed for DATA602 Principles of Data Science at the University of Maryland (Fall 2024)." },
    { id: "proj-visualsearch-model", type: "project", tags: ["yolo-v3","darknet-53","object-detection","training","metrics"], source: "Project / Visual Product Search — Model",
      text: "YOLO v3 inference pipeline using Darknet-53 backbone with three-scale detection: 416 by 416 blob preprocessing, confidence threshold filtering at 0.5, and Non-Max Suppression for bounding-box refinement. Trained for 10 epochs achieving 86.93 percent final accuracy, 0.94 recall, and peak 94.75 percent accuracy at epoch 6. Tracked precision and recall curves per epoch for stability analysis." },
    { id: "proj-visualsearch-pipeline", type: "project", tags: ["product-matching","keyword-matching","amazon","data-cleaning","eda"], source: "Project / Visual Product Search — Pipeline",
      text: "Keyword-matching engine maps detected COCO class labels to Amazon product entries via case-insensitive string matching. Preprocessed 10,000-entry product dataset: missing-value imputation, price-range averaging where range strings like dollar twenty to dollar forty were converted to midpoints, and category standardization. Produced EDA artifacts including histograms, boxplots, and category pie charts informing the recommendation strategy." },
    { id: "proj-visiodetect-overview", type: "project", tags: ["object-detection","yolo","ssd","benchmark","edge"], source: "Project / Visio-Detect",
      text: "Visio-Detect is a real-time object detection benchmarking framework that evaluates speed-versus-accuracy tradeoffs between YOLO v3 and SSD MobileNet. The framework is reusable for evaluating any pair of object detection models for edge deployment scenarios." },
    { id: "proj-visiodetect-results", type: "project", tags: ["yolo-v3","ssd-mobilenet","coco","benchmark","metrics"], source: "Project / Visio-Detect — Results",
      text: "Implemented YOLO v3 and SSD MobileNet on the COCO test-dev dataset. Benchmark results: YOLO v3 achieved 65.8 mAP at 20 FPS, while SSD ran at 23 FPS with approximately 5 percentage points lower mAP. The framework produced quantitative characterization of the precision-throughput frontier for edge deployment decisions." },
    { id: "skills-dl-genai", type: "skills", tags: ["deep-learning","pytorch","transformers","diffusion","lora"], source: "Skills / Deep Learning & GenAI",
      text: "Deep Learning and Generative AI skills: PyTorch, TensorFlow, Keras, CNNs, LSTMs, Transformers, BERT, Transfer Learning, LoRA Fine-Tuning, Stable Diffusion, ControlNet, LangChain, RAG Pipelines. Hands-on production experience with LoRA fine-tuning on diffusion models and deployment of generative pipelines." },
    { id: "skills-ml-stats", type: "skills", tags: ["ml","lightgbm","xgboost","classical-ml","statistics"], source: "Skills / ML & Data Analysis",
      text: "Machine Learning and Data Analysis skills: Scikit-learn, LightGBM, XGBoost, Pandas, NumPy, Feature Engineering, Ensemble Methods, K-Means Clustering, Regression, Classification, Time Series Forecasting, A/B Testing, Hypothesis Testing, Statistical Inference." },
    { id: "skills-cloud-mlops", type: "skills", tags: ["aws","mlops","cloud","docker","ci-cd"], source: "Skills / Cloud & MLOps",
      text: "Cloud and MLOps skills: AWS services including S3, RDS, Lambda, ECS Fargate, Step Functions, EventBridge, SNS. Container orchestration with Docker. Version control with Git. CI/CD pipelines. Deployment platforms including Gradio and Hugging Face Spaces." },
    { id: "skills-cv-nlp", type: "skills", tags: ["computer-vision","nlp","opencv","yolo"], source: "Skills / Computer Vision & NLP",
      text: "Computer Vision and NLP skills: OpenCV, YOLO v3, SSD MobileNet, Image Super-Resolution, Real-ESRGAN, Object Detection, Sentiment Analysis. NLP work includes BERT-based sentiment classification and RAG-pipeline development with LangChain." },
    { id: "skills-data-eng", type: "skills", tags: ["sql","etl","data-engineering","databases"], source: "Skills / Data Engineering",
      text: "Data Engineering skills: Python, R, SQL, MySQL, PostgreSQL, ETL/ELT Pipelines, Parquet format, Data Cleaning, Exploratory Data Analysis, Statistical Analysis. Production experience persisting ML predictions to RDS and serializing large feature datasets to Parquet on S3." },
    { id: "skills-viz", type: "skills", tags: ["visualization","bi","dashboards"], source: "Skills / Visualization",
      text: "Data Visualization skills: Power BI, Tableau, Matplotlib, Seaborn. Has built executive-facing BI dashboards as well as code-driven analytical visualizations for model evaluation and exploratory data analysis." },
    { id: "differentiators", type: "about", tags: ["strengths","differentiators","why-hire"], source: "Why hire Roshan",
      text: "Roshan combines production engineering rigor with research depth. Unique strengths: he has shipped both classical-ML systems (LightGBM trading pipeline on AWS) and frontier-AI systems (LoRA-fine-tuned diffusion models). He owns the full lifecycle, from feature engineering through cloud orchestration to model evaluation. Strong academic record (3.9 GPA at UMD) backed by measurable production wins (40 percent cost reduction, 35 percent latency reduction, best-in-class LPIPS scores)." }
  ]
};

// ─── EMBEDDINGS (optional, lazy loaded) ───────────────────────────
const __dirname = dirname(fileURLToPath(import.meta.url));
const EMBEDDINGS_PATH = resolve(__dirname, '../data/embeddings.json');
let cachedEmbeddings = null;
let embeddingsAttempted = false;

function tryLoadEmbeddings() {
  if (embeddingsAttempted) return cachedEmbeddings;
  embeddingsAttempted = true;
  try {
    if (existsSync(EMBEDDINGS_PATH)) {
      cachedEmbeddings = JSON.parse(readFileSync(EMBEDDINGS_PATH, 'utf-8'));
    }
  } catch (e) {
    console.warn('Embeddings load failed:', e.message);
  }
  return cachedEmbeddings;
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
  return text.toLowerCase().replace(/[^\w\s-]/g, ' ').split(/\s+/)
    .filter(w => w.length > 1 && !STOPWORDS.has(w));
}

// ─── BM25 ─────────────────────────────────────────────────────────
const K1 = 1.5;
const B = 0.75;

const corpusStats = (() => {
  const docs = corpus.chunks.map(c => ({ id: c.id, tokens: tokenize(c.text) }));
  const docLens = docs.map(d => d.tokens.length);
  const avgDocLen = docLens.reduce((a, b) => a + b, 0) / docs.length;
  const df = new Map();
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
  const fused = new Map();
  for (const list of rankings) {
    list.forEach((item, rank) => {
      fused.set(item.id, (fused.get(item.id) || 0) + 1 / (k + rank + 1));
    });
  }
  return Array.from(fused.entries()).map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score);
}

// ─── RETRIEVE TOP-K ───────────────────────────────────────────────
function retrieve(query, queryEmbedding = null, topK = 5) {
  const bm25Results = bm25(query).slice(0, 20);
  let denseResults = null;
  if (queryEmbedding) {
    const embeddings = tryLoadEmbeddings();
    if (embeddings) denseResults = denseRetrieve(queryEmbedding, embeddings).slice(0, 20);
  }
  const fused = denseResults ? rrf([bm25Results, denseResults]) : bm25Results;
  const top = fused.slice(0, topK);
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
  const evidence = retrieved.map((r, i) => `[${i + 1}] (id: ${r.id}) ${r.text}`).join('\n\n');
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
    return res.status(500).json({ error: 'ANTHROPIC_API_KEY not set in Vercel environment variables' });
  }

  try {
    const { messages, queryEmbedding } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'messages array required' });
    }

    const lastUser = [...messages].reverse().find(m => m.role === 'user');
    if (!lastUser) return res.status(400).json({ error: 'no user message' });

    const retrieved = retrieve(lastUser.content, queryEmbedding, 5);
    const retrievalMode = retrieved[0]?.mode || 'bm25';

    const systemPrompt = buildSystemPrompt(retrieved);
    const trimmedHistory = messages.slice(-8);

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
      return res.status(500).json({ error: 'Generation API error: ' + r.status });
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
    return res.status(500).json({ error: 'Internal server error: ' + err.message });
  }
}
