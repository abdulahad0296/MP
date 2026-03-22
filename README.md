# Agentic Research Planning Framework
MSc AI & Machine Learning

---

## What This System Does

Takes a research topic as input and produces validated, novel research proposals by:
1. Retrieving real papers from arXiv
2. Extracting concepts and identifying underexplored research gaps
3. Generating structured candidate research plans per gap
4. Scoring each plan for novelty (embedding-based) and feasibility (rule-based)
5. Outputting accepted proposals with titles, blueprints, and novelty scores

---

## Quick Start

```bash
# 1. Install dependencies
pip install arxiv groq sentence-transformers scikit-learn numpy python-dotenv

# 2. Get a free Groq API key (no credit card required)
#    https://console.groq.com → sign up → API Keys → Create API Key

# 3. Configure
cp .env.example .env
# Edit .env: GROQ_API_KEY=your_key_here

# 4. Run
python main.py --topic "federated learning privacy"
python main.py --topic "vision transformers medical imaging"
python main.py --topic "graph neural networks drug discovery"
```

---

## Project Structure

```
research_planner/
├── main.py                    # Pipeline orchestrator — run this
├── config.py                  # Constants, thresholds, API keys
├── .env                       # .env to add your Groq API key
│
├── models/
│   └── schemas.py             # Paper, ResearchGap, ResearchPlan, ReviewResult
│
├── tools/
│   ├── arxiv_tool.py          # arXiv paper retrieval
│   ├── embedder.py            # Sentence-transformer embeddings
│   └── feasibility_checker.py # Dataset + metric rule-based validation
│
├── agents/
│   ├── librarian_agent.py     # Concept extraction + gap detection
│   ├── planner_agent.py       # Research plan generation
│   └── reviewer_agent.py      # Novelty scoring + feasibility + output fields
│
├── outputs/
│   └── results.json           # Saved pipeline output (auto-generated)
│
└── test_step1/2/3/4.py        # Per-step validation tests
```

---

## System Architecture

```
User Topic
    │
    ▼
Librarian Agent ── arXiv API ──► 20 papers
    │                retrieve + extract concepts + identify gaps
    ▼
Planner Agent ──────────────►  3 plans × N gaps
    │                select context + generate structured plans
    ▼
Reviewer Agent
    ├── Novelty Score  (sentence-transformers + cosine similarity)
    ├── Feasibility    (rule-based: known datasets + valid metrics)
    └── Output Fields  (LLM: title + direction + blueprint)
    │
    ▼
Revision Loop ── re-plan rejected (max 2 attempts)
    │
    ▼
outputs/results.json  +  terminal display
```

---

## LLM

All agents use **Groq free tier** with `llama-3.3-70b-versatile`.
- Free at https://console.groq.com — no credit card required
- JSON mode: `response_format={"type": "json_object"}`

---

## Key Configuration (config.py)

| Constant | Default | Description |
|---|---|---|
| `MAX_PAPERS` | 20 | Papers retrieved from arXiv |
| `MAX_PLANS_PER_GAP` | 3 | Plans generated per research gap |
| `NOVELTY_THRESHOLD` | 1.5 | Minimum score to accept a plan |
| `MAX_REVISION_ATTEMPTS` | 2 | Retry attempts for rejected plans |

---

## Implementation Progress

| Step | Module | Status |
|---|---|---|
| Foundation | models/schemas.py, config.py | ✅ Complete |
| Step 1 | tools/arxiv_tool.py | ✅ Complete |
| Step 2 | agents/librarian_agent.py | ✅ Complete |
| Step 3 | agents/planner_agent.py | ✅ Complete |
| Step 4 | agents/reviewer_agent.py + tools/feasibility_checker.py | ✅ Complete |
| Step 5 | tools/embedder.py + integration | ✅ Complete |
| Integration | main.py | ✅ Complete |
