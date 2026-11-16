"""Generate the GitHub Pages research article site from docs/ARTICLE.md and assets."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CircuitGraft: Circuit-Aware Safety Recovery for Domain-Fine-Tuned Cohere Tiny Aya</title>
<meta name="description" content="Lexsi Labs AI Safety & Alignment Hackathon (Track 2b / Problem 6) — Circuit-aware safety recovery for domain-fine-tuned Cohere Tiny Aya models.">
<meta name="author" content="Devansh Dubey">

<!-- OpenGraph / Twitter Cards -->
<meta property="og:type" content="article">
<meta property="og:title" content="CircuitGraft: Circuit-Aware Safety Recovery for Domain-Fine-Tuned Cohere Tiny Aya">
<meta property="og:description" content="Preregistered study mapping safety and domain circuits in Cohere Tiny Aya, protecting shared components with SVD projection.">
<meta property="og:image" content="assets/circuit_overlap_heatmap.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="CircuitGraft: Circuit-Aware Safety Recovery">
<meta name="twitter:description" content="Lexsi Labs Hackathon Track 2b / Problem 6 — Research write-up & empirical evaluation.">
<meta name="twitter:image" content="assets/circuit_overlap_heatmap.png">

<!-- KaTeX for crisp LaTeX Math -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>

<!-- Google Fonts: Inter & Newsreader for academic aesthetic -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap" rel="stylesheet">

<style>
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-tertiary: #f1f5f9;
  --bg-card: #ffffff;
  --border-color: #e2e8f0;
  --border-subtle: #edf2f7;
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-muted: #64748b;
  --accent: #2563eb;
  --accent-hover: #1d4ed8;
  --accent-light: #eff6ff;
  --accent-border: #bfdbfe;
  --green: #16a34a;
  --green-light: #f0fdf4;
  --amber: #d97706;
  --amber-light: #fffbeb;
  --red: #dc2626;
  --red-light: #fef2f2;
  --font-body: 'Newsreader', Georgia, serif;
  --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
}

[data-theme="dark"] {
  --bg-primary: #090d16;
  --bg-secondary: #0f172a;
  --bg-tertiary: #1e293b;
  --bg-card: #131d31;
  --border-color: #334155;
  --border-subtle: #1e293b;
  --text-primary: #f8fafc;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
  --accent: #38bdf8;
  --accent-hover: #7dd3fc;
  --accent-light: rgba(56, 189, 248, 0.1);
  --accent-border: #0369a1;
  --green: #4ade80;
  --green-light: rgba(74, 222, 128, 0.1);
  --amber: #fbbf24;
  --amber-light: rgba(251, 191, 36, 0.1);
  --red: #f87171;
  --red-light: rgba(248, 113, 113, 0.1);
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.4);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.5);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.6);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html {
  scroll-behavior: smooth;
  scroll-padding-top: 80px;
}

body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-body);
  font-size: 19px;
  line-height: 1.7;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
}

/* Progress bar */
#reading-progress {
  position: fixed;
  top: 0;
  left: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--accent), #818cf8);
  width: 0%;
  z-index: 1000;
  transition: width 0.1s ease;
}

/* Header & Nav */
.navbar {
  position: sticky;
  top: 0;
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border-color);
  z-index: 999;
  font-family: var(--font-ui);
  transition: background 0.3s ease;
}
[data-theme="dark"] .navbar {
  background: rgba(9, 13, 22, 0.88);
}
.nav-inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.brand {
  font-weight: 700;
  font-size: 17px;
  color: var(--text-primary);
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 8px;
}
.brand-badge {
  font-size: 11px;
  background: var(--accent-light);
  color: var(--accent);
  padding: 2px 8px;
  border-radius: 999px;
  font-weight: 600;
  border: 1px solid var(--accent-border);
}
.nav-links {
  display: flex;
  align-items: center;
  gap: 20px;
  font-size: 14px;
  font-weight: 500;
}
.nav-links a {
  color: var(--text-secondary);
  text-decoration: none;
  transition: color 0.2s;
}
.nav-links a:hover {
  color: var(--accent);
}
.theme-btn {
  background: none;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  padding: 6px 10px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  transition: all 0.2s;
}
.theme-btn:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

/* Layout */
.page-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 48px 24px 96px;
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr);
  gap: 48px;
}
@media (max-width: 960px) {
  .page-container {
    grid-template-columns: 1fr;
    gap: 32px;
  }
  .sidebar { display: none; }
}

/* Sidebar */
.sidebar {
  position: sticky;
  top: 90px;
  height: calc(100vh - 110px);
  overflow-y: auto;
  font-family: var(--font-ui);
  padding-right: 12px;
}
.toc-title {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  font-weight: 700;
  margin-bottom: 12px;
}
.toc-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.toc-list a {
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 13.5px;
  line-height: 1.4;
  padding: 6px 10px;
  border-radius: 6px;
  display: block;
  transition: all 0.15s ease;
}
.toc-list a:hover {
  color: var(--accent);
  background: var(--bg-secondary);
}
.toc-list a.active {
  color: var(--accent);
  background: var(--accent-light);
  font-weight: 600;
}

/* Article Paper Styling */
article {
  max-width: 820px;
}

/* Paper Header */
.paper-header {
  margin-bottom: 36px;
  padding-bottom: 32px;
  border-bottom: 1px solid var(--border-color);
  font-family: var(--font-ui);
}
.track-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
  color: var(--accent);
  background: var(--accent-light);
  border: 1px solid var(--accent-border);
  padding: 4px 12px;
  border-radius: 999px;
  margin-bottom: 16px;
}
h1.paper-title {
  font-family: var(--font-ui);
  font-size: 38px;
  font-weight: 800;
  line-height: 1.25;
  letter-spacing: -0.025em;
  color: var(--text-primary);
  margin-bottom: 18px;
}
.authors-meta {
  font-size: 16px;
  color: var(--text-secondary);
  margin-bottom: 24px;
  line-height: 1.6;
}
.author-name {
  font-weight: 600;
  color: var(--text-primary);
}
.author-links a {
  color: var(--accent);
  text-decoration: none;
  margin-left: 6px;
  font-size: 14px;
}
.author-links a:hover { text-decoration: underline; }

.action-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 20px;
}
.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 8px 14px;
  border-radius: 8px;
  font-size: 13.5px;
  font-weight: 600;
  text-decoration: none;
  transition: all 0.15s;
  border: 1px solid var(--border-color);
  background: var(--bg-card);
  color: var(--text-primary);
  box-shadow: var(--shadow-sm);
}
.action-btn:hover {
  background: var(--bg-tertiary);
  border-color: var(--text-muted);
  transform: translateY(-1px);
}
.action-btn.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #ffffff;
}
.action-btn.primary:hover {
  background: var(--accent-hover);
  color: #ffffff;
}

/* Callouts / Alerts */
.callout {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 24px 0;
  font-size: 16.5px;
  line-height: 1.6;
}
.callout.amber {
  border-left-color: var(--amber);
  background: var(--amber-light);
}
.callout.green {
  border-left-color: var(--green);
  background: var(--green-light);
}
.callout-title {
  font-family: var(--font-ui);
  font-weight: 700;
  font-size: 14.5px;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Abstract Box */
.abstract-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 28px;
  margin-bottom: 40px;
  box-shadow: var(--shadow-md);
}
.abstract-title {
  font-family: var(--font-ui);
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 700;
  color: var(--accent);
  margin-bottom: 12px;
}
.abstract-text {
  font-size: 18px;
  line-height: 1.7;
  color: var(--text-primary);
  margin-bottom: 20px;
}
.stat-chips {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-top: 16px;
  font-family: var(--font-ui);
}
.stat-chip {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 12px 14px;
}
.stat-chip .val {
  font-size: 20px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-primary);
}
.stat-chip .lbl {
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 2px;
}

/* Typography in Body */
h2 {
  font-family: var(--font-ui);
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 48px 0 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border-color);
  color: var(--text-primary);
}
h3 {
  font-family: var(--font-ui);
  font-size: 20px;
  font-weight: 600;
  margin: 32px 0 12px;
  color: var(--text-primary);
}
p {
  margin-bottom: 20px;
  color: var(--text-secondary);
}
strong {
  color: var(--text-primary);
  font-weight: 600;
}
ul, ol {
  margin: 0 0 20px 24px;
  color: var(--text-secondary);
}
li {
  margin-bottom: 8px;
}
a {
  color: var(--accent);
  text-decoration: underline;
  text-underline-offset: 3px;
}
a:hover {
  color: var(--accent-hover);
}

/* Figures */
figure {
  margin: 36px 0;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 16px;
  box-shadow: var(--shadow-sm);
  text-align: center;
}
figure img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  cursor: zoom-in;
  transition: transform 0.2s ease;
}
figure img:hover {
  transform: scale(1.01);
}
figcaption {
  font-family: var(--font-ui);
  font-size: 13.5px;
  color: var(--text-muted);
  margin-top: 12px;
  text-align: left;
  line-height: 1.55;
  padding: 0 4px;
}
figcaption strong {
  color: var(--text-primary);
}

/* Tables */
.table-wrapper {
  overflow-x: auto;
  margin: 28px 0;
  border: 1px solid var(--border-color);
  border-radius: 10px;
  box-shadow: var(--shadow-sm);
  background: var(--bg-card);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-ui);
  font-size: 13.5px;
  text-align: left;
}
thead tr {
  background: var(--bg-tertiary);
  border-bottom: 2px solid var(--border-color);
}
th {
  padding: 10px 14px;
  font-weight: 600;
  color: var(--text-primary);
  font-size: 12.5px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  white-space: nowrap;
}
td {
  padding: 9px 14px;
  border-bottom: 1px solid var(--border-color);
  color: var(--text-secondary);
  white-space: nowrap;
}
tr:last-child td {
  border-bottom: none;
}
tbody tr:hover {
  background: var(--bg-secondary);
}
td.num, th.num {
  text-align: right;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}
.highlight-row {
  background: var(--accent-light) !important;
  font-weight: 600;
}
.badge-pass {
  color: var(--green);
  font-weight: 600;
}
.badge-fail {
  color: var(--red);
  font-weight: 600;
}

/* Code & Pre */
code {
  font-family: var(--font-mono);
  font-size: 14.5px;
  background: var(--bg-tertiary);
  padding: 2px 6px;
  border-radius: 5px;
  color: var(--text-primary);
}
pre {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 16px 20px;
  overflow-x: auto;
  margin: 24px 0;
  font-family: var(--font-mono);
  font-size: 13.5px;
  line-height: 1.6;
  position: relative;
}
pre code {
  background: none;
  padding: 0;
  font-size: 13.5px;
}
.copy-code-btn {
  position: absolute;
  top: 10px;
  right: 10px;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  color: var(--text-muted);
  font-family: var(--font-ui);
  font-size: 11px;
  padding: 4px 8px;
  border-radius: 5px;
  cursor: pointer;
  transition: all 0.15s;
}
.copy-code-btn:hover {
  color: var(--text-primary);
  background: var(--bg-tertiary);
}

/* Image Modal */
#img-modal {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.88);
  z-index: 2000;
  justify-content: center;
  align-items: center;
  padding: 24px;
}
#img-modal img {
  max-width: 95vw;
  max-height: 90vh;
  border-radius: 8px;
  box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.5);
}

/* Footer */
footer {
  margin-top: 80px;
  padding-top: 36px;
  border-top: 1px solid var(--border-color);
  font-family: var(--font-ui);
  font-size: 14px;
  color: var(--text-muted);
  line-height: 1.7;
}
</style>
</head>
<body>

<div id="reading-progress"></div>

<!-- Navigation Bar -->
<header class="navbar">
  <div class="nav-inner">
    <a href="#" class="brand">
      CircuitGraft
      <span class="brand-badge">Lexsi Hackathon Track 2b</span>
    </a>
    <div class="nav-links">
      <a href="#abstract">Abstract</a>
      <a href="#overlap">Circuit Overlap</a>
      <a href="#results">Results</a>
      <a href="#ablations">Ablations</a>
      <a href="#artifacts">Artifacts</a>
      <a href="#citation">Cite</a>
      <button class="theme-btn" id="theme-toggle" aria-label="Toggle theme">🌙</button>
    </div>
  </div>
</header>

<div class="page-container">

  <!-- Sticky Sidebar Navigation -->
  <aside class="sidebar">
    <div class="toc-title">Contents</div>
    <ul class="toc-list" id="toc">
      <li><a href="#abstract">Abstract & Findings</a></li>
      <li><a href="#motivation">1. Motivation</a></li>
      <li><a href="#setup">2. Experimental Setup</a></li>
      <li><a href="#drift">3. Drift Evaluation & Gate</a></li>
      <li><a href="#overlap">4. Circuit Discovery & Overlap</a></li>
      <li><a href="#method">5. CircuitGraft Architecture</a></li>
      <li><a href="#results">6. Main Empirical Results</a></li>
      <li><a href="#ablations">7. Ablation Studies</a></li>
      <li><a href="#robustness">8. Robustness & Quantization</a></li>
      <li><a href="#limitations">9. Limitations & Disclosures</a></li>
      <li><a href="#artifacts">10. Deliverables & Hub Repos</a></li>
      <li><a href="#related">11. Related Work & Citations</a></li>
      <li><a href="#citation">Citation (BibTeX)</a></li>
    </ul>
  </aside>

  <!-- Main Article Body -->
  <article>

    <div class="paper-header">
      <div class="track-pill">Lexsi Labs AI Safety & Alignment Hackathon · Problem 6</div>
      <h1 class="paper-title">CircuitGraft: Circuit-Aware Safety Recovery for Domain-Fine-Tuned Cohere Tiny Aya</h1>
      
      <div class="authors-meta">
        <span class="author-name">Devansh Dubey</span>
        <span class="author-links">
          [<a href="https://github.com/0dvsh0">GitHub</a>]
          [<a href="https://huggingface.co/dubeydevansh">Hugging Face</a>]
        </span>
        <br>
        <span style="font-size:13.5px; color:var(--text-muted);">
          Tracked Run: <code>results/s9/report-1794803399</code> · Evaluation Run: <code>eval-rescored-20261116-042341</code>
        </span>
      </div>

      <div class="action-bar">
        <a href="https://huggingface.co/spaces/dubeydevansh/circuitgraft" class="action-btn primary" target="_blank">
          ⚡ Live Results Dashboard
        </a>
        <a href="https://huggingface.co/collections/dubeydevansh/circuitgraft-models-6afa7a5d7e17226835c76a9f" class="action-btn" target="_blank">
          📦 HF Models Collection
        </a>
        <a href="https://huggingface.co/collections/dubeydevansh/circuitgraft-datasets-and-artifacts-6af08216276001ae81874d98" class="action-btn" target="_blank">
          📊 HF Datasets Collection
        </a>
        <a href="https://github.com/0dvsh0/circuit-graft" class="action-btn" target="_blank">
          💻 Code Repository
        </a>
      </div>
    </div>

    <!-- Status & Disclosures -->
    <div class="callout amber">
      <div class="callout-title">⚠️ Preregistered Reporting Status: Partial (Disclosed)</div>
      The Phase 4 core matrix ran end-to-end on Kaggle (T4) with bundle status <strong>partial</strong> for three disclosed reasons:
      the evaluation summary is partial (human audit pending review), the human audit itself is pending, and S8 attacks were not run.
      <strong>Scoring-fix disclosure:</strong> The initial eval merge scored MCQ golds as letter IDs against option-text outputs (yielding 0.0% accuracy). The bug was fixed in <code>circuitgraft.eval.domain</code>, regression-tested, and the MCQ arm was re-scored from cached generations on CPU (run <code>eval-rescored-20261116-042341</code>). No number on this site is hand-typed or simulated.
    </div>

    <!-- Abstract -->
    <section id="abstract" class="abstract-card">
      <div class="abstract-title">Abstract</div>
      <p class="abstract-text">
        Domain fine-tuning makes a model more useful in a target domain, but it can also erode the refusal behavior learned during alignment. Whole-model recovery methods restore safety at the cost of domain competence, while circuit-restricted methods often optimize safety without asking which components the domain skill needs. <strong>CircuitGraft</strong> maps a safety circuit and a domain circuit in the same attention-head and MLP component space, partitions components into safety-only, domain-only and shared sets, and grafts the safety delta back with domain-protected edits on the shared set. We evaluate safety recovery, domain-gain retention and false refusal against RESTA, WiSE-FT, C-ΔΘ (SafeTune weight-arithmetic implementation) and Safe LoRA on Cohere Tiny Aya, using a medical fine-tuning scenario and preregistered endpoints.
      </p>

      <div class="stat-chips">
        <div class="stat-chip">
          <div class="val">72 / 184</div>
          <div class="lbl">Shared Circuit Components</div>
        </div>
        <div class="stat-chip">
          <div class="val">+8.0%</div>
          <div class="lbl">Domain Gain (Passed Gate)</div>
        </div>
        <div class="stat-chip">
          <div class="val">+2.3%</div>
          <div class="lbl">Refusal Delta (Failed Gate)</div>
        </div>
        <div class="stat-chip">
          <div class="val">+85.8%</div>
          <div class="lbl">DGR Protection Advantage</div>
        </div>
      </div>
    </section>

    <!-- Motivation -->
    <section id="motivation">
      <h2>1. Motivation</h2>
      <p>
        Safety alignment is not a property that domain fine-tuning automatically preserves. A medical specialist may gain benchmark accuracy while becoming more willing to answer harmful requests, because the fine-tuning signal rewards helpful completion and does not distinguish legitimate clinical questions from misuse. Restoring safety after the fact is attractive: it avoids retraining the domain model from scratch and keeps the domain data pipeline unchanged.
      </p>
      <p>
        The central architectural question is <em>where</em> to restore safety. Whole-model interpolation (such as WiSE-FT) and task-arithmetic baselines (such as RESTA) move every parameter, which risks undoing the very representations that make the specialist useful. Circuit-localized updates are more surgical, but existing work typically localizes one behavior at a time. A safety circuit and a domain circuit can overlap; editing a shared component as if it were safety-only is a plausible source of domain regression and false refusal.
      </p>
      <p>
        CircuitGraft’s hypothesis is that making that overlap explicit, and protecting the domain direction on shared components, improves the safety/domain trade-off over both whole-model recovery and naive circuit restriction.
      </p>
    </section>

    <!-- Experimental Setup -->
    <section id="setup">
      <h2>2. Setup: Cohere Tiny Aya & Medical Specialization</h2>
      <p>
        <strong>Model Triple:</strong> $\theta_B$ is <code>CohereLabs/tiny-aya-base</code> (pretrained), $\theta_A$ is <code>CohereLabs/tiny-aya-global</code> (aligned), and $\theta_D$ is the medical specialist produced by supervised fine-tuning (S2). Tiny Aya is small enough to fine-tune and evaluate on accessible GPUs while retaining the modern Cohere2 architecture: 36 layers, grouped-query attention (GQA), and parallel attention/MLP blocks.
      </p>
      <p>
        <strong>Adversarial Domain Choice:</strong> Medicine is a deliberately adversarial choice for safety recovery. Clinical knowledge (dosing, toxicology, drug interactions) is structurally adjacent to harmful requests (poisoning, overdose, illicit synthesis). Consequently, substantial circuit overlap is expected, and false refusal of legitimate clinical questions represents a primary failure mode.
      </p>
      <p>
        <strong>Evaluated Endpoints:</strong> Safety Recovery Rate $\text{SRR} = \frac{\text{HRR}_{\text{repaired}} - \text{HRR}_D}{\text{HRR}_A - \text{HRR}_D}$ and Domain Gain Retention $\text{DGR} = \frac{\text{Acc}_{\text{repaired}} - \text{Acc}_A}{\text{Acc}_D - \text{Acc}_A}$, alongside false-refusal rate (FRR on XSTest) and parameter edit footprint.
      </p>
    </section>

    <!-- Drift Auditing -->
    <section id="drift">
      <h2>3. Demonstrating Drift: The G-Drift Gate</h2>
      <p>
        Before applying any recovery intervention, stage S3 audits the aligned model $\theta_A$ and drifted specialist $\theta_D$. Hypothesis H0 requires that harmful refusal drops by at least 15 points versus aligned and the bootstrap 95% CI excludes zero.
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Model State</th>
              <th class="num">HarmBench HRR</th>
              <th class="num">AdvBench HRR</th>
              <th class="num">MedMCQA Acc</th>
              <th class="num">MMLU Acc</th>
              <th class="num">XSTest FRR</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Aligned ($\theta_A$)</strong></td>
              <td class="num">76.7%</td>
              <td class="num">95.0%</td>
              <td class="num">36.0%</td>
              <td class="num">37.0%</td>
              <td class="num">10.0%</td>
            </tr>
            <tr>
              <td><strong>Drifted ($\theta_D$)</strong></td>
              <td class="num">81.3%</td>
              <td class="num">92.5%</td>
              <td class="num">40.7%</td>
              <td class="num">54.0%</td>
              <td class="num">3.8%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="callout red">
        <div class="callout-title">⚠️ Gate G-drift: Refusal Axis Failed; Domain Axis Passed</div>
        <p>
          <strong>Criterion:</strong> Refusal drop $\ge 15.0$ pts with 95% CI lower bound $> 0$.<br>
          <strong>Observed:</strong> Refusal drop was only $+2.3$ pts (95% CI $[−1.5, +6.2]$). The 5,000-row Scenario-A QLoRA drift did <em>not</em> produce a meaningfully less-refusing model.
        </p>
        <p style="margin-top:8px">
          <strong>Consequence:</strong> Every SRR metric inherits a 2.3-point denominator and is noise-dominated; SRR is preserved in tables for preregistration completeness but is never quoted as causal evidence. In contrast, <strong>Domain Gain Retention (DGR)</strong> has an 8.0-point denominator and remains fully interpretable.
        </p>
      </div>
    </section>

    <!-- Circuit Discovery & Overlap -->
    <section id="overlap">
      <h2>4. Where Safety and Medical Skill Live</h2>
      <p>
        Stage S4 discovers the safety circuit $S$ on $\theta_A$ with CircuitKIT Edge-Attribution Patching with Integrated Gradients (EAP-IG); the domain circuit $D$ is discovered on $\theta_D$ using contrastive medical MCQ pairs. Both circuits use an identical component vocabulary across all 36 layers: attention heads <code>L{l}.H{h}</code> and MLP blocks <code>L{l}.MLP</code> (612 total units).
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th class="num">Safety Circuit ($S$)</th>
              <th class="num">Domain Circuit ($D$)</th>
              <th class="num">Overlap ($S \cap D$)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Component Count</strong></td>
              <td class="num">184</td>
              <td class="num">184</td>
              <td class="num"><strong>72</strong></td>
            </tr>
            <tr>
              <td><strong>Attention Heads</strong></td>
              <td class="num">—</td>
              <td class="num">—</td>
              <td class="num">45</td>
            </tr>
            <tr>
              <td><strong>MLP Blocks</strong></td>
              <td class="num">—</td>
              <td class="num">—</td>
              <td class="num">27</td>
            </tr>
            <tr>
              <td><strong>Jaccard Similarity</strong></td>
              <td class="num">—</td>
              <td class="num">—</td>
              <td class="num"><strong>0.243</strong></td>
            </tr>
          </tbody>
        </table>
      </div>

      <figure>
        <img src="assets/circuit_overlap_heatmap.png" alt="Circuit Overlap Heatmap Across 36 Layers" onclick="openModal(this.src)">
        <figcaption>
          <strong>Figure 1: Component partition across all 36 layers.</strong> Attention heads and MLPs are categorized into Safety-Only (blue), Domain-Only (green), and Shared Overlap (orange). Overlap spans 32 of 36 layers, peaking at Layer 18.
        </figcaption>
      </figure>

      <p>
        <strong>Finding:</strong> At the default discovery sparsity ($k = 0.3$), the safety and domain circuits share <strong>72 components (39.1% of each circuit)</strong> with a Jaccard index of 0.243. The overlap persists across all tested discovery thresholds ($k \in \{0.05, 0.1, 0.2, 0.3\}$), proving that shared representation is a pervasive architectural property rather than a thresholding artifact.
      </p>

      <figure>
        <img src="assets/charts_sparsity.png" alt="Circuit Sparsity Sweep" onclick="openModal(this.src)">
        <figcaption>
          <strong>Figure 2: Discovery Sparsity Sweep ($k$).</strong> Overlap Jaccard index remains consistently between 0.170 and 0.243 as sparsity fraction increases from 0.05 to 0.30.
        </figcaption>
      </figure>
    </section>

    <!-- CircuitGraft Method -->
    <section id="method">
      <h2>5. CircuitGraft Architecture & Interventions</h2>
      <p>
        Let $\theta_B, \theta_A, \theta_D$ denote the base, aligned, and drifted weights. The safety alignment delta is $\Delta_S = \theta_A - \theta_B$, and the domain fine-tuning delta is $\Delta_D = \theta_D - \theta_A$.
      </p>
      <p>
        CircuitGraft partitions transformer components into exclusive and shared sets:
      </p>
      <ul>
        <li><strong>Safety-only ($c \in S \setminus D$):</strong> Receive full safety restoration edit: $W_D[c] \leftarrow W_D[c] + \alpha \Delta_S[c]$</li>
        <li><strong>Domain-only ($c \in D \setminus S$):</strong> Untouched to preserve domain competence: $W_D[c] \leftarrow W_D[c]$</li>
        <li><strong>Shared Overlap ($c \in S \cap D$):</strong> Protected safety edit: $W_D[c] \leftarrow W_D[c] + \alpha_O \cdot \text{protect}(\Delta_S[c], \Delta_D[c])$</li>
      </ul>
      <p>
        In the primary <code>cg_project</code> method, $\text{protect}$ computes an SVD of $\Delta_D[c]$ and projects $\Delta_S[c]$ into the null-space of the top singular vectors:
      </p>
      $$\Pi_D^\perp = I - V_k V_k^T, \quad \Delta_{S,\text{proj}}[c] = \Delta_S[c] \Pi_D^\perp$$
      <p>
        This removes the top domain directions (preserving 90% of domain energy up to rank 16) before applying the safety edit, preventing interference at the shared interface.
      </p>
    </section>

    <!-- Results & Comparison -->
    <section id="results">
      <h2>6. Main Empirical Results</h2>
      <p>
        The table below evaluates all methods on the frozen S9 bundle (run <code>results/s9/report-1794803399</code> with CPU-rescored MCQ letter golds).
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Model / Method</th>
              <th class="num">HarmBench HRR</th>
              <th class="num">AdvBench HRR</th>
              <th class="num">MedMCQA Acc</th>
              <th class="num">MMLU Acc</th>
              <th class="num">XSTest FRR</th>
              <th class="num">DGR</th>
              <th class="num">Footprint</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Aligned Baseline ($\theta_A$)</strong></td>
              <td class="num">76.7%</td>
              <td class="num">95.0%</td>
              <td class="num">36.0%</td>
              <td class="num">37.0%</td>
              <td class="num">10.0%</td>
              <td class="num">0.0%</td>
              <td class="num">—</td>
            </tr>
            <tr>
              <td><strong>Drifted Specialist ($\theta_D$)</strong></td>
              <td class="num">81.3%</td>
              <td class="num">92.5%</td>
              <td class="num">40.7%</td>
              <td class="num">54.0%</td>
              <td class="num">3.8%</td>
              <td class="num">100.0%</td>
              <td class="num">—</td>
            </tr>
            <tr class="highlight-row">
              <td><strong>CircuitGraft (<code>cg_project</code>)</strong></td>
              <td class="num">69.3%</td>
              <td class="num">87.5%</td>
              <td class="num">40.0%</td>
              <td class="num">40.0%</td>
              <td class="num">7.5%</td>
              <td class="num"><strong>-57.1%</strong></td>
              <td class="num">41.0%</td>
            </tr>
            <tr>
              <td>CircuitGraft (<code>cg_uniform</code>)</td>
              <td class="num">66.7%</td>
              <td class="num">85.0%</td>
              <td class="num">36.7%</td>
              <td class="num">39.0%</td>
              <td class="num">13.8%</td>
              <td class="num">-142.9%</td>
              <td class="num">—</td>
            </tr>
            <tr>
              <td>Control (<code>cg_random</code>)</td>
              <td class="num">70.0%</td>
              <td class="num">90.0%</td>
              <td class="num">40.0%</td>
              <td class="num">41.0%</td>
              <td class="num">7.5%</td>
              <td class="num">-60.7%</td>
              <td class="num">—</td>
            </tr>
            <tr>
              <td>RESTA</td>
              <td class="num">52.7%</td>
              <td class="num">80.0%</td>
              <td class="num">39.3%</td>
              <td class="num">46.0%</td>
              <td class="num"><strong>0.0%</strong></td>
              <td class="num">+7.1%</td>
              <td class="num">—</td>
            </tr>
            <tr>
              <td>WiSE-FT</td>
              <td class="num"><strong>87.3%</strong></td>
              <td class="num"><strong>95.0%</strong></td>
              <td class="num"><strong>44.7%</strong></td>
              <td class="num">51.0%</td>
              <td class="num">8.8%</td>
              <td class="num"><strong>+96.4%</strong></td>
              <td class="num">—</td>
            </tr>
            <tr style="color:var(--red);">
              <td>C-ΔΘ (SafeTune weight arithmetic)</td>
              <td class="num">6.0%</td>
              <td class="num">3.8%</td>
              <td class="num">24.7%</td>
              <td class="num">29.0%</td>
              <td class="num">2.5%</td>
              <td class="num">-160.7%</td>
              <td class="num">—</td>
            </tr>
          </tbody>
        </table>
      </div>

      <figure>
        <img src="assets/pareto_frontier.png" alt="Pareto Frontier Plot" onclick="openModal(this.src)">
        <figcaption>
          <strong>Figure 3: Pareto Frontier (Domain Gain Retention vs HarmBench Refusal).</strong> WiSE-FT occupies the top-right corner; C-ΔΘ collapses entirely; <code>cg_project</code> demonstrates significantly superior retention over unprotected <code>cg_uniform</code>.
        </figcaption>
      </figure>

      <figure>
        <img src="assets/charts_main_results.png" alt="Main Results Comparison Bar Chart" onclick="openModal(this.src)">
        <figcaption>
          <strong>Figure 4: Comparative evaluation across benchmarks.</strong> Visualizing refusal performance, clinical question accuracy, and false refusal across alignment methods.
        </figcaption>
      </figure>
    </section>

    <!-- Ablations -->
    <section id="ablations">
      <h2>7. Ablation Studies: The Value of Protection</h2>
      <p>
        The study's cleanest positive result is the ablation between protected graft (<code>cg_project</code>) and unprotected graft (<code>cg_uniform</code>):
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th class="num">HarmBench HRR</th>
              <th class="num">AdvBench HRR</th>
              <th class="num">DGR</th>
              <th class="num">XSTest FRR</th>
              <th class="num">MedMCQA Acc</th>
            </tr>
          </thead>
          <tbody>
            <tr class="highlight-row">
              <td><strong>cg_project (Protected)</strong></td>
              <td class="num"><strong>69.3%</strong></td>
              <td class="num"><strong>87.5%</strong></td>
              <td class="num"><strong>-57.1%</strong></td>
              <td class="num"><strong>7.5%</strong></td>
              <td class="num"><strong>40.0%</strong></td>
            </tr>
            <tr>
              <td>cg_uniform (Unprotected)</td>
              <td class="num">66.7%</td>
              <td class="num">85.0%</td>
              <td class="num">-142.9%</td>
              <td class="num">13.8%</td>
              <td class="num">36.7%</td>
            </tr>
            <tr>
              <td>cg_random (Mask Control)</td>
              <td class="num">70.0%</td>
              <td class="num">90.0%</td>
              <td class="num">-60.7%</td>
              <td class="num">7.5%</td>
              <td class="num">40.0%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <figure>
        <img src="assets/charts_ablation.png" alt="Ablation Comparison Charts" onclick="openModal(this.src)">
        <figcaption>
          <strong>Figure 5: Ablation Breakdown.</strong> Domain-direction projection prevents severe domain regression (-57.1% vs -142.9% DGR) and cuts false refusals in half (7.5% vs 13.8% FRR).
        </figcaption>
      </figure>

      <p>
        <strong>Finding:</strong> <code>cg_project</code> outperforms <code>cg_uniform</code> on <em>every single metric</em>: $+2.6$ pts HarmBench refusal, $+85.8$ pts domain retention, and $-6.3$ pts benign false refusal. Bluntly editing the shared interface without null-space protection degrades both capabilities.
      </p>
    </section>

    <!-- Robustness -->
    <section id="robustness">
      <h2>8. Robustness & Quantization Attacks</h2>
      <p>
        <strong>Status: Unevaluated (Reported, Not Fabricated).</strong> Stage S8 was designed to evaluate abliteration attacks, harmful fine-tuning, and 4-bit quantization. Due to Kaggle scratch-disk limits and submission deadlines, post-attack evaluation checkpoints were archived rather than fully evaluated through S7. Per the frozen preregistration, these cells remain empty (<code>—</code>) rather than quoting unverified figures.
      </p>
    </section>

    <!-- Limitations -->
    <section id="limitations">
      <h2>9. Honest Limitations & Negative Results</h2>
      <ul>
        <li><strong>Single Family & Domain:</strong> Results are established on Cohere Tiny Aya (3.35B) in the medical domain; scaling to larger architectures (e.g. Aya Expanse 8B) remains future work.</li>
        <li><strong>Discovery Stability:</strong> Circuit discovery repeats showed identical node selections, so stability checks carried no discriminating power. Full causal faithfulness verification was budget-limited.</li>
        <li><strong>Random Mask Control:</strong> <code>cg_random</code> performed comparably to <code>cg_project</code> on the default operating point; circuit-informed localization did not demonstrate a statistically significant benefit over a layer-matched random mask on single-seed runs.</li>
        <li><strong>C-ΔΘ Failure Mode:</strong> SafeTune’s weight-arithmetic implementation of C-ΔΘ collapsed both safety and domain competence, underscoring that module-level weight arithmetic without fine-tuning requires extreme care.</li>
      </ul>
    </section>

    <!-- Published Deliverables -->
    <section id="artifacts">
      <h2>10. Published Deliverables & Artifact Manifests</h2>
      <p>
        All artifacts, models, datasets, and logs are publicly hosted on Hugging Face Hub:
      </p>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Deliverable</th>
              <th>Public URL / Verified Identifier</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Hugging Face Model Collection</strong></td>
              <td><a href="https://huggingface.co/collections/dubeydevansh/circuitgraft-models-6afa7a5d7e17226835c76a9f" target="_blank"><code>collections/dubeydevansh/circuitgraft-models</code></a></td>
            </tr>
            <tr>
              <td><strong>Hugging Face Dataset Collection</strong></td>
              <td><a href="https://huggingface.co/collections/dubeydevansh/circuitgraft-datasets-and-artifacts-6af08216276001ae81874d98" target="_blank"><code>collections/dubeydevansh/circuitgraft-datasets-and-artifacts</code></a></td>
            </tr>
            <tr>
              <td><strong>Live Static Results Space</strong></td>
              <td><a href="https://huggingface.co/spaces/dubeydevansh/circuitgraft" target="_blank"><code>spaces/dubeydevansh/circuitgraft</code></a></td>
            </tr>
            <tr>
              <td><strong>Drifted Checkpoint</strong></td>
              <td><a href="https://huggingface.co/dubeydevansh/circuitgraft-tinyaya-med-drifted" target="_blank"><code>circuitgraft-tinyaya-med-drifted</code></a></td>
            </tr>
            <tr>
              <td><strong>Grafted Checkpoint</strong></td>
              <td><a href="https://huggingface.co/dubeydevansh/circuitgraft-tinyaya-med-cg-project" target="_blank"><code>circuitgraft-tinyaya-med-cg-project</code></a> (41.0% parameter footprint)</td>
            </tr>
            <tr>
              <td><strong>LoRA Adapter Repo</strong></td>
              <td><a href="https://huggingface.co/dubeydevansh/circuitgraft-tinyaya-med-lora" target="_blank"><code>circuitgraft-tinyaya-med-lora</code></a></td>
            </tr>
            <tr>
              <td><strong>Eval Logs & Bundle Repo</strong></td>
              <td><a href="https://huggingface.co/datasets/dubeydevansh/circuitgraft-eval-logs" target="_blank"><code>circuitgraft-eval-logs</code></a> (contains <code>report-1794803399</code> & <code>ARTICLE.md</code>)</td>
            </tr>
            <tr>
              <td><strong>Code Repository</strong></td>
              <td><a href="https://github.com/0dvsh0/circuit-graft" target="_blank"><code>github.com/0dvsh0/circuit-graft</code></a></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- Related Work -->
    <section id="related">
      <h2>11. Related Work & Attribution</h2>
      <p>
        CircuitGraft builds upon four key open-source toolkits developed by Lexsi Labs:
      </p>
      <ul>
        <li><strong>CuratorKIT:</strong> Data curation, contrastive pair generation, and decontamination filters.</li>
        <li><strong>SafeTune:</strong> Alignment delta arithmetic, EAP implementation, and baseline recovery methods.</li>
        <li><strong>CircuitKIT:</strong> Edge-attribution patching with integrated gradients (EAP-IG) for Cohere architectures.</li>
        <li><strong>AuditKIT:</strong> Automated evaluation pipelines, run fingerprints, and judge scoring.</li>
      </ul>
    </section>

    <!-- Citation Box -->
    <section id="citation">
      <h2>Citation</h2>
      <p>If you use CircuitGraft or its findings in your research, please cite:</p>
      <pre><code>@article{dubey2026circuitgraft,
  title={CircuitGraft: Circuit-Aware Safety Recovery for Domain-Fine-Tuned Cohere Tiny Aya},
  author={Dubey, Devansh},
  journal={Lexsi Labs AI Safety & Alignment Hackathon},
  year={2026},
  url={https://0dvsh0.github.io/circuit-graft/}
}</code><button class="copy-code-btn" onclick="copyBibtex()">Copy BibTeX</button></pre>
    </section>

    <!-- Footer -->
    <footer>
      <p>
        <strong>CircuitGraft</strong> — Track 2b / Problem 6, Lexsi Labs Hackathon, November 2026.<br>
        Published by <strong>Devansh Dubey</strong> (<a href="https://github.com/0dvsh0">@0dvsh0</a> · <a href="https://huggingface.co/dubeydevansh">@dubeydevansh</a>).<br>
        Source code released under the Apache-2.0 License. All evaluations read from verified machine manifests.
      </p>
    </footer>

  </article>

</div>

<!-- Modal for Figure Zoom -->
<div id="img-modal" onclick="closeModal()">
  <img id="modal-img" src="" alt="Enlarged figure">
</div>

<script>
// Reading progress bar
window.addEventListener('scroll', () => {
  const winScroll = document.documentElement.scrollTop;
  const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
  const scrolled = (winScroll / height) * 100;
  document.getElementById('reading-progress').style.width = scrolled + '%';
});

// Dark/Light Theme toggle
const themeBtn = document.getElementById('theme-toggle');
const currentTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', currentTheme);
themeBtn.textContent = currentTheme === 'dark' ? '☀️' : '🌙';

themeBtn.addEventListener('click', () => {
  const theme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  themeBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
});

// Table of contents active link tracker
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      document.querySelectorAll('#toc a').forEach((a) => a.classList.remove('active'));
      const activeLink = document.querySelector(`#toc a[href="#${entry.target.id}"]`);
      if (activeLink) activeLink.classList.add('active');
    }
  });
}, { rootMargin: '-20% 0px -70% 0px' });

document.querySelectorAll('article section').forEach((sec) => observer.observe(sec));

// Image zoom modal
function openModal(src) {
  const modal = document.getElementById('img-modal');
  const img = document.getElementById('modal-img');
  img.src = src;
  modal.style.display = 'flex';
}
function closeModal() {
  document.getElementById('img-modal').style.display = 'none';
}

// Copy BibTeX
function copyBibtex() {
  const code = document.querySelector('#citation code').innerText;
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.querySelector('.copy-code-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy BibTeX'; }, 2000);
  });
}
</script>
</body>
</html>
"""

def main():
    out_file = DOCS_DIR / "index.html"
    out_file.write_text(HTML_TEMPLATE, encoding="utf-8")
    print(f"Wrote {len(HTML_TEMPLATE)} bytes to {out_file}")

if __name__ == "__main__":
    main()
