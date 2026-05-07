"""
Non-executable reference file.

This file intentionally stores patch snippets as plain text so it stays valid
Python (and doesn't produce parsing/lint errors).
"""

INSTRUCTIONS = r"""
NSE SENTINEL - ANIMATIONS PATCH (manual app.py edit)

STEP 1 - CSS (paste before the closing </style> tag)

In app.py, find the big CSS block that contains:
    .breakdown-box {

Scroll down to the line:
    </style>

Paste the CSS block below just BEFORE that </style> line.

----- PASTE CSS HERE -----
/* ANIMATION KEYFRAMES */
@keyframes fadeSlideUp {
  from { opacity:0; transform:translateY(22px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes glowPulse {
  0%,100% { box-shadow:0 0 6px currentColor,0 0 4px currentColor; }
  50%     { box-shadow:0 0 22px currentColor,0 0 14px currentColor; }
}
@keyframes borderBreath {
  0%,100% { border-color:#00d4a8; box-shadow:0 0 8px rgba(0,212,168,.25); }
  50%     { border-color:#0094ff; box-shadow:0 0 22px rgba(0,148,255,.35); }
}
@keyframes scanSweep {
  0%   { left:-6%;  opacity:.85; }
  100% { left:106%; opacity:0;   }
}
@keyframes shimmer {
  0%   { background-position:-400px 0; }
  100% { background-position: 400px 0; }
}
@keyframes barReveal { from { width:0 !important; } }
@keyframes radarPing {
  0%  { transform:scale(1);   opacity:.85; }
  70% { transform:scale(3);   opacity:0;   }
  100%{ transform:scale(3);   opacity:0;   }
}
@keyframes floatUp {
  0%,100% { transform:translateY(0);    }
  50%     { transform:translateY(-5px); }
}
@keyframes logoGradient {
  0%  { background-position:  0% 50%; }
  50% { background-position:100% 50%; }
  100%{ background-position:  0% 50%; }
}
@keyframes rowFlash {
  0%   { background:rgba(0,212,168,.22); }
  100% { background:transparent; }
}
@keyframes tickerScroll {
  from { transform:translateX(0);    }
  to   { transform:translateX(-50%); }
}

/* APPLY TO EXISTING CLASSES */
.pick-card {
  animation: fadeSlideUp .45s cubic-bezier(.22,.68,0,1.2) both;
  transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease !important;
}
.pick-card:nth-child(1){ animation-delay:.00s }
.pick-card:nth-child(2){ animation-delay:.08s }
.pick-card:nth-child(3){ animation-delay:.16s }
.pick-card:nth-child(4){ animation-delay:.24s }
.pick-card:nth-child(5){ animation-delay:.32s }
.pick-card:nth-child(6){ animation-delay:.40s }
.pick-card:hover {
  transform: translateY(-6px) !important;
  border-color: #243550 !important;
  box-shadow: 0 14px 36px rgba(0,0,0,.5), 0 0 18px rgba(0,212,168,.13) !important;
}
.banner-logo {
  background: linear-gradient(270deg,#00d4a8,#0094ff,#f0b429,#00d4a8);
  background-size: 400% 400%;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: logoGradient 6s ease infinite;
}
.scan-header-wrap { position: relative; overflow: hidden; }
.scan-header-wrap::after {
  content: '';
  position: absolute;
  top: 0; bottom: 0;
  width: 5%;
  background: linear-gradient(90deg,transparent,rgba(0,212,168,.6),transparent);
  animation: scanSweep 2.4s linear infinite;
  pointer-events: none;
}
.sig-buy   { color: #00d4a8 !important; animation: glowPulse 2.2s ease-in-out infinite; }
.sig-sell,
.sig-avoid { color: #ff4d6d !important; animation: glowPulse 2.5s ease-in-out infinite; }
.sig-watch { color: #f0b429 !important; animation: glowPulse 3s ease-in-out infinite; }

.winner-card,
div[style*="border:2px solid #00d4a8"],
div[style*="border:2px solid #0094ff"],
div[style*="border:2px solid #ff4d6d"],
div[style*="border:2px solid #f0b429"] {
  animation: borderBreath 3.5s ease-in-out infinite;
}

[data-testid="stProgressBar"] > div > div { animation: barReveal 1s cubic-bezier(.4,0,.2,1) both; }
[data-testid="stMetric"] { animation: floatUp 5s ease-in-out infinite; }
[data-testid="stMetric"]:nth-child(2){ animation-delay: .9s }
[data-testid="stMetric"]:nth-child(3){ animation-delay:1.8s }
[data-testid="stMetric"]:nth-child(4){ animation-delay:2.7s }

.live-dot { position: relative; }
.live-dot::after {
  content: '';
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  border: 2px solid var(--accent);
  animation: radarPing 2.2s ease-out infinite;
  pointer-events: none;
}
.section-lbl { animation: fadeSlideUp .5s ease both; }
.grade-s { color:#00d4a8 !important; font-weight:800 !important; }
.grade-a { color:#0094ff !important; font-weight:800 !important; }
.grade-b { color:#f0b429 !important; font-weight:800 !important; }
.grade-c { color:#ff8c00 !important; font-weight:800 !important; }
.grade-d { color:#ff4d6d !important; font-weight:800 !important; }
.skeleton {
  border-radius: 6px;
  background: linear-gradient(90deg,#0f1823 25%,#1a2d42 50%,#0f1823 75%);
  background-size: 400px 100%;
  animation: shimmer 1.4s ease-in-out infinite;
}
.row-new { animation: rowFlash 1.4s ease-out both; }

.ticker-strip {
  overflow: hidden;
  white-space: nowrap;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  background: var(--bg2);
  padding: 5px 0;
  font-family: var(--mono);
  font-size: 11px;
}
.ticker-inner { display: inline-block; animation: tickerScroll 28s linear infinite; }
.ticker-item      { display:inline-block; padding:0 20px; color:var(--muted); }
.ticker-item.up   { color:#00d4a8; }
.ticker-item.down { color:#ff4d6d; }

/* Scrollbar */
::-webkit-scrollbar             { width:5px; height:5px; }
::-webkit-scrollbar-track       { background:var(--bg2); }
::-webkit-scrollbar-thumb       { background:var(--border2); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--accent); }
----- END CSS -----


STEP 2 - JavaScript (paste inside the existing components.html <script> block)

In app.py, find the existing:
    components.html(
        \"\"\"
        <script>
        ...
        </script>
        \"\"\",
        height=0,
        width=0,
    )

Paste the JS block below just BEFORE the closing </script> tag.

----- PASTE JS HERE -----
const BUY_KW  = ["buy","breakout","long","strong","fire","🔥","✅"];
const SELL_KW = ["sell","avoid","short","❌","trap","bearish","weak"];
const WATCH_KW= ["watch","wait","caution","neutral","👀","⚠️"];

function patchSignals() {
  document.querySelectorAll(
    ".mode-pill, span[style*='border-radius:6px'], span[style*='border-radius:20px'], .sig-badge"
  ).forEach(el => {
    if (el.dataset.sp) return;
    const t = el.innerText.toLowerCase();
    if (BUY_KW.some(w  => t.includes(w))) el.classList.add("sig-buy");
    else if (SELL_KW.some(w => t.includes(w))) el.classList.add("sig-sell");
    else if (WATCH_KW.some(w => t.includes(w))) el.classList.add("sig-watch");
    el.dataset.sp = "1";
  });
}

function patchGrades() {
  document.querySelectorAll("td").forEach(el => {
    if (el.dataset.gp) return;
    const t = el.innerText.trim().toUpperCase();
    if      (/^S\+?$/.test(t)) { el.classList.add("grade-s"); el.dataset.gp="1"; }
    else if (/^A[+\-]?$/.test(t)){ el.classList.add("grade-a"); el.dataset.gp="1"; }
    else if (/^B[+\-]?$/.test(t)){ el.classList.add("grade-b"); el.dataset.gp="1"; }
    else if (/^C[+\-]?$/.test(t)){ el.classList.add("grade-c"); el.dataset.gp="1"; }
    else if (/^D[+\-]?$/.test(t)){ el.classList.add("grade-d"); el.dataset.gp="1"; }
  });
}

function staggerCards() {
  document.querySelectorAll(".pick-card, .breakdown-box").forEach((el, i) => {
    if (el.dataset.sf) return;
    el.dataset.sf = "1";
    el.style.opacity = "0";
    el.style.transform = "translateY(18px)";
    el.style.transition =
      `opacity .4s ease ${i*0.08}s, transform .4s cubic-bezier(.22,.68,0,1.2) ${i*0.08}s`;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      el.style.opacity = "1";
      el.style.transform = "translateY(0)";
    }));
  });
}

function animCounters() {
  document.querySelectorAll("[data-testid='stMetricValue']").forEach(el => {
    if (el.dataset.ac) return;
    const raw = el.innerText.trim();
    const pfx = raw.startsWith("₹") ? "₹" : "";
    const sfx = raw.endsWith("%") ? "%" : "";
    const num = parseFloat(raw.replace(/[₹%,]/g, ""));
    if (isNaN(num)) return;
    el.dataset.ac = "1";
    const dur = 900, t0 = performance.now(), isInt = Number.isInteger(num);
    (function tick(now) {
      const p = Math.min((now - t0) / dur, 1);
      const e = 1 - Math.pow(1 - p, 3);
      const v = num * e;
      el.innerText = pfx + (isInt ? Math.round(v).toLocaleString("en-IN") : v.toFixed(2)) + sfx;
      if (p < 1) requestAnimationFrame(tick);
    })(performance.now());
  });
}

function staggerRows() {
  document.querySelectorAll(".stDataFrame tbody tr").forEach((row, i) => {
    if (row.dataset.rs) return;
    row.dataset.rs = "1";
    row.style.opacity = "0";
    row.style.transition = `opacity .3s ease ${Math.min(i * 0.04, 0.8)}s`;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      row.style.opacity = "1";
    }));
  });
}

let _prevRowCount = 0;
function flashNewRows() {
  const rows = document.querySelectorAll(".stDataFrame tbody tr");
  if (rows.length > _prevRowCount) {
    for (let i = _prevRowCount; i < rows.length; i++) {
      rows[i].classList.remove("row-new");
      void rows[i].offsetWidth;
      rows[i].classList.add("row-new");
    }
  }
  _prevRowCount = rows.length;
}

function wireScanHeaders() {
  document.querySelectorAll("h2, h3").forEach(h => {
    if (!h.innerText.includes("Scanner") && !h.innerText.includes("Scan")) return;
    const p = h.parentElement;
    if (!p.classList.contains("scan-header-wrap")) {
      p.classList.add("scan-header-wrap");
    }
  });
}

function runAll() {
  patchSignals();
  patchGrades();
  flashNewRows();
}

patchSignals();
patchGrades();
flashNewRows();
staggerCards();
animCounters();
staggerRows();
wireScanHeaders();

let _animTimer = null;
const _animObserver = new MutationObserver(() => {
  if (_animTimer) return;
  _animTimer = setTimeout(() => {
    _animTimer = null;
    runAll();
  }, 150);
});
_animObserver.observe(document.body, { childList: true, subtree: true });
----- END JS -----

DONE. Save app.py, then run:
    .\.venv\Scripts\python.exe -m streamlit run app.py
"""
