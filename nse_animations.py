"""
nse_animations.py — NSE Sentinel MAX Animation Layer
═══════════════════════════════════════════════════════
Drop this file into the same folder as app.py, then add
TWO lines to app.py right after the existing st.markdown CSS block:

    from nse_animations import inject_animations
    inject_animations()

That's it. All animations are additive — nothing existing breaks.

Animations included
───────────────────
CSS side
  • fadeSlideUp        — result cards / pick cards enter from below
  • glowPulse          — BUY / signal badges throb with color glow
  • borderGlow         — winner-card border breathes
  • scanSweep          — horizontal radar line across scanner header
  • shimmer            — skeleton-loading sheen on placeholders
  • barFill            — progress bars fill left→right on paint
  • radarPing          — concentric ring pulse for Live Breakout Pulse
  • floatCard          — subtle levitation for metric / pick cards
  • gradientShift      — animated gradient for the top banner logo
  • rowFlash           — brief green flash on newly inserted table rows
  • typeReveal         — left→right clip reveal for section headings
  • ticker-scroll      — horizontal marquee for a live-ticker strip

JS side (Intersection Observer + direct DOM tweaks)
  • Stagger fade-in    — each result card enters 80 ms after the previous
  • Counter animation  — metric numbers count up from 0 on entry
  • Signal badge glow  — adds .sig-buy / .sig-sell class after render
  • Table row stagger  — tbody rows slide in sequentially
  • Hover card lift    — JS adds precise translateY(-4px) on hover
  • Scan line          — animates a moving line across the scan progress bar
  • Auto-class patcher — reads text of badges to apply correct glow colour
"""

import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# CSS BLOCK
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ══════════════════════════════════════════════
   KEYFRAMES
══════════════════════════════════════════════ */

/* 1 · cards slide up from 24 px below */
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(24px); }
  to   { opacity: 1; transform: translateY(0);    }
}

/* 2 · signal badge glow pulse */
@keyframes glowPulse {
  0%,100% { box-shadow: 0 0 6px  currentColor, 0 0  4px currentColor; }
  50%     { box-shadow: 0 0 18px currentColor, 0 0 10px currentColor; }
}

/* 3 · winner / top-pick card border breathes */
@keyframes borderGlow {
  0%,100% { border-color: var(--accent);  box-shadow: 0 0  6px rgba(0,212,168,.25); }
  50%     { border-color: var(--accent2); box-shadow: 0 0 20px rgba(0,148,255,.35); }
}

/* 4 · horizontal radar sweep line */
@keyframes scanSweep {
  0%   { left: -4%; opacity: .9; }
  100% { left: 104%; opacity: 0; }
}

/* 5 · skeleton shimmer */
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position:  400px 0; }
}

/* 6 · progress bar fill */
@keyframes barFill {
  from { width: 0 !important; }
  /* "to" is handled via inline width — the bar just reveals itself */
}

/* 7 · radar ping (Live Breakout Pulse) */
@keyframes radarPing {
  0%   { transform: scale(1);   opacity: .9; }
  70%  { transform: scale(2.8); opacity: 0;  }
  100% { transform: scale(2.8); opacity: 0;  }
}

/* 8 · subtle card float */
@keyframes floatCard {
  0%,100% { transform: translateY(0);    }
  50%     { transform: translateY(-4px); }
}

/* 9 · animated gradient for banner logo */
@keyframes gradientShift {
  0%   { background-position:   0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position:   0% 50%; }
}

/* 10 · row flash on new data */
@keyframes rowFlash {
  0%   { background: rgba(0,212,168,.22); }
  100% { background: transparent;         }
}

/* 11 · text reveal left→right */
@keyframes typeReveal {
  from { clip-path: inset(0 100% 0 0); }
  to   { clip-path: inset(0 0% 0 0);   }
}

/* 12 · horizontal ticker scroll */
@keyframes tickerScroll {
  from { transform: translateX(0);      }
  to   { transform: translateX(-50%);   }
}

/* ══════════════════════════════════════════════
   APPLY ANIMATIONS TO EXISTING CLASSES
══════════════════════════════════════════════ */

/* Pick cards — fade + slide on paint */
.pick-card {
  animation: fadeSlideUp 0.45s cubic-bezier(.22,.68,0,1.2) both;
  will-change: transform, opacity;
}

/* Stagger pick cards 1–6 */
.pick-card:nth-child(1) { animation-delay: 0.00s; }
.pick-card:nth-child(2) { animation-delay: 0.07s; }
.pick-card:nth-child(3) { animation-delay: 0.14s; }
.pick-card:nth-child(4) { animation-delay: 0.21s; }
.pick-card:nth-child(5) { animation-delay: 0.28s; }
.pick-card:nth-child(6) { animation-delay: 0.35s; }

/* Winner card border breathes */
.winner-card,
div[style*="border:2px solid #00d4a8"],
div[style*="border:2px solid var(--accent)"] {
  animation: borderGlow 3s ease-in-out infinite;
}

/* Signal badges glow — applied by JS via class */
.sig-buy {
  color: #00d4a8 !important;
  animation: glowPulse 2s ease-in-out infinite;
}
.sig-sell, .sig-avoid {
  color: #ff4d6d !important;
  animation: glowPulse 2.5s ease-in-out infinite;
}
.sig-watch {
  color: #f0b429 !important;
  animation: glowPulse 3s ease-in-out infinite;
}

/* Progress / Streamlit progress bar fill reveal */
[data-testid="stProgressBar"] > div > div {
  animation: barFill 1s cubic-bezier(.4,0,.2,1) both;
}

/* Metric cards float subtly */
[data-testid="stMetric"] {
  animation: floatCard 5s ease-in-out infinite;
}
[data-testid="stMetric"]:nth-child(2) { animation-delay: 0.8s; }
[data-testid="stMetric"]:nth-child(3) { animation-delay: 1.6s; }
[data-testid="stMetric"]:nth-child(4) { animation-delay: 2.4s; }

/* Section labels reveal left→right */
.section-lbl {
  animation: typeReveal 0.6s cubic-bezier(.22,.68,0,1.2) both;
  display: inline-block;
}

/* Table rows stagger (handled by JS — class added there) */
.row-anim {
  animation: fadeSlideUp 0.35s ease both;
}

/* Live dot enhanced ping ring */
.live-dot::after {
  content: '';
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  border: 2px solid var(--accent);
  animation: radarPing 2s ease-out infinite;
  pointer-events: none;
}
.live-dot { position: relative; }

/* Banner logo animated gradient */
.banner-logo {
  background: linear-gradient(
    270deg,
    #00d4a8, #0094ff, #f0b429, #00d4a8
  );
  background-size: 400% 400%;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: gradientShift 6s ease infinite;
}

/* ── Scanner sweep line ──────────────────────────────── */
.scan-header-wrap {
  position: relative;
  overflow: hidden;
}
.scan-header-wrap::after {
  content: '';
  position: absolute;
  top: 0; bottom: 0;
  width: 4%;
  background: linear-gradient(
    90deg, transparent, rgba(0,212,168,.55), transparent
  );
  animation: scanSweep 2.4s linear infinite;
  pointer-events: none;
}

/* ── Skeleton shimmer placeholder ────────────────────── */
.skeleton {
  border-radius: 6px;
  background: linear-gradient(
    90deg,
    #0f1823 25%,
    #162030 50%,
    #0f1823 75%
  );
  background-size: 400px 100%;
  animation: shimmer 1.4s ease-in-out infinite;
  height: 18px;
  margin: 6px 0;
}

/* ── Live ticker strip ───────────────────────────────── */
.ticker-strip {
  overflow: hidden;
  white-space: nowrap;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  background: var(--bg2);
  padding: 6px 0;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
}
.ticker-strip-inner {
  display: inline-block;
  animation: tickerScroll 30s linear infinite;
}
.ticker-item {
  display: inline-block;
  padding: 0 28px;
}
.ticker-item.up   { color: #00d4a8; }
.ticker-item.down { color: #ff4d6d; }
.ticker-item.flat { color: #4a6480; }

/* ── Hover lift for pick-cards (belt + suspenders over JS) ── */
.pick-card:hover {
  transform: translateY(-5px) !important;
  border-color: var(--border2) !important;
  box-shadow: 0 12px 32px rgba(0,0,0,.45),
              0  0  16px rgba(0,212,168,.12) !important;
  transition: transform 0.22s ease, box-shadow 0.22s ease,
              border-color 0.22s ease !important;
}

/* ── Grade badge colours ─────────────────────────────────── */
.grade-a  { color: #00d4a8; font-weight: 800; }
.grade-b  { color: #0094ff; font-weight: 800; }
.grade-c  { color: #f0b429; font-weight: 800; }
.grade-d  { color: #ff4d6d; font-weight: 800; }

/* ── Scan result flash when new rows appear ──────────────── */
.row-new {
  animation: rowFlash 1.2s ease-out both;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar             { width: 6px; height: 6px; }
::-webkit-scrollbar-track       { background: var(--bg2); }
::-webkit-scrollbar-thumb       { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# JAVASCRIPT BLOCK
# ─────────────────────────────────────────────────────────────────────────────
_JS = """
<script>
(function () {
  "use strict";

  /* ── 1. Stagger fade-in for result / pick cards ── */
  function staggerCards() {
    const cards = document.querySelectorAll(
      ".pick-card, .breakdown-box, [data-testid='stMetric']"
    );
    cards.forEach((el, i) => {
      if (el.dataset.staggerDone) return;
      el.dataset.staggerDone = "1";
      el.style.opacity = "0";
      el.style.transform = "translateY(20px)";
      el.style.transition =
        "opacity 0.42s ease " + i * 0.07 + "s, " +
        "transform 0.42s cubic-bezier(.22,.68,0,1.2) " + i * 0.07 + "s";
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          el.style.opacity = "1";
          el.style.transform = "translateY(0)";
        });
      });
    });
  }

  /* ── 2. Counter animation for metric number elements ── */
  function animateCounters() {
    document.querySelectorAll(
      "[data-testid='stMetricValue'], .pick-rank"
    ).forEach((el) => {
      if (el.dataset.counted) return;
      const raw = el.innerText.trim();
      // Only count purely numeric values (integers or decimals, optional % / ₹)
      const prefix = raw.startsWith("₹") ? "₹" : "";
      const suffix = raw.endsWith("%") ? "%" : "";
      const numStr = raw.replace(/[₹%,]/g, "");
      const target = parseFloat(numStr);
      if (isNaN(target)) return;
      el.dataset.counted = "1";
      const start = performance.now();
      const duration = 900;
      const isInt = Number.isInteger(target);
      function tick(now) {
        const elapsed = Math.min(now - start, duration);
        const progress = elapsed / duration;
        // ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = target * eased;
        el.innerText =
          prefix +
          (isInt
            ? Math.round(current).toLocaleString("en-IN")
            : current.toFixed(2)) +
          suffix;
        if (elapsed < duration) requestAnimationFrame(tick);
        else
          el.innerText =
            prefix +
            (isInt
              ? target.toLocaleString("en-IN")
              : target.toFixed(2)) +
            suffix;
      }
      requestAnimationFrame(tick);
    });
  }

  /* ── 3. Signal badge colour patcher ── */
  const BUY_WORDS  = ["buy", "breakout", "long", "strong", "🔥", "✅"];
  const SELL_WORDS = ["sell", "avoid", "short", "❌", "trap", "bearish"];
  const WATCH_WORDS= ["watch", "wait", "caution", "neutral", "👀", "⚠️"];

  function patchSignalBadges() {
    document.querySelectorAll(
      ".mode-pill, .sig-badge, span[style*='border-radius:6px']," +
      "span[style*='border-radius:20px']"
    ).forEach((el) => {
      if (el.dataset.sigPatched) return;
      const txt = el.innerText.toLowerCase();
      if (BUY_WORDS.some((w)  => txt.includes(w))) {
        el.classList.add("sig-buy");
      } else if (SELL_WORDS.some((w) => txt.includes(w))) {
        el.classList.add("sig-sell");
      } else if (WATCH_WORDS.some((w)) => txt.includes(w)) {
        el.classList.add("sig-watch");
      }
      el.dataset.sigPatched = "1";
    });
  }

  /* ── 4. Table row stagger ── */
  function staggerTableRows() {
    document.querySelectorAll(
      ".stDataFrame tbody tr, [data-testid='data-grid-canvas'] tr"
    ).forEach((row, i) => {
      if (row.dataset.rowAnim) return;
      row.dataset.rowAnim = "1";
      row.style.opacity = "0";
      row.style.transition =
        "opacity 0.3s ease " + Math.min(i * 0.04, 0.8) + "s";
      requestAnimationFrame(() => {
        requestAnimationFrame(() => { row.style.opacity = "1"; });
      });
    });
  }

  /* ── 5. Card hover lift (JS reinforcement) ── */
  function wireCardHover() {
    document.querySelectorAll(".pick-card").forEach((card) => {
      if (card.dataset.hoverWired) return;
      card.dataset.hoverWired = "1";
      card.addEventListener("mouseenter", () => {
        card.style.transform = "translateY(-5px)";
        card.style.transition = "transform 0.22s ease, box-shadow 0.22s ease";
        card.style.boxShadow =
          "0 14px 36px rgba(0,0,0,.5), 0 0 18px rgba(0,212,168,.14)";
      });
      card.addEventListener("mouseleave", () => {
        card.style.transform = "translateY(0)";
        card.style.boxShadow = "";
      });
    });
  }

  /* ── 6. Grade badge colouring ── */
  function patchGrades() {
    document.querySelectorAll("td, .grade-val").forEach((el) => {
      if (el.dataset.gradePatched) return;
      const txt = el.innerText.trim().toUpperCase();
      if (/^A[+-]?$/.test(txt)) { el.classList.add("grade-a"); el.dataset.gradePatched="1"; }
      else if (/^B[+-]?$/.test(txt)) { el.classList.add("grade-b"); el.dataset.gradePatched="1"; }
      else if (/^C[+-]?$/.test(txt)) { el.classList.add("grade-c"); el.dataset.gradePatched="1"; }
      else if (/^D[+-]?$/.test(txt)) { el.classList.add("grade-d"); el.dataset.gradePatched="1"; }
    });
  }

  /* ── 7. Scan header sweep wrapper (add class if missing) ── */
  function wireScanHeaders() {
    document.querySelectorAll("h2, h3").forEach((h) => {
      const parent = h.closest(".scan-header-wrap");
      if (!parent && h.innerText.includes("Scanner")) {
        h.parentElement.classList.add("scan-header-wrap");
      }
    });
  }

  /* ── 8. Row flash on newly inserted table rows ── */
  let _lastRowCount = 0;
  function flashNewRows() {
    const rows = document.querySelectorAll(".stDataFrame tbody tr");
    if (rows.length > _lastRowCount) {
      for (let i = _lastRowCount; i < rows.length; i++) {
        rows[i].classList.remove("row-new");
        void rows[i].offsetWidth;
        rows[i].classList.add("row-new");
      }
    }
    _lastRowCount = rows.length;
  }

  /* ── 9. Intersection Observer — trigger animations on scroll into view ── */
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        staggerCards();
        animateCounters();
        patchSignalBadges();
        staggerTableRows();
        wireCardHover();
        patchGrades();
        wireScanHeaders();
        flashNewRows();
        io.unobserve(e.target);
      });
    },
    { threshold: 0.05 }
  );

  /* ── 10. MutationObserver — catch Streamlit rerenders ── */
  const mo = new MutationObserver(() => {
    staggerCards();
    animateCounters();
    patchSignalBadges();
    staggerTableRows();
    wireCardHover();
    patchGrades();
    wireScanHeaders();
    flashNewRows();
  });

  function boot() {
    // Initial run
    staggerCards();
    animateCounters();
    patchSignalBadges();
    staggerTableRows();
    wireCardHover();
    patchGrades();
    wireScanHeaders();

    // Observe the whole app shell
    const shell = document.querySelector(".stApp") || document.body;
    mo.observe(shell, { childList: true, subtree: true });

    // Also watch each main block
    document.querySelectorAll("section, .element-container, .block-container")
      .forEach((el) => io.observe(el));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
</script>
"""

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Skeleton placeholder (use while scan is running)
# ─────────────────────────────────────────────────────────────────────────────
def skeleton_rows(n: int = 5) -> str:
    """
    Return HTML for n skeleton loading rows.
    Usage:
        st.markdown(skeleton_rows(8), unsafe_allow_html=True)
    """
    rows = "".join(
        f'<div class="skeleton" style="width:{85 + (i % 4) * 4}%;height:14px;"></div>'
        for i in range(n)
    )
    return (
        '<div style="padding:12px 0;display:flex;flex-direction:column;gap:8px;">'
        + rows
        + "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Live ticker strip
# ─────────────────────────────────────────────────────────────────────────────
def render_ticker_strip(items: list[tuple[str, float, float]] | None = None) -> None:
    """
    Render a scrolling ticker strip.

    items = list of (symbol, price, pct_change)
    If None, renders a placeholder strip with dummy symbols.
    """
    if items is None:
        items = [
            ("NIFTY50", 22500, 0.42),
            ("SENSEX",  74200, 0.38),
            ("BANKNIFTY", 48100, -0.15),
            ("RELIANCE", 2890, 1.2),
            ("TCS", 3740, -0.3),
            ("INFY", 1650, 0.8),
            ("HDFCBANK", 1590, -0.5),
            ("ICICIBANK", 1100, 1.1),
        ]

    def _item_html(sym, price, chg):
        cls   = "up" if chg > 0 else ("down" if chg < 0 else "flat")
        arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "—")
        color = "#00d4a8" if chg > 0 else ("#ff4d6d" if chg < 0 else "#4a6480")
        return (
            f'<span class="ticker-item {cls}">'
            f'{sym} &nbsp;₹{price:,.0f}&nbsp;'
            f'<span style="color:{color}">{arrow}{abs(chg):.2f}%</span>'
            f"</span>"
        )

    inner = "".join(_item_html(*i) for i in items) * 2  # duplicate for seamless loop
    st.markdown(
        f'<div class="ticker-strip">'
        f'  <div class="ticker-strip-inner">{inner}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Animated score badge
# ─────────────────────────────────────────────────────────────────────────────
def score_badge(value: float, max_val: float = 100) -> str:
    """
    Return HTML for an animated score badge with colour-coded glow.
    """
    if value >= 70:
        color = "#00d4a8"
    elif value >= 50:
        color = "#0094ff"
    elif value >= 35:
        color = "#f0b429"
    else:
        color = "#ff4d6d"

    pct = min(max(value / max_val * 100, 0), 100)
    return (
        f'<span style="'
        f"background:{color}18;border:1px solid {color};border-radius:8px;"
        f"padding:3px 10px;font-size:12px;font-weight:800;color:{color};"
        f'animation:glowPulse 2.5s ease-in-out infinite;display:inline-block;'
        f'">{value:.1f}</span>'
        f'<div style="height:4px;background:#1a2535;border-radius:2px;margin-top:4px;">'
        f'  <div style="width:{pct:.0f}%;height:4px;background:{color};'
        f'border-radius:2px;animation:barFill 1s ease both;"></div>'
        f"</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Animated signal chip
# ─────────────────────────────────────────────────────────────────────────────
def signal_chip(signal: str) -> str:
    """
    Return a glowing signal chip HTML string.
    """
    sl = signal.lower()
    if any(w in sl for w in ("buy", "breakout", "long", "strong", "🔥")):
        color, cls = "#00d4a8", "sig-buy"
    elif any(w in sl for w in ("sell", "avoid", "short", "❌")):
        color, cls = "#ff4d6d", "sig-sell"
    elif any(w in sl for w in ("watch", "wait", "caution", "👀")):
        color, cls = "#f0b429", "sig-watch"
    else:
        color, cls = "#4a6480", ""

    return (
        f'<span class="sig-badge {cls}" style="'
        f"background:{color}18;border:1px solid {color};border-radius:6px;"
        f"padding:2px 10px;font-size:11px;font-weight:700;color:{color};"
        f'display:inline-block;">{signal}</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INJECTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def inject_animations() -> None:
    """
    Call this ONCE in app.py right after the existing st.markdown CSS block.

        from nse_animations import inject_animations
        inject_animations()
    """
    # CSS injected via st.markdown (rendered in every rerun)
    st.markdown(_CSS, unsafe_allow_html=True)

    # JS injected via components.html (height=0 so it's invisible)
    components.html(_JS, height=0, width=0)
