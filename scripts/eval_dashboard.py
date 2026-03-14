"""Interactive evaluation dashboard generator.

This side-project script builds a beautiful HTML dashboard
from `evaluation_report.json` emitted by `scripts/evaluate_model.py`.

Note: If `plotly` is installed in Python, the dashboard embeds Plotly JS
inline for offline interactivity. Otherwise it falls back to Plotly CDN.

Usage:
    python scripts/eval_dashboard.py --report logs/evaluation_artifacts/evaluation_report.json
    python scripts/eval_dashboard.py --report logs/evaluation_artifacts/evaluation_report.json --output logs/evaluation_artifacts/interactive_dashboard.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote


def _read_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_for_js(value: Any) -> str:
    s = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return s.replace("</", "<\\/").replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _plotly_script_tag() -> str:
    """Return inline Plotly bundle if available, else CDN script tag."""
    try:
        from plotly.offline.offline import get_plotlyjs  # type: ignore

        return f"<script>{get_plotlyjs()}</script>"
    except Exception:  # noqa: BLE001 - optional dependency fallback
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


def _build_dashboard_html(report: dict[str, Any]) -> str:
    artifacts = report.get("artifacts", {})
    images = artifacts.get("images", [])
    image_names = [Path(str(img)).name for img in images]

    threshold_points = report.get("threshold_sweep", {}).get("points", [])
    curves = report.get("curves", {}).get("roc_pr", {})
    cm = report.get("confusion_matrix", {})
    calibration = report.get("calibration", {})
    cal_curve = calibration.get("calibration_curve", {})

    base_threshold = float(report.get("threshold_used", 0.5))
    near_index = 0
    best_delta = float("inf")
    for i, p in enumerate(threshold_points):
        d = abs(float(p.get("threshold", 0.0)) - base_threshold)
        if d < best_delta:
            best_delta = d
            near_index = i

    def fmt(v: float | int | None, n: int = 4) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, (int,)):
            return str(v)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "N/A"
        return f"{float(v):.{n}f}"

    model_type_h = html.escape(str(report.get("model_type", "unknown")), quote=True)
    split_h = html.escape(str(report.get("split", "unknown")), quote=True)
    n_samples = int(report.get("n_samples", 0) or 0)

    gallery_blocks: list[str] = []
    for name in image_names:
        safe_name = html.escape(name, quote=True)
        safe_src = quote(name)
        gallery_blocks.append(
            f'<div class="gallery-item">'
            f'<img src="{safe_src}" alt="{safe_name}" data-src="{safe_src}" data-title="{safe_name}"/>'
            f'<div class="cap">{safe_name}</div>'
            f'<div class="img-actions">'
            f'<button type="button" data-open-src="{safe_src}" data-open-title="{safe_name}">Enlarge</button>'
            f'<a href="{safe_src}" target="_blank" rel="noopener noreferrer">Open full size</a>'
            f'</div>'
            f'</div>'
        )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Wake-Word Evaluation Dashboard</title>
  {_plotly_script_tag()}
  <style>
    :root {{
      --bg: #0b1220;
      --card: #121c2f;
      --card2: #10223b;
      --text: #e9f0ff;
      --muted: #98afd4;
      --line: #1e3154;
      --accent: #57b3ff;
      --ok: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
    }}
    body {{ margin: 0; background: radial-gradient(circle at top, #111b31, var(--bg)); color: var(--text); font-family: Inter, system-ui, Segoe UI, Roboto, Arial, sans-serif; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .hero {{ display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 34px; }}
    .sub {{ color: var(--muted); margin: 6px 0 0; }}
    .badge {{ padding: 8px 12px; background: var(--card2); border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .card {{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.08)); border: 1px solid var(--line); border-radius: 14px; padding: 14px; }}
    .k {{ color: var(--muted); font-size: 12px; letter-spacing: .06em; text-transform: uppercase; }}
    .v {{ margin-top: 4px; font-size: 30px; font-weight: 700; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }}
    @media (max-width: 980px) {{ .row {{ grid-template-columns: 1fr; }} }}
    .section-title {{ margin: 26px 0 10px; font-size: 21px; }}
    .plot {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 8px; min-height: 320px; }}
    .controls {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px; margin-top: 12px; }}
    input[type=range] {{ width: 100%; }}
    .cm {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }}
    .cm .cell {{ background: #0e1728; border: 1px solid var(--line); border-radius: 10px; padding: 10px; }}
    .tiny {{ color: var(--muted); font-size: 12px; }}
    .gallery {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:12px; }}
    .gallery-item {{ background: var(--card); border:1px solid var(--line); border-radius: 12px; overflow: hidden; }}
    .gallery-item img {{ width:100%; display:block; background:white; cursor: zoom-in; }}
    .gallery-item .cap {{ padding:8px 10px; color:var(--muted); font-size:12px; }}
    .img-actions {{ display:flex; gap:10px; padding:0 10px 10px; font-size:12px; }}
    .img-actions a, .img-actions button {{
      border: 1px solid var(--line);
      background: #0f1a2f;
      color: var(--text);
      border-radius: 8px;
      padding: 6px 8px;
      cursor: pointer;
      text-decoration: none;
    }}
    .img-actions button:hover, .img-actions a:hover {{ border-color: #3a5f95; }}
    .lightbox {{
      position: fixed;
      inset: 0;
      background: rgba(3, 8, 18, 0.88);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 999;
      padding: 24px;
    }}
    .lightbox.open {{ display: flex; }}
    .lightbox-inner {{ max-width: 96vw; max-height: 96vh; display: flex; flex-direction: column; gap: 8px; }}
    .lightbox img {{
      max-width: min(96vw, 2200px);
      max-height: calc(96vh - 60px);
      width: auto;
      height: auto;
      object-fit: contain;
      background: #fff;
      border-radius: 8px;
      border: 1px solid #27416b;
    }}
    .lightbox-bar {{ display:flex; justify-content: space-between; align-items:center; gap: 10px; color: #d5e4ff; }}
    .lightbox-close {{
      border: 1px solid #3b5f97;
      background: #10223b;
      color: #eaf2ff;
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    .lightbox-nav {{
      border: 1px solid #3b5f97;
      background: #10223b;
      color: #eaf2ff;
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <div>
        <h1>Wake-Word Evaluation Dashboard</h1>
        <p class=\"sub\">Interactive explorer generated from <code>evaluation_report.json</code></p>
      </div>
      <div class=\"badge\">Model: {model_type_h} · Split: {split_h} · Samples: {n_samples}</div>
    </div>

    <div class=\"grid\">
      <div class=\"card\"><div class=\"k\">Recall</div><div class=\"v\">{fmt(report.get("recall"))}</div></div>
      <div class=\"card\"><div class=\"k\">Precision</div><div class=\"v\">{fmt(report.get("precision"))}</div></div>
      <div class=\"card\"><div class=\"k\">F1</div><div class=\"v\">{fmt(report.get("f1_score"))}</div></div>
      <div class=\"card\"><div class=\"k\">FAH</div><div class=\"v\">{fmt(report.get("ambient_false_positives_per_hour"))}</div></div>
      <div class=\"card\"><div class=\"k\">AUC-ROC</div><div class=\"v\">{fmt(report.get("auc_roc"))}</div></div>
      <div class=\"card\"><div class=\"k\">AUC-PR</div><div class=\"v\">{fmt(report.get("auc_pr"))}</div></div>
      <div class=\"card\"><div class=\"k\">ECE</div><div class=\"v\">{fmt(calibration.get("ece"))}</div></div>
      <div class=\"card\"><div class=\"k\">Threshold</div><div class=\"v\">{fmt(report.get("threshold_used"))}</div></div>
    </div>

    <div class=\"section-title\">Interactive Threshold Explorer</div>
    <div class=\"controls\">
      <label for=\"thr\">Threshold: <strong id=\"thrValue\"></strong></label>
      <input id=\"thr\" type=\"range\" min=\"0\" max=\"100\" value=\"{near_index}\" step=\"1\" />
      <div class=\"cm\" id=\"cmBox\"></div>
      <div class=\"tiny\">Move the slider to inspect the precision/recall/FAH trade-off across thresholds.</div>
    </div>

    <div class=\"row\">
      <div class=\"plot\" id=\"rocPlot\"></div>
      <div class=\"plot\" id=\"prPlot\"></div>
    </div>
    <div class=\"row\">
      <div class=\"plot\" id=\"thrPlot\"></div>
      <div class=\"plot\" id=\"calPlot\"></div>
    </div>

    <div class=\"section-title\">Generated Artifacts</div>
    <div class=\"gallery\">
      {''.join(gallery_blocks)}
    </div>

    <div id=\"lightbox\" class=\"lightbox\" role=\"dialog\" aria-modal=\"true\" aria-label=\"Image preview\">
      <div class=\"lightbox-inner\">
        <div class=\"lightbox-bar\">
          <strong id=\"lightboxTitle\">Preview</strong>
          <button id=\"lightboxClose\" class=\"lightbox-close\" type=\"button\">Close</button>
        </div>
        <img id=\"lightboxImg\" src=\"\" alt=\"Expanded artifact\" />
      </div>
    </div>

    <p class=\"tiny\" style=\"margin-top:14px\">Tip: Keep this dashboard in the same folder as <code>evaluation_report.json</code> and the generated PNG artifacts for embedded previews.</p>
  </div>

  <script>
    const report = {_json_for_js(report)};
    const points = {_json_for_js(threshold_points)};
    const curves = {_json_for_js(curves)};
    const cm = {_json_for_js(cm)};
    const calCurve = {_json_for_js(cal_curve)};

    if (!window.Plotly) {{
      document.body.insertAdjacentHTML('afterbegin',
        '<div style="position:sticky;top:0;z-index:99;background:#7f1d1d;color:#fff;padding:10px 14px;font-family:system-ui">' +
        'Plotly CDN failed to load. Interactive charts are unavailable offline. ' +
        'Use executive_report.html for offline viewing.' +
        '</div>'
      );
      throw new Error('Plotly is unavailable');
    }}

    function linePlot(div, traces, title, xTitle, yTitle) {{
      Plotly.newPlot(div, traces, {{
        title,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {{color:'#e9f0ff'}},
        xaxis: {{title: xTitle, gridcolor:'#23385f'}},
        yaxis: {{title: yTitle, gridcolor:'#23385f'}},
        margin: {{l:50,r:20,t:50,b:50}}
      }}, {{displayModeBar: false, responsive: true}});
    }}

    linePlot('rocPlot', [{{x: curves.fpr || [], y: curves.tpr || [], mode:'lines', name:'ROC', line:{{color:'#57b3ff'}}}}], 'ROC Curve', 'FPR', 'TPR');
    linePlot('prPlot', [{{x: curves.recall || [], y: curves.precision || [], mode:'lines', name:'PR', line:{{color:'#22c55e'}}}}], 'PR Curve', 'Recall', 'Precision');

    const th = points.map(p => p.threshold);
    const rec = points.map(p => p.recall);
    const pre = points.map(p => p.precision);
    const fah = points.map(p => p.ambient_false_positives_per_hour);
    linePlot('thrPlot', [
      {{x: th, y: rec, mode:'lines', name:'Recall', line:{{color:'#57b3ff'}}}},
      {{x: th, y: pre, mode:'lines', name:'Precision', line:{{color:'#22c55e'}}}},
      {{x: th, y: fah, mode:'lines', name:'FAH', line:{{color:'#f59e0b'}}, yaxis:'y2'}}
    ], 'Threshold Sweep', 'Threshold', 'Recall / Precision');
    Plotly.relayout('thrPlot', {{
      yaxis2: {{overlaying:'y', side:'right', title:'FAH', gridcolor:'#23385f'}}
    }});

    linePlot('calPlot', [
      {{x: calCurve.prob_pred || [], y: calCurve.prob_true || [], mode:'lines+markers', name:'Model', line:{{color:'#57b3ff'}}}},
      {{x:[0,1], y:[0,1], mode:'lines', name:'Perfect', line:{{color:'#9aa', dash:'dash'}}}}
    ], 'Reliability Diagram', 'Predicted Probability', 'Observed Frequency');

    const slider = document.getElementById('thr');
    const thrValue = document.getElementById('thrValue');
    const cmBox = document.getElementById('cmBox');
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightboxImg');
    const lightboxTitle = document.getElementById('lightboxTitle');
    const lightboxClose = document.getElementById('lightboxClose');

    function renderCell(label, value) {{
      return `<div class=\"cell\"><div class=\"tiny\">${{label}}</div><div style=\"font-size:24px;font-weight:700\">${{value}}</div></div>`;
    }}

    function renderAt(idx) {{
      if (!points.length) return;
      const p = points[idx];
      thrValue.textContent = `${{p.threshold.toFixed(4)}} · Recall ${{p.recall.toFixed(4)}} · Precision ${{p.precision.toFixed(4)}} · FAH ${{p.ambient_false_positives_per_hour.toFixed(4)}}`;

      const totalPos = (cm.tp || 0) + (cm.fn || 0);
      const totalNeg = (cm.tn || 0) + (cm.fp || 0);
      const tp = Math.round(p.recall * totalPos);
      const fn = totalPos - tp;
      const fp = Math.round(p.fpr * totalNeg);
      const tn = totalNeg - fp;

      cmBox.innerHTML = [
        renderCell('TN', tn),
        renderCell('FP', fp),
        renderCell('FN', fn),
        renderCell('TP', tp),
      ].join('');
    }}

    slider.max = Math.max(0, points.length - 1);
    slider.value = Math.min({near_index}, Math.max(0, points.length - 1));
    slider.addEventListener('input', () => renderAt(Number(slider.value)));
    renderAt(Number(slider.value));

    function openLightbox(src, title) {{
      lightboxImg.src = src;
      lightboxImg.alt = title || 'Expanded artifact';
      lightboxTitle.textContent = title || 'Preview';
      lightbox.classList.add('open');
      document.body.style.overflow = 'hidden';
    }}

    function closeLightbox() {{
      lightbox.classList.remove('open');
      lightboxImg.src = '';
      document.body.style.overflow = '';
    }}

    lightboxClose.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', (ev) => {{
      if (ev.target === lightbox) closeLightbox();
    }});
    document.addEventListener('keydown', (ev) => {{
      if (ev.key === 'Escape' && lightbox.classList.contains('open')) closeLightbox();
    }});

    document.querySelectorAll('.gallery-item img').forEach((img) => {{
      img.addEventListener('click', () => openLightbox(img.dataset.src || img.getAttribute('src'), img.dataset.title || img.getAttribute('alt')));
    }});
    document.querySelectorAll('[data-open-src]').forEach((btn) => {{
      btn.addEventListener('click', () => openLightbox(btn.dataset.openSrc, btn.dataset.openTitle));
    }});
  </script>
</body>
</html>
"""


def build_dashboard(report_path: Path, output_path: Path) -> None:
    report = _read_report(report_path)
    html = _build_dashboard_html(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build interactive dashboard from evaluation_report.json")
    parser.add_argument("--report", type=str, required=True, help="Path to evaluation_report.json")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: <report_dir>/interactive_dashboard.html)",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    output_path = Path(args.output) if args.output else report_path.parent / "interactive_dashboard.html"

    try:
        build_dashboard(report_path=report_path, output_path=output_path)
    except Exception as exc:  # noqa: BLE001 - CLI error surfacing
        print(f"Error: {exc}")
        return 1

    print(f"✓ Dashboard generated: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
