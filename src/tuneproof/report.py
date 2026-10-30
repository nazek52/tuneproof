"""A portable HTML report: all numbers come from the experiment JSON."""

from __future__ import annotations

import html
import math


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _curve(points: list[dict]) -> str:
    phases = [("pretrain", "#a5b4fc"), ("lora", "#43e2b4"), ("zero_lr", "#f4b46a")]
    all_losses = [p["loss"] for p in points]
    if not all_losses or not all(math.isfinite(v) for v in all_losses):
        raise ValueError("curve requires finite measurements")
    ceiling = max(max(all_losses) * 1.05, 0.1)
    elements = []
    for i in range(5):
        y = 24 + i * 48
        value = ceiling * (1 - i / 4)
        elements.append(
            f'<line x1="55" y1="{y}" x2="880" y2="{y}" stroke="#273340"/>'
            f'<text x="44" y="{y + 4}" text-anchor="end">{value:.1f}</text>'
        )
    for phase, color in phases:
        subset = [p for p in points if p["phase"] == phase]
        if not subset:
            continue
        max_step = max(p["step"] for p in subset)
        coords = " ".join(
            f"{55 + p['step'] / max_step * 825:.2f},{216 - p['loss'] / ceiling * 192:.2f}"
            for p in subset
        )
        elements.append(
            f'<polyline data-phase="{phase}" points="{coords}" fill="none" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        for p in subset:
            x, y = 55 + p["step"] / max_step * 825, 216 - p["loss"] / ceiling * 192
            elements.append(
                f'<circle data-phase="{phase}" cx="{x:.2f}" cy="{y:.2f}" '
                f'r="3" fill="{color}"><title>{phase}: step {p["step"]}, '
                f"loss {p['loss']:.4f}</title></circle>"
            )
    return (
        '<svg viewBox="0 0 910 250" role="img" aria-label="Measured training loss by phase; '
        'each phase has its own normalized step axis">'
        + "".join(elements)
        + '<text x="455" y="245" text-anchor="middle">Progress within each phase →</text></svg>'
    )


def render_report(results: dict, curve: list[dict]) -> str:
    r = results
    table = "".join(
        f'<tr><td><span class="dot {key}"></span>{label}</td>'
        f"<td>{r[key]['response_nll']:.4f}</td>"
        f"<td>{100 * r[key]['exact_match']:.1f}%</td>"
        f"<td>{100 * r[key]['routing_label_match']:.1f}%</td>"
        f"<td>{r[key]['correct']} / {r[key]['examples']}</td></tr>"
        for key, label in [
            ("base", "Base checkpoint"),
            ("lora", "LoRA adapted"),
            ("zero_lr", "Zero-LR control"),
        ]
    )
    checks = "".join(
        f"<li><span>{'PASS' if passed else 'FAIL'}</span>{_escape(name.replace('_', ' '))}</li>"
        for name, passed in r["checks"].items()
    )
    examples = "".join(
        f"<tr><td>{_escape(s['prompt'])}</td><td>{_escape(s['expected'])}</td>"
        f"<td>{_escape(b['generated'])}</td><td>{_escape(s['generated'])}</td></tr>"
        for s, b in zip(r["lora"]["samples"], r["base"]["samples"], strict=True)
    )
    limitations = "".join(f"<li>{_escape(item)}</li>" for item in r["limitations"])
    gain = r["base"]["response_nll"] - r["lora"]["response_nll"]
    status = "Lower held-out response loss" if gain > 0 else "No held-out loss improvement"
    style = """
    :root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;
      background:#0b1117;color:#e4eaf0}*{box-sizing:border-box}body{margin:0}
    main{max-width:1180px;margin:auto;padding:36px 32px 64px}header{display:flex;
      align-items:center;justify-content:space-between;border-bottom:1px solid #273340;
      padding-bottom:22px}.brand{font-size:24px;font-weight:750;letter-spacing:-1px}
    .brand b{color:#43e2b4}.tag,.pill{font:11px ui-monospace,monospace;letter-spacing:1px;
      border:1px solid #33404e;border-radius:5px;padding:8px 11px;color:#9cabb9}
    .eyebrow{font:11px ui-monospace,monospace;color:#43e2b4;letter-spacing:2px;margin-top:44px}
    h1{font-size:clamp(32px,5vw,54px);letter-spacing:-2px;margin:12px 0 14px;line-height:1.08}
    .lead{max-width:720px;color:#9cabb9;line-height:1.7;font-size:15px}
    .pills{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px}.pill{letter-spacing:0}
    .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:30px 0}
    .card,section{background:#111a23;border:1px solid #273340;border-radius:10px}
    .card{padding:20px}.label{color:#9cabb9;font-size:12px}.value{font-size:30px;
      margin-top:12px;font-weight:650;letter-spacing:-1px}.note{font-size:11px;color:#8293a4;
      margin-top:8px;line-height:1.5}section{padding:24px;margin-top:18px}h2{font-size:16px;
      margin:0 0 8px}p{line-height:1.6}svg{width:100%;margin-top:12px;overflow:visible}
    svg text{fill:#8293a4;font:11px ui-monospace,monospace}.legend{display:flex;gap:18px;
      flex-wrap:wrap;margin:18px 0 0}button{background:transparent;color:#b6c5d3;
      border:1px solid #33404e;padding:8px 12px;border-radius:5px;cursor:pointer}
    button[aria-pressed=false]{opacity:.4}button:focus-visible,input:focus-visible{outline:2px solid #43e2b4}
    .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:9px}
    .base,.pretrain{background:#a5b4fc}.lora{background:#43e2b4}.zero_lr{background:#f4b46a}
    .cols{display:grid;grid-template-columns:1.15fr 1fr;gap:18px}.cols section{min-width:0}
    table{width:100%;border-collapse:collapse;font:12px ui-monospace,monospace}
    th{text-align:left;color:#8293a4;font-weight:400;font-size:10px;text-transform:uppercase;
      letter-spacing:1px}td,th{padding:16px 10px;border-bottom:1px solid #273340}
    .scroll{overflow-x:auto}.checks{list-style:none;padding:0;font-size:12px;line-height:2.6;
      color:#b6c5d3}.checks span{font:10px ui-monospace,monospace;color:#43e2b4;margin-right:12px}
    input{width:100%;background:#0b1117;border:1px solid #33404e;border-radius:5px;
      padding:11px;color:#e4eaf0;margin:14px 0 4px}footer{color:#8293a4;font-size:11px;
      padding-top:24px;line-height:1.8}.limits{color:#9cabb9;font-size:12px;line-height:1.9;padding-left:18px}
    a{color:#43e2b4}.good{color:#43e2b4}code{color:#b6c5d3}
    @media(max-width:760px){main{padding:24px 16px}.metrics{grid-template-columns:repeat(2,1fr)}
      .cols{grid-template-columns:1fr}section{padding:18px}.tag{display:none}h1{letter-spacing:-1px}}
    """
    script = """
    document.querySelectorAll('[data-toggle]').forEach(button=>button.addEventListener('click',()=>{
      const show=button.getAttribute('aria-pressed')==='false';
      button.setAttribute('aria-pressed',String(show));
      document.querySelectorAll('[data-phase="'+button.dataset.toggle+'"]').forEach(el=>{
        el.style.display=show?'':'none';
      });
    }));
    document.querySelector('#filter').addEventListener('input',event=>{
      const term=event.target.value.toLowerCase();
      document.querySelectorAll('#samples tbody tr').forEach(row=>{
        row.hidden=!row.textContent.toLowerCase().includes(term);
      });
    });
    """
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TuneProof · Experiment evidence</title>
<style>{style}</style></head><body><main>
<header><div class="brand">tune<span>proof</span><b>↗</b></div><span class="tag">EXPERIMENT / {_escape(r["config"]["seed"])} · MEASURED ON CPU</span></header>
<div class="eyebrow">EVIDENCE BEFORE CLAIMS</div><h1>Did fine-tuning<br>actually work?</h1>
<p class="lead">A tiny causal transformer. A frozen base. A trainable LoRA adapter.
A matched zero-learning-rate control. Every result below comes from a real local run.</p>
<div class="pills"><span class="pill">{_escape(r["environment"]["torch"])} · PyTorch</span>
<span class="pill">{r["data"]["train_rows"]} train / {r["data"]["validation_rows"]} held out</span>
<span class="pill">Synthetic ticket routing</span><span class="pill">No external API</span></div>
<div class="metrics">
<div class="card"><div class="label">Compact-format exact match · LoRA</div><div class="value good">{100 * r["lora"]["exact_match"]:.1f}%</div><div class="note">{r["lora"]["correct"]} of {r["lora"]["examples"]} labels + exact formatting</div></div>
<div class="card"><div class="label">Held-out response NLL · LoRA</div><div class="value">{r["lora"]["response_nll"]:.4f}</div><div class="note">{_escape(status)} · byte-token objective</div></div>
<div class="card"><div class="label">Trainable adapter parameters</div><div class="value">{r["parameters"]["adapter_percent"]:.1f}%</div><div class="note">{r["parameters"]["adapter"]:,} / {r["parameters"]["total"]:,} total parameters</div></div>
<div class="card"><div class="label">Control logit change</div><div class="value">{r["max_control_logit_delta"]:.1e}</div><div class="note">Zero LR must preserve predictions</div></div></div>
<section><h2>Training, with a negative control</h2><div class="note">Response-only cross-entropy on sampled training batches. Phases have independent step axes and different answer formats; these lines are not validation curves.</div>
{_curve(curve)}<div class="legend">
<button data-toggle="pretrain" aria-pressed="true"><i class="dot pretrain"></i>Base pretraining</button>
<button data-toggle="lora" aria-pressed="true"><i class="dot lora"></i>LoRA adaptation</button>
<button data-toggle="zero_lr" aria-pressed="true"><i class="dot zero_lr"></i>Zero-LR control</button></div></section>
<div class="cols"><section><h2>Same holdout. Three checkpoints.</h2><div class="scroll"><table><thead><tr><th>Checkpoint</th><th>NLL ↓</th><th>Format match ↑</th><th>Routing label ↑</th><th>Correct</th></tr></thead><tbody>{table}</tbody></table></div><div class="note">Format match requires the exact compact label. Routing-label match separately checks for the correct unique category in the generated text; verbose correct answers can pass it.</div></section>
<section><h2>Integrity checks</h2><ul class="checks">{checks}</ul></section></div>
<section><h2>Inspect every held-out prediction</h2><label class="note" for="filter">Filter by prompt, expected label or generated answer</label><input id="filter" type="search" placeholder="Try billing, crash, or ticket 039…">
<div class="scroll"><table id="samples"><thead><tr><th>Prompt</th><th>Expected</th><th>Base output</th><th>LoRA output</th></tr></thead><tbody>{examples}</tbody></table></div></section>
<section><h2>What this experiment does and does not show</h2><ul class="limits">{limitations}</ul></section>
<footer>TuneProof · independent educational project · seed {_escape(r["config"]["seed"])} · {r["duration_seconds"]:.1f}s on this machine<br>
Verify exact artifact bytes with <code>tuneproof verify PATH</code>. Hashes detect changes, not authenticity.
Raw evidence: <a href="results.json">results.json</a> · <a href="curves.json">curves.json</a> · <a href="manifest.json">manifest.json</a></footer>
</main><script>{script}</script></body></html>"""
