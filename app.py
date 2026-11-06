"""Bored Pile Settlement Predictor — Streamlit GUI for the trained LSSVM, LSSVM-PSO and LSSVM-GWO models.

Run locally:   pip install -r requirements.txt   then   streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lssvm_model import MODEL_NAMES, load_models

st.set_page_config(page_title="Bored Pile Settlement Predictor", page_icon="🏗️", layout="wide")

# ----------------------------------------------------------------------------- style
NAVY, TEAL, AMBER, PLUM, GREY, RED = "#12345a", "#1a9c8c", "#e0893a", "#6d4c9f", "#7b8794", "#d9402b"
MODEL_COL = {"LSSVM-PSO": NAVY, "LSSVM-GWO": AMBER, "LSSVM": GREY}
MODEL_DASH = {"LSSVM-PSO": "solid", "LSSVM-GWO": "dash", "LSSVM": "dot"}
st.markdown(f"""
<style>
.block-container {{padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1300px;}}
.hero {{border-radius: 16px; padding: 20px 26px; color: #fff; margin-bottom: 10px;
  background: linear-gradient(110deg, #0f2b4c 0%, #14456e 45%, #13867c 100%);}}
.hero h1 {{font-family: Georgia, 'Times New Roman', serif; font-size: 30px; margin: 0 0 4px 0; color: #fff;}}
.hero p {{margin: 0; opacity: .92; font-size: 14.5px;}}
.badge {{display: inline-block; border: 1px solid rgba(255,255,255,.6); border-radius: 999px; padding: 3px 12px;
  font-size: 12.5px; font-weight: 600; margin: 8px 8px 0 0; background: rgba(255,255,255,.1);}}
.kpi {{border-radius: 12px; padding: 12px 14px; color: #fff; min-height: 104px;}}
.kpi .t {{font-size: 13px; font-weight: 600; opacity: .95;}}
.kpi .v {{font-family: Georgia, serif; font-size: 27px; font-weight: 700; margin-top: 2px; white-space: nowrap;}}
.kpi .v small {{font-size: 15px;}}
.kpi .s {{font-size: 12px; opacity: .9;}}
.k1 {{background: linear-gradient(135deg, #0f2b4c, #1d5b8f);}}
.k2 {{background: linear-gradient(135deg, #0f8a7a, #2cc0a6);}}
.k3 {{background: linear-gradient(135deg, #d0702a, #f0a25a);}}
.k4 {{background: linear-gradient(135deg, #5a3d8a, #8a67c4);}}
.sec {{font-family: Georgia, serif; font-size: 18px; font-weight: 700; border-left: 4px solid {AMBER};
  padding-left: 10px; margin: 6px 0 8px 0;}}
.chip {{display: inline-block; border-radius: 999px; padding: 3px 11px; font-size: 12.5px; font-weight: 700;
  border: 1.5px solid; margin: 0 6px 6px 0;}}
.ok {{color: #1f8a4c; background: #e6f4ec; border-color: #1f8a4c;}}
.warn {{color: #b26a00; background: #fdf1de; border-color: #b26a00;}}
.bad {{color: #b3261e; background: #fbe7e5; border-color: #b3261e;}}
.note {{color: #5b6b80; font-size: 12.5px;}}
</style>
<div class="hero">
  <h1>🏗️ Bored Pile Settlement Predictor</h1>
  <p>Pile-head settlement from pile geometry, SPT soil profile and applied load ·
     LSSVM models optimised by PSO and GWO · 90% conformal prediction intervals</p>
  <span class="badge">51 static load tests · 750 points</span><span class="badge">Trained models loaded from models/</span>
  <span class="badge">Research prototype</span>
</div>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- models
@st.cache_resource
def get_models():
    F, M = load_models()
    Xtr = M["LSSVM-PSO"].training_inputs()
    piles = np.unique(np.round(Xtr[:, :8], 6), axis=0)              # distinct tested piles
    lo, hi = Xtr.min(0), Xtr.max(0)
    b, a = np.polyfit(piles[:, 1], piles[:, 7], 1)                    # sigma'_v,tip vs L
    sd = float(np.std(piles[:, 7] - (a + b * piles[:, 1])))
    z = (Xtr - lo) / np.where(hi > lo, hi - lo, 1)
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
    nn95 = float(np.quantile(d.min(1), 0.95))
    return F, M, Xtr, piles, lo, hi, (a, b, sd), nn95


F, M, XTR, PILES, LO, HI, (A0, A1, RSD), NN95 = get_models()
META = {  # label, unit, step, format
    "D": ("D – pile diameter", "m", 0.05, "%.2f"),
    "L": ("L – pile length", "m", 0.5, "%.2f"),
    "R_c": ("R_c – clay ratio along shaft", "–", 0.01, "%.3f"),
    "N_c_avg": ("N_c,avg – average SPT N in clay", "blows/0.3 m", 0.5, "%.2f"),
    "N_s_avg": ("N_s,avg – average SPT N in sand", "blows/0.3 m", 0.5, "%.2f"),
    "N_tip": ("N_tip – SPT N at pile tip", "blows/0.3 m", 1.0, "%.2f"),
    "I_tc": ("I_tc – tip-clay index", "–", 0.05, "%.3f"),
    "sigma_v_tip": ("σ′_v,tip – effective stress at tip", "t/m²", 0.5, "%.2f"),
    "Q": ("Q – applied load", "t", 50.0, "%.0f"),
}

# ----------------------------------------------------------------------------- inputs
left, right = st.columns([1, 2.15], gap="medium")
with left:
    st.markdown('<div class="sec">Pile and soil inputs</div>', unsafe_allow_html=True)
    opts = ["Custom input"] + [f"Tested pile {i+1} · D {p[0]:.2f} m · L {p[1]:.1f} m" for i, p in enumerate(PILES)]
    choice = st.selectbox("Start from a tested pile", opts, index=1 + int(np.argmin(np.abs(PILES[:, 0] - 1.2) + np.abs(PILES[:, 1] - 61) + np.abs(PILES[:, 2] - 0.803))))
    if choice != "Custom input" and st.session_state.get("_pile") != choice:
        p = PILES[opts.index(choice) - 1]
        for k, v in zip(F[:8], p):
            st.session_state[k] = float(round(v, 4))
        st.session_state["_pile"] = choice
    st.session_state.setdefault("Q", 1600.0)
    c = st.columns(3)
    x = []
    for j, k in enumerate(F[:8]):
        lab, unit, step, fmt = META[k]
        with c[j % 3]:
            x.append(st.number_input(f"{k.replace('_', ' ')} ({unit})", key=k, step=step, format=fmt, help=lab))
    q = st.slider("Q – applied load (t)", 0.0, 3300.0, step=10.0, key="Q")
    st.caption(f"Q = {q:,.0f} t = {q * 9.81:,.0f} kN")
    x.append(q)
    x = np.array(x, float)
    main = st.selectbox("Main model", ["LSSVM-PSO", "LSSVM-GWO", "LSSVM"],
                        format_func=lambda n: n + (" (selected model)" if n == "LSSVM-PSO" else ""))

    # applicability domain
    st.markdown('<div class="sec">Applicability domain</div>', unsafe_allow_html=True)
    chips, worst = [], "ok"
    for j, k in enumerate(F):
        if x[j] < LO[j] - 1e-9 or x[j] > HI[j] + 1e-9:
            chips.append(("bad", f"✗ {k} outside {LO[j]:.4g}–{HI[j]:.4g}")); worst = "bad"
    if worst == "ok":
        chips.append(("ok", "✓ All inputs within tested ranges"))
    exp_s = A0 + A1 * x[1]
    if abs(x[7] - exp_s) > 3 * RSD:
        chips.append(("warn", f"⚠ σ′v,tip unusual for L = {x[1]:.1f} m (expected ≈ {exp_s:.1f})")); worst = max(worst, "warn", key=["ok", "warn", "bad"].index)
    else:
        chips.append(("ok", "✓ σ′v,tip consistent with L"))
    z = (x - LO) / np.where(HI > LO, HI - LO, 1); zt = (XTR - LO) / np.where(HI > LO, HI - LO, 1)
    if np.sqrt(((zt - z) ** 2).sum(1)).min() > 1.5 * NN95:
        chips.append(("warn", "⚠ Far from tested combinations")); worst = max(worst, "warn", key=["ok", "warn", "bad"].index)
    else:
        chips.append(("ok", "✓ Close to tested data"))
    chips.append({"ok": ("ok", "✓ Inside applicability domain"), "warn": ("warn", "⚠ Near the edge of the domain"),
                  "bad": ("bad", "✗ Extrapolation: use with caution")}[worst])
    st.markdown("".join(f'<span class="chip {c_}">{t}</span>' for c_, t in chips), unsafe_allow_html=True)
    st.markdown('<p class="note">1 t = 9.81 kN; 1 t/m² = 9.81 kPa. Predictions outside the tested ranges are extrapolations.</p>',
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------- predictions
pred = {n: float(M[n].predict(x)[0]) for n in MODEL_NAMES}
other = "LSSVM-GWO" if main != "LSSVM-GWO" else "LSSVM-PSO"
qm = M[main].q90
top = float(min(3300, max(500, np.ceil(max(x[8] * 1.25, 1000) / 100) * 100)))
Qs = np.linspace(0, top, 61)
curve = {n: M[n].predict(np.column_stack([np.tile(x[:8], (len(Qs), 1)), Qs])) for n in MODEL_NAMES}


def layout(fig, xt, yt, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10), template="simple_white",
                      font=dict(family="Times New Roman, serif", size=14),
                      legend=dict(orientation="h", y=-0.22, x=0), xaxis_title=xt, yaxis_title=yt)
    fig.update_xaxes(showgrid=True, gridcolor="#e3e8ee", mirror=True, ticks="inside", showline=True)
    fig.update_yaxes(showgrid=True, gridcolor="#e3e8ee", mirror=True, ticks="inside", showline=True)
    return fig


with right:
    k = st.columns(4)
    cards = [("k1", "Predicted settlement", f"{pred[main]:.2f}", f"{main}" + (" · selected model" if main == "LSSVM-PSO" else "")),
             ("k2", "90% interval", f"{max(0, pred[main]-qm):.2f}–{pred[main]+qm:.2f}", f"± {qm:.2f} mm, calibrated on 150 testing samples"),
             ("k3", f"{other} prediction", f"{pred[other]:.2f}", f"difference {pred[other]-pred[main]:+.2f} mm"),
             ("k4", "Equivalent load", f"{x[8]*9.81/1000:,.2f}", "MN applied at the pile head")]
    for col, (cls, t, v, s) in zip(k, cards):
        unit = "MN" if cls == "k4" else "mm"
        col.markdown(f'<div class="kpi {cls}"><div class="t">{t}</div><div class="v">{v} <small>{unit}</small></div>'
                     f'<div class="s">{s}</div></div>', unsafe_allow_html=True)
    m = M[main]
    st.caption(f"{main}: γ = {m.gamma:.3g}, σ = {m.sigma:.4f}, {len(m.core.alpha_)} support vectors · loaded from {m.source}")

    t1, t2, t3 = st.tabs(["Settlement prediction", "Load–settlement curve", "Model comparison"])

    def curve_fig(models, h):
        fig = go.Figure()
        c0 = curve[main]
        fig.add_trace(go.Scatter(x=np.r_[Qs, Qs[::-1]], y=np.r_[c0 + qm, (c0 - qm)[::-1]], fill="toself",
                                 fillcolor="rgba(18,52,90,0.15)", line=dict(width=0), name="90% interval", hoverinfo="skip"))
        for n in models:
            fig.add_trace(go.Scatter(x=Qs, y=curve[n], name=n + (" (main)" if n == main else ""),
                                     line=dict(color=MODEL_COL[n], dash=MODEL_DASH[n], width=3 if n == main else 2)))
        fig.add_trace(go.Scatter(x=[x[8]], y=[pred[main]], mode="markers", name="Current load",
                                 marker=dict(size=13, color=TEAL, line=dict(color="white", width=2))))
        return layout(fig, "Applied load, Q (t)", "Settlement, s (mm)", h)

    with t1:
        a, b = st.columns(2)
        with a:
            st.markdown('<div class="sec">Applicability domain</div>', unsafe_allow_html=True)
            Ls = np.array([38, 72]); fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.r_[Ls, Ls[::-1]], y=np.r_[A0 + A1 * Ls + 3 * RSD, (A0 + A1 * Ls - 3 * RSD)[::-1]],
                                     fill="toself", fillcolor="rgba(26,156,140,0.18)", line=dict(width=0), name="Tested range (±3 SD)"))
            fig.add_trace(go.Scatter(x=PILES[:, 1], y=PILES[:, 7], mode="markers", name="Tested pile",
                                     marker=dict(size=9, color="white", line=dict(color=NAVY, width=1.6))))
            fig.add_trace(go.Scatter(x=[x[1]], y=[x[7]], mode="markers", name="Current input",
                                     marker=dict(symbol="diamond", size=16, color=AMBER, line=dict(color="#4a2a0c", width=1.4))))
            st.plotly_chart(layout(fig, "Pile length, L (m)", "σ′v,tip (t/m²)"), use_container_width=True)
        with b:
            st.markdown('<div class="sec">Predicted load–settlement curve</div>', unsafe_allow_html=True)
            st.plotly_chart(curve_fig([main], 380), use_container_width=True)

    with t2:
        st.markdown('<div class="sec">Load–settlement curve of the current pile</div>', unsafe_allow_html=True)
        st.plotly_chart(curve_fig(MODEL_NAMES, 440), use_container_width=True)
        tbl = pd.DataFrame({"Q (t)": Qs[::2], "Q (kN)": np.round(Qs[::2] * 9.81)})
        for n in MODEL_NAMES:
            tbl[f"{n} (mm)"] = np.round(curve[n][::2], 3)
        tbl[f"{main} lower 90% (mm)"] = np.round(curve[main][::2] - qm, 3)
        tbl[f"{main} upper 90% (mm)"] = np.round(curve[main][::2] + qm, 3)
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=320)
        st.download_button("⬇ Download table (CSV)", tbl.to_csv(index=False).encode(), "load_settlement_prediction.csv", "text/csv")

    with t3:
        a, b = st.columns(2)
        with a:
            st.markdown('<div class="sec">Prediction of each model</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=MODEL_NAMES, y=[pred[n] for n in MODEL_NAMES],
                                   marker_color=[MODEL_COL[n] for n in MODEL_NAMES],
                                   error_y=dict(type="data", array=[M[n].q90 for n in MODEL_NAMES], color="#15233a", thickness=1.6),
                                   text=[f"{pred[n]:.2f} mm" for n in MODEL_NAMES], textposition="outside"))
            fig.update_layout(showlegend=False)
            st.plotly_chart(layout(fig, "", "Settlement, s (mm)"), use_container_width=True)
        with b:
            st.markdown('<div class="sec">Testing performance</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"Model": MODEL_NAMES, "R²": [0.820, 0.979, 0.966], "RMSE (mm)": [2.219, 0.764, 0.964],
                                       "MAE (mm)": [1.557, 0.506, 0.472], "a20-index": [0.497, 0.902, 0.937]}),
                         hide_index=True, use_container_width=True)
            st.caption("150 testing samples (from the manuscript).")
        st.markdown('<div class="sec">One-at-a-time sensitivity at the current input</div>', unsafe_allow_html=True)
        p10, p90 = np.percentile(XTR, 10, axis=0), np.percentile(XTR, 90, axis=0)
        rows = []
        for j, kk in enumerate(F):
            lo_, hi_ = x.copy(), x.copy(); lo_[j], hi_[j] = p10[j], p90[j]
            rows.append((kk, float(M[main].predict(lo_)[0]) - pred[main], float(M[main].predict(hi_)[0]) - pred[main]))
        rows.sort(key=lambda r: max(abs(r[1]), abs(r[2])))
        fig = go.Figure()
        fig.add_trace(go.Bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h", name="Input at 10th percentile", marker_color=TEAL))
        fig.add_trace(go.Bar(y=[r[0] for r in rows], x=[r[2] for r in rows], orientation="h", name="Input at 90th percentile", marker_color=AMBER))
        fig.update_layout(barmode="overlay")
        st.plotly_chart(layout(fig, "Change in predicted settlement (mm)", "", 360), use_container_width=True)
        st.caption("Inputs are varied one at a time, so correlated inputs (L and σ′v,tip) can create combinations outside the tested data.")

st.markdown('<p class="note" style="text-align:center">Predictions use the trained LSSVM, LSSVM-PSO and LSSVM-GWO models '
            'saved from the study. For research use; verify with load tests before design.</p>', unsafe_allow_html=True)
