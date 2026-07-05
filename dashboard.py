# dashboard.py
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
sys.path.append(str(ROOT))


try:
    from data_loader import DataConfig, FEATURES, load_data

    DATA_PIPELINE_AVAILABLE = True
    DATA_PIPELINE_ERROR = None
except Exception as exc:  # pragma: no cover - defensive app fallback
    DATA_PIPELINE_AVAILABLE = False
    DATA_PIPELINE_ERROR = exc
    FEATURES = ["log_returns", "volatility", "rsi", "macd_norm", "momentum_5"]
    DataConfig = None
    load_data = None


try:
    from environment import EnvConfig, TradingEnv
    from evaluate import load_model_and_norm

    DRL_AVAILABLE = True
    DRL_ERROR = None
except Exception as exc:  # pragma: no cover - stable-baselines is optional
    DRL_AVAILABLE = False
    DRL_ERROR = exc
    EnvConfig = None
    TradingEnv = None
    load_model_and_norm = None


st.set_page_config(
    page_title="DRL Trading Desk",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #07090d;
        --panel: #10151d;
        --panel-2: #141b25;
        --line: #273244;
        --text: #edf3ff;
        --muted: #91a0b8;
        --green: #18c99f;
        --blue: #4da3ff;
        --amber: #f0b429;
        --red: #ff5a6e;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: #0b0f15;
        border-right: 1px solid var(--line);
    }
    [data-testid="stHeader"] {
        background: rgba(7, 9, 13, 0.94);
    }
    .block-container {
        max-width: 1560px;
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .desk-header {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #0d1219;
        padding: 22px 24px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        gap: 20px;
        align-items: flex-start;
    }
    .desk-title {
        font-size: 2.15rem;
        font-weight: 760;
        line-height: 1.06;
        margin: 0;
        color: var(--text);
    }
    .desk-subtitle {
        color: var(--muted);
        margin-top: 8px;
        max-width: 880px;
        font-size: 0.98rem;
    }
    .eyebrow {
        color: var(--green);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 760;
        margin-bottom: 8px;
    }
    .scope-grid {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: flex-end;
    }
    .scope-pill {
        border: 1px solid var(--line);
        color: #c6d4ea;
        background: #111822;
        border-radius: 999px;
        padding: 7px 10px;
        font-size: 0.78rem;
        white-space: nowrap;
    }
    .section-title {
        margin: 18px 0 10px;
    }
    .section-title h2 {
        margin: 0;
        font-size: 1.18rem;
        color: var(--text);
    }
    .section-title p {
        margin: 4px 0 0;
        color: var(--muted);
        font-size: 0.88rem;
    }
    .note-box {
        border: 1px solid var(--line);
        background: #0d1219;
        border-radius: 8px;
        padding: 14px 16px;
        color: #c5d1e3;
        font-size: 0.9rem;
    }
    .compact-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 4px;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 4px solid var(--green);
        border-radius: 8px;
        padding: 14px 14px 12px;
        min-height: 106px;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.02em;
    }
    div[data-testid="stMetricValue"] {
        color: var(--text);
        font-weight: 760;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
    div[data-testid="stTabs"] button {
        color: #b6c3d9;
        border-radius: 0;
        padding-top: 12px;
        padding-bottom: 12px;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--text);
        border-bottom: 2px solid var(--green);
    }
    .stButton > button {
        border-radius: 7px;
        border: 1px solid var(--line);
        font-weight: 700;
    }
    .stDownloadButton > button {
        border-radius: 7px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }
    hr {
        border-color: var(--line);
    }
    @media (max-width: 900px) {
        .desk-header {
            display: block;
        }
        .scope-grid {
            justify-content: flex-start;
            margin-top: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Surcouche "terminal" : fontes + composants (cascade sur le CSS de base) ──
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, sans-serif;
        font-feature-settings: 'tnum' 1;
    }
    /* Les CHIFFRES en mono partout : la signature terminal */
    div[data-testid="stMetricValue"],
    [data-testid="stDataFrame"] *,
    .kpi .v, .tape, .num {
        font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
    }

    /* ── bandeau défilant façon tape ── */
    .tape {
        display: flex; gap: 26px; overflow: hidden; white-space: nowrap;
        border: 1px solid var(--line); border-radius: 8px;
        background: #0a0f16; padding: 9px 16px; margin-bottom: 14px;
        font-size: 0.82rem; color: var(--muted);
    }
    .tape b.up { color: var(--green); }
    .tape b.dn { color: var(--red); }

    /* ── cartes KPI custom ── */
    .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
               gap: 10px; margin: 4px 0 16px; }
    .kpi {
        background: linear-gradient(180deg, #121926 0%, #0e141d 100%);
        border: 1px solid var(--line); border-radius: 9px; padding: 13px 15px 11px;
        position: relative; overflow: hidden;
    }
    .kpi::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 3px;
                   background: var(--accent, var(--green)); }
    .kpi .l { color: var(--muted); font-size: 0.7rem; text-transform: uppercase;
              letter-spacing: 0.09em; margin-bottom: 5px; }
    .kpi .v { font-size: 1.62rem; font-weight: 700; line-height: 1.05;
              color: var(--text); }
    .kpi .v.up { color: var(--green); }
    .kpi .v.dn { color: var(--red); }
    .kpi .s { color: var(--muted); font-size: 0.74rem; margin-top: 5px; }

    /* ── badges verdict É1-É6 ── */
    .badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 14px; }
    .badge {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 600;
        border-radius: 7px; padding: 7px 12px; border: 1px solid;
    }
    .badge.pass { color: var(--green); border-color: rgba(24,201,159,.45);
                  background: rgba(24,201,159,.09); }
    .badge.fail { color: var(--red); border-color: rgba(255,90,110,.45);
                  background: rgba(255,90,110,.09); }
    .verdict-banner {
        font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 1.05rem;
        border-radius: 9px; padding: 13px 18px; margin: 8px 0 16px; border: 1px solid;
    }
    .verdict-banner.nogo { color: var(--red); border-color: rgba(255,90,110,.5);
                           background: rgba(255,90,110,.07); }
    .verdict-banner.go { color: var(--green); border-color: rgba(24,201,159,.5);
                         background: rgba(24,201,159,.07); }

    /* ── titres de section avec barre d'accent ── */
    .section-title h2 { padding-left: 10px; border-left: 3px solid var(--green); }
    </style>
    """,
    unsafe_allow_html=True,
)


ACTION_LABELS = {
    "hold_pct": "Hold",
    "long_pct": "Long",
    "flat_pct": "Flat",
    "short_pct": "Short",
}
ACTION_COLORS = {
    "Hold": "#8fa3bd",
    "Long": "#18c99f",
    "Flat": "#f0b429",
    "Short": "#ff5a6e",
}
REGIME_COLORS = {
    "Risk-on": "#18c99f",
    "Trend up, high vol": "#4da3ff",
    "Distribution": "#f0b429",
    "Stress": "#ff5a6e",
}


@st.cache_data(show_spinner=False)
def load_reports() -> dict:
    report_files = {
        "metrics": REPORTS_DIR / "metrics.json",
        "walk_forward": REPORTS_DIR / "walk_forward.json",
        "seed_robustness": REPORTS_DIR / "seed_robustness.json",
        "stress_tests": REPORTS_DIR / "stress_tests.json",
        "ablation_bug": REPORTS_DIR / "ablation_bug.json",
        "regime_experiment": REPORTS_DIR / "regime_experiment.json",
        "diffusion_validation": REPORTS_DIR / "diffusion_validation.json",
    }

    loaded = {}
    for key, path in report_files.items():
        if not path.exists():
            loaded[key] = {}
            continue
        with path.open("r", encoding="utf-8") as fh:
            loaded[key] = json.load(fh)
    return loaded


@st.cache_data(show_spinner=False)
def get_market_data(ticker, start_date, end_date, vol_window, rsi_window, sma_window):
    if not DATA_PIPELINE_AVAILABLE:
        raise RuntimeError(f"Pipeline indisponible: {DATA_PIPELINE_ERROR}")

    cfg = DataConfig(
        ticker=ticker,
        start_date=str(start_date),
        end_date=str(end_date),
        vol_window=int(vol_window),
        rsi_window=int(rsi_window),
        sma_window=int(sma_window),
    )
    train, val, test, scaler = load_data(cfg)
    return train, val, test


def section(title: str, subtitle: str = ""):
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-title">
            <h2>{title}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[dict]):
    """
    Bandeau de KPI façon terminal. Chaque carte : {label, value, sub, tone}
    avec tone ∈ {'up', 'dn', ''} (couleur de la valeur) — accent latéral assorti.
    """
    tones = {"up": "var(--green)", "dn": "var(--red)", "": "var(--blue)"}
    html = ['<div class="kpi-row">']
    for c in cards:
        tone = c.get("tone", "")
        html.append(
            f'<div class="kpi" style="--accent:{tones.get(tone, tones[""])}">'
            f'<div class="l">{c["label"]}</div>'
            f'<div class="v {tone}">{c["value"]}</div>'
            f'<div class="s">{c.get("sub", "")}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def fmt_pct(value, digits=1, signed=True):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}%}"


def fmt_dd(value, digits=1):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return f"-{abs(value):.{digits}%}"


def fmt_num(value, digits=2):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_money(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"${value:,.0f}"


def percent_points(df: pd.DataFrame, columns: list[str], invert_abs: list[str] | None = None):
    view = df.copy()
    for column in invert_abs or []:
        if column in view.columns:
            view[column] = -pd.to_numeric(view[column], errors="coerce").abs()
    for column in columns:
        if column in view.columns:
            view[column] = pd.to_numeric(view[column], errors="coerce") * 100
    return view


def style_fig(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0c1119",
        font=dict(color="#dce6f7", family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=34, r=24, t=44, b=34),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#141b25", bordercolor="#273244",
                        font=dict(family="IBM Plex Mono, monospace", size=12)),
    )
    fig.update_xaxes(
        gridcolor="rgba(145,160,184,0.10)", zerolinecolor="#273244",
        tickfont=dict(family="IBM Plex Mono, monospace", size=11),
    )
    fig.update_yaxes(
        gridcolor="rgba(145,160,184,0.10)", zerolinecolor="#273244",
        tickfont=dict(family="IBM Plex Mono, monospace", size=11),
    )
    return fig


def empty_fig(message: str, height=320):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=15, color="#91a0b8"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig, height=height)


def report_model_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, values in metrics.get("full_test", {}).items():
        rows.append({"model": model_name, **values})
    return pd.DataFrame(rows)


def cross_ticker_frame(metrics: dict) -> pd.DataFrame:
    cross = metrics.get("cross_ticker", {})
    if not cross:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(cross, orient="index")
    df.index.name = "ticker"
    return df.reset_index()


def robust_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, values in metrics.get("robustesse", {}).items():
        rows.append({"model": model_name, **values})
    return pd.DataFrame(rows)


def walk_forward_frame(walk_forward: dict) -> pd.DataFrame:
    rows = []
    for fold in walk_forward.get("folds", []):
        year = fold.get("test_year")
        for ticker, values in fold.get("per_ticker", {}).items():
            rows.append({"year": year, "ticker": ticker, **values})
    return pd.DataFrame(rows)


def fee_frame(stress_tests: dict) -> pd.DataFrame:
    rows = []
    for model_name, grid in stress_tests.get("fee_grid", {}).items():
        for fee, values in grid.items():
            rows.append(
                {
                    "model": model_name,
                    "fee": float(fee),
                    "fee_bps": float(fee) * 10_000,
                    **values,
                }
            )
    return pd.DataFrame(rows)


def market_stats(data: pd.DataFrame) -> dict:
    price = data["price"].astype(float).dropna()
    returns = price.pct_change().dropna()
    if price.empty or len(price) < 2:
        return {}

    total_return = price.iloc[-1] / price.iloc[0] - 1
    years = max(len(returns) / 252.0, 1 / 252.0)
    ann_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else np.nan
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    equity = price / price.iloc[0]
    drawdown = equity / equity.cummax() - 1
    cvar_95 = returns[returns <= returns.quantile(0.05)].mean() if len(returns) >= 20 else np.nan

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": drawdown.min(),
        "hit_rate": (returns > 0).mean(),
        "cvar_95": cvar_95,
        "best_day": returns.max(),
        "worst_day": returns.min(),
    }


def build_market_regimes(data: pd.DataFrame) -> pd.DataFrame:
    price = data["price"].astype(float)
    returns = price.pct_change().fillna(0.0)
    vol_20 = returns.rolling(20, min_periods=5).std() * np.sqrt(252)
    mom_63 = price.pct_change(63)
    high_vol = vol_20.quantile(0.75)

    regimes = np.select(
        [
            (mom_63 >= 0) & (vol_20 <= high_vol),
            (mom_63 >= 0) & (vol_20 > high_vol),
            (mom_63 < 0) & (vol_20 <= high_vol),
            (mom_63 < 0) & (vol_20 > high_vol),
        ],
        ["Risk-on", "Trend up, high vol", "Distribution", "Stress"],
        default="Distribution",
    )

    frame = pd.DataFrame(index=data.index)
    frame["returns"] = returns
    frame["vol_20"] = vol_20
    frame["mom_63"] = mom_63
    frame["regime"] = regimes
    frame["drawdown"] = price / price.cummax() - 1
    frame["equity"] = price / price.iloc[0] * 10_000
    return frame


def make_overview_chart(model_df: pd.DataFrame):
    if model_df.empty:
        return empty_fig("Aucun rapport de performance disponible.")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Performance test", "Risque et rendement ajusté"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=model_df["model"],
            y=model_df["return"],
            name="Agent",
            marker_color="#18c99f",
            text=[fmt_pct(v) for v in model_df["return"]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=model_df["model"],
            y=model_df["bh"],
            name="Buy & Hold",
            marker_color="#4da3ff",
            text=[fmt_pct(v) for v in model_df["bh"]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=model_df["model"],
            y=model_df["alpha"],
            name="Alpha",
            marker_color=[
                "#18c99f" if value >= 0 else "#ff5a6e" for value in model_df["alpha"]
            ],
            text=[fmt_pct(v) for v in model_df["alpha"]],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=model_df["model"],
            y=-model_df["max_dd"].abs(),
            name="Max drawdown",
            marker_color="#ff5a6e",
            text=[fmt_dd(v) for v in model_df["max_dd"]],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    return style_fig(fig, height=430)


def make_action_chart(model_df: pd.DataFrame):
    if model_df.empty:
        return empty_fig("Aucune distribution d'actions disponible.")

    fig = go.Figure()
    for key, label in ACTION_LABELS.items():
        if key not in model_df.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=model_df["model"],
                y=model_df[key],
                name=label,
                marker_color=ACTION_COLORS[label],
                text=[fmt_pct(v, digits=0, signed=False) for v in model_df[key]],
                textposition="inside",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return style_fig(fig, height=330)


def make_cross_chart(cross_df: pd.DataFrame):
    if cross_df.empty:
        return empty_fig("Aucune donnée cross-ticker disponible.")

    ordered = cross_df.sort_values("alpha", ascending=False)
    fig = go.Figure(
        go.Bar(
            x=ordered["ticker"],
            y=ordered["alpha"],
            marker_color=["#18c99f" if v >= 0 else "#ff5a6e" for v in ordered["alpha"]],
            text=[fmt_pct(v) for v in ordered["alpha"]],
            textposition="outside",
            name="Alpha",
        )
    )
    fig.add_hline(y=0, line_color="#91a0b8", line_width=1)
    fig.update_yaxes(tickformat=".0%")
    return style_fig(fig, height=390)


def make_risk_return_chart(cross_df: pd.DataFrame):
    if cross_df.empty:
        return empty_fig("Aucune donnée risque/rendement disponible.")

    fig = go.Figure(
        go.Scatter(
            x=cross_df["max_dd"],
            y=cross_df["return"],
            mode="markers+text",
            text=cross_df["ticker"],
            textposition="top center",
            marker=dict(
                size=np.clip(cross_df["sharpe"].fillna(0).abs() * 18 + 12, 12, 42),
                color=cross_df["alpha"],
                colorscale=[[0, "#ff5a6e"], [0.5, "#f0b429"], [1, "#18c99f"]],
                showscale=True,
                colorbar=dict(title="Alpha"),
                line=dict(color="#edf3ff", width=1),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>Return %{y:.1%}<br>"
                "Max DD -%{x:.1%}<br>Alpha %{marker.color:.1%}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Max drawdown", tickformat=".0%", autorange="reversed")
    fig.update_yaxes(title="Return", tickformat=".0%")
    return style_fig(fig, height=390)


def make_walk_forward_heatmap(wf_df: pd.DataFrame):
    if wf_df.empty:
        return empty_fig("Aucun walk-forward disponible.")

    pivot = wf_df.pivot(index="year", columns="ticker", values="alpha").sort_index()
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index.astype(str),
            zmid=0,
            colorscale=[[0, "#ff5a6e"], [0.5, "#111822"], [1, "#18c99f"]],
            text=np.vectorize(lambda v: fmt_pct(v, digits=0))(pivot.values),
            texttemplate="%{text}",
            colorbar=dict(title="Alpha"),
        )
    )
    fig.update_xaxes(side="top")
    return style_fig(fig, height=390)


def make_walk_forward_year_chart(walk_forward: dict):
    folds = walk_forward.get("folds", [])
    if not folds:
        return empty_fig("Aucun agrégat annuel disponible.")

    years = [fold.get("test_year") for fold in folds]
    mean_alpha = [fold.get("mean_alpha", np.nan) for fold in folds]
    mean_return = [fold.get("mean_return", np.nan) for fold in folds]
    mean_bh = [fold.get("mean_bh", np.nan) for fold in folds]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=years,
            y=mean_alpha,
            name="Alpha moyen",
            marker_color=["#18c99f" if v >= 0 else "#ff5a6e" for v in mean_alpha],
            text=[fmt_pct(v) for v in mean_alpha],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=mean_return,
            name="Return agent",
            mode="lines+markers",
            line=dict(color="#18c99f", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=mean_bh,
            name="Buy & Hold",
            mode="lines+markers",
            line=dict(color="#4da3ff", width=2, dash="dash"),
        )
    )
    fig.add_hline(y=0, line_color="#91a0b8", line_width=1)
    fig.update_yaxes(tickformat=".0%")
    return style_fig(fig, height=390)


def make_fee_chart(fees: pd.DataFrame):
    if fees.empty:
        return empty_fig("Aucun stress test de frais disponible.")

    fig = go.Figure()
    for model, subset in fees.groupby("model"):
        fig.add_trace(
            go.Scatter(
                x=subset["fee_bps"],
                y=subset["alpha"],
                mode="lines+markers",
                name=model,
                line=dict(width=3),
                hovertemplate="Frais %{x:.0f} bps<br>Alpha %{y:.1%}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color="#91a0b8", line_width=1)
    fig.update_xaxes(title="Frais de transaction, aller simple (bps)")
    fig.update_yaxes(title="Alpha", tickformat=".0%")
    return style_fig(fig, height=380)


def make_kill_switch_chart(stress_tests: dict):
    raw = stress_tests.get("no_killswitch", {})
    if not raw:
        return empty_fig("Aucun test kill-switch disponible.")

    rows = []
    for model, values in raw.items():
        rows.append(
            {
                "model": model,
                "alpha_no_stop": values.get("alpha"),
                "alpha_with_stop": values.get("alpha_with_stop"),
                "max_dd_no_stop": values.get("max_dd"),
                "stop_contribution": values.get("stop_contribution"),
            }
        )
    df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["model"],
            y=df["alpha_no_stop"],
            name="Sans stop",
            marker_color="#ff5a6e",
            text=[fmt_pct(v) for v in df["alpha_no_stop"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["model"],
            y=df["alpha_with_stop"],
            name="Avec stop",
            marker_color="#18c99f",
            text=[fmt_pct(v) for v in df["alpha_with_stop"]],
            textposition="outside",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    return style_fig(fig, height=380)


def make_market_chart(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, ticker: str):
    data = pd.concat([train, val, test])
    regimes = build_market_regimes(data)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.56, 0.22, 0.22],
        vertical_spacing=0.055,
        subplot_titles=(f"{ticker} price and regimes", "Drawdown", "20-day annualized volatility"),
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["price"],
            mode="lines",
            line=dict(color="#dce6f7", width=1.6),
            name="Price",
        ),
        row=1,
        col=1,
    )
    for regime, color in REGIME_COLORS.items():
        mask = regimes["regime"] == regime
        fig.add_trace(
            go.Scatter(
                x=data.index[mask],
                y=data.loc[mask, "price"],
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.72),
                name=regime,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=regimes["drawdown"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(255,90,110,0.22)",
            line=dict(color="#ff5a6e", width=1.4),
            name="Drawdown",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=regimes["vol_20"],
            mode="lines",
            line=dict(color="#f0b429", width=1.5),
            name="Volatility",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    for split_date, label, color in [
        (train.index[-1], "Train | Val", "#91a0b8"),
        (val.index[-1], "Val | Test", "#f0b429"),
    ]:
        for row in [1, 2, 3]:
            fig.add_vline(x=split_date, line_dash="dash", line_color=color, row=row, col=1)
        fig.add_annotation(
            x=split_date,
            y=data["price"].max(),
            text=label,
            showarrow=False,
            yshift=12,
            font=dict(color=color, size=11),
            row=1,
            col=1,
        )

    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    return style_fig(fig, height=700)


def make_feature_chart(data: pd.DataFrame, selected_features: list[str]):
    if not selected_features:
        return empty_fig("Sélectionne au moins une feature.")

    fig = make_subplots(
        rows=len(selected_features),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=selected_features,
    )
    palette = {
        "log_returns": "#18c99f",
        "volatility": "#f0b429",
        "rsi": "#4da3ff",
        "macd_norm": "#b279ff",
        "momentum_5": "#ff8f5a",
    }
    for idx, feature in enumerate(selected_features, start=1):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[feature],
                mode="lines",
                line=dict(color=palette.get(feature, "#dce6f7"), width=1.4),
                name=feature,
                showlegend=False,
            ),
            row=idx,
            col=1,
        )
        fig.add_hline(y=0, line_color="rgba(145,160,184,0.28)", line_width=1, row=idx, col=1)
    return style_fig(fig, height=max(300, 185 * len(selected_features)))


def split_stats_frame(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, split in splits.items():
        stats = market_stats(split)
        rows.append(
            {
                "split": name,
                "days": len(split),
                "start": split.index[0].date().isoformat(),
                "end": split.index[-1].date().isoformat(),
                "return": stats.get("total_return"),
                "ann_vol": stats.get("ann_vol"),
                "sharpe": stats.get("sharpe"),
                "max_dd": stats.get("max_dd"),
            }
        )
    return pd.DataFrame(rows)


def run_drl_episode(model_path, data, cfg_env, deterministic: bool, seed: int):
    model, vec_normalize = load_model_and_norm(model_path, data, cfg_env)
    with contextlib.redirect_stdout(io.StringIO()):
        env = TradingEnv(data=data, cfg=cfg_env)
        obs, _ = env.reset(seed=seed, options={"random_start": not deterministic})
        done = False
        while not done:
            if vec_normalize is not None:
                obs_input = vec_normalize.normalize_obs(np.array([obs], dtype=np.float32))[0]
            else:
                obs_input = obs
            action, _ = model.predict(obs_input, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

    portfolio = np.asarray(env.history["portfolio_values"], dtype=float)
    prices = np.asarray(env.history["prices"], dtype=float)
    actions = np.asarray(env.history["actions"], dtype=int)
    returns = np.diff(portfolio) / (portfolio[:-1] + 1e-8)
    peak = np.maximum.accumulate(portfolio)
    max_dd = np.max((peak - portfolio) / (peak + 1e-8)) if len(portfolio) else np.nan
    total_return = portfolio[-1] / portfolio[0] - 1 if len(portfolio) > 1 else np.nan
    bh_return = prices[-1] / prices[0] - 1 if len(prices) > 1 else np.nan
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252) if len(returns) else np.nan

    return {
        "portfolio": portfolio,
        "prices": prices,
        "actions": actions,
        "return": total_return,
        "bh": bh_return,
        "alpha": total_return - bh_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "n_trades": env.history["n_trades"],
        "vec_normalize": vec_normalize is not None,
    }


reports = load_reports()
metrics = reports.get("metrics", {})
walk_forward = reports.get("walk_forward", {})
seed_robustness = reports.get("seed_robustness", {})
stress_tests = reports.get("stress_tests", {})
diffusion_val = reports.get("diffusion_validation", {})

model_df = report_model_frame(metrics)
cross_df = cross_ticker_frame(metrics)
robust_df = robust_frame(metrics)
wf_df = walk_forward_frame(walk_forward)
fees_df = fee_frame(stress_tests)


with st.sidebar:
    st.markdown("### DRL Trading Desk")
    st.caption("Streamlit cockpit pour PPO, validation multi-actifs et contrôle du risque.")
    st.markdown("---")

    st.markdown("#### Marché")
    ticker = st.selectbox(
        "Actif",
        ["AAPL", "MSFT", "GOOGL", "SPY", "TSLA", "BTC-USD", "ETH-USD"],
        index=0,
    )
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Début", value=pd.to_datetime("2010-01-01"))
    with col_end:
        end_date = st.date_input("Fin", value=pd.to_datetime("2023-01-01"))

    vol_window = st.slider("Volatilité rolling", 5, 60, 20)
    rsi_window = st.slider("RSI", 5, 30, 14)
    sma_window = st.slider("SMA", 20, 200, 50)

    load_btn = st.button("Charger le marché", type="primary", use_container_width=True)
    if load_btn:
        if start_date >= end_date:
            st.error("La date de début doit être antérieure à la date de fin.")
        else:
            with st.spinner(f"Chargement de {ticker}..."):
                try:
                    train, val, test = get_market_data(
                        ticker,
                        start_date,
                        end_date,
                        vol_window,
                        rsi_window,
                        sma_window,
                    )
                    st.session_state.market_payload = {
                        "ticker": ticker,
                        "train": train,
                        "val": val,
                        "test": test,
                    }
                    st.success(f"{ticker} chargé.")
                except Exception as exc:
                    st.error(f"Chargement impossible: {exc}")

    st.markdown("---")
    st.markdown("#### Scope")
    st.markdown(
        """
        <div class="note-box">
            PPO, backtests, walk-forward, robustesse, stress tests —
            et le verdict du générateur de scénarios (onglet Génératif).
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Tape cross-ticker : l'alpha par actif en un coup d'œil, façon bandeau ──
if not cross_df.empty:
    tape_cells = " ".join(
        f'<span>{row.ticker} '
        f'<b class="{"up" if row.alpha > 0 else "dn"}">{row.alpha:+.1%}</b></span>'
        for row in cross_df.itertuples()
    )
    st.markdown(
        f'<div class="tape"><span style="color:var(--green);font-weight:700">'
        f'ALPHA×TICKER</span> {tape_cells} '
        f'<span style="color:var(--muted)">· test 2021-22 · modèle Multi</span></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="desk-header">
        <div>
            <div class="eyebrow">Portfolio reinforcement learning · IA générative</div>
            <h1 class="desk-title">DRL Trading Desk</h1>
            <div class="desk-subtitle">
                Vue opérateur pour analyser l'agent PPO : performance test,
                alpha cross-ticker, robustesse walk-forward, budget de risque,
                et verdict du générateur de scénarios (protocole pré-enregistré).
            </div>
        </div>
        <div class="scope-grid">
            <span class="scope-pill">PPO</span>
            <span class="scope-pill">Multi-asset</span>
            <span class="scope-pill">Walk-forward</span>
            <span class="scope-pill">DDPM · É1-É6</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


multi_row = {}
if not model_df.empty:
    multi_match = model_df[model_df["model"].str.contains("Multi", case=False, na=False)]
    multi_row = (multi_match.iloc[0] if not multi_match.empty else model_df.iloc[0]).to_dict()

cross_positive = int((cross_df["alpha"] > 0).sum()) if not cross_df.empty else 0
walk_agg = walk_forward.get("aggregate", {})
robust_multi = {}
if not robust_df.empty:
    robust_match = robust_df[robust_df["model"].str.contains("Multi", case=False, na=False)]
    robust_multi = (robust_match.iloc[0] if not robust_match.empty else robust_df.iloc[0]).to_dict()

_alpha = multi_row.get("alpha")
_sharpe = multi_row.get("sharpe")
kpi_row([
    {"label": "Alpha test", "value": fmt_pct(_alpha),
     "sub": f"vs B&H {fmt_pct(multi_row.get('bh'))}",
     "tone": "up" if (_alpha or 0) > 0 else "dn"},
    {"label": "Return agent", "value": fmt_pct(multi_row.get("return")),
     "sub": "full test 2021-22",
     "tone": "up" if (multi_row.get("return") or 0) > 0 else "dn"},
    {"label": "Max drawdown", "value": fmt_dd(multi_row.get("max_dd")),
     "sub": "kill-switch 25 %", "tone": "dn"},
    {"label": "Sharpe", "value": fmt_num(_sharpe),
     "sub": "annualisé √252", "tone": ""},
    {"label": "Cross-ticker",
     "value": f"{cross_positive}/{len(cross_df)}" if not cross_df.empty else "n/a",
     "sub": "alphas positifs",
     "tone": "up" if not cross_df.empty and cross_positive == len(cross_df) else ""},
    {"label": "Bruit inter-seeds", "value": fmt_pct(robust_multi.get("alpha_std"), signed=False),
     "sub": "±σ sur l'alpha (5 seeds)", "tone": ""},
])


tab_desk, tab_market, tab_strategy, tab_walk, tab_risk, tab_gen, tab_model, tab_data = st.tabs(
    [
        "Desk",
        "Marché",
        "Stratégie",
        "Walk-forward",
        "Risk lab",
        "Génératif",
        "Modèle live",
        "Données",
    ]
)


with tab_desk:
    left, right = st.columns([1.35, 1.0], gap="large")
    with left:
        section("Performance centrale", "Comparaison du PPO contre Buy & Hold sur le split test.")
        st.plotly_chart(make_overview_chart(model_df), use_container_width=True, key="desk_overview")
    with right:
        section("Allocation des décisions", "Lecture rapide du comportement appris par l'agent.")
        st.plotly_chart(make_action_chart(model_df), use_container_width=True, key="desk_actions")

    col_a, col_b, col_c = st.columns([1.1, 1.1, 0.8], gap="large")
    with col_a:
        section("Alpha par ticker", "Généralisation du modèle multi-actifs.")
        st.plotly_chart(make_cross_chart(cross_df), use_container_width=True, key="desk_cross_alpha")
    with col_b:
        section("Carte risque / rendement", "Taille des points proportionnelle au Sharpe.")
        st.plotly_chart(make_risk_return_chart(cross_df), use_container_width=True, key="desk_risk_return")
    with col_c:
        section("Audit express", "Points à regarder avant de vendre le résultat comme robuste.")
        st.markdown(
            f"""
            <div class="note-box">
                <div class="compact-label">Robustesse seeds</div>
                Alpha moyen multi: <b>{fmt_pct(robust_multi.get("alpha"))}</b><br>
                Std alpha: <b>{fmt_pct(robust_multi.get("alpha_std"), signed=False)}</b><br><br>
                <div class="compact-label">Walk-forward</div>
                Cellules positives: <b>{fmt_pct(walk_agg.get("pct_positive"), signed=False)}</b><br>
                Worst cell: <b>{fmt_pct(walk_agg.get("worst_cell_alpha"))}</b><br>
                Best cell: <b>{fmt_pct(walk_agg.get("best_cell_alpha"))}</b><br><br>
                <div class="compact-label">Conclusion desk</div>
                Bon alpha test, mais validation temporelle encore instable:
                c'est un prototype sérieux, pas une stratégie prête pour capital réel.
            </div>
            """,
            unsafe_allow_html=True,
        )


with tab_market:
    payload = st.session_state.get("market_payload")
    if not payload:
        section("Marché live", "Charge un actif depuis la sidebar pour inspecter prix, drawdown et features.")
        st.plotly_chart(
            empty_fig("Aucun marché chargé pour cette session.", height=360),
            use_container_width=True,
            key="market_empty",
        )
    else:
        train = payload["train"]
        val = payload["val"]
        test = payload["test"]
        ticker_loaded = payload["ticker"]
        data = pd.concat([train, val, test])
        stats = market_stats(data)

        section(f"{ticker_loaded} market monitor", "Prix, drawdown, volatilité et régimes heuristiques du dashboard.")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Return", fmt_pct(stats.get("total_return")), delta="période chargée")
        m2.metric("Vol ann.", fmt_pct(stats.get("ann_vol"), signed=False), delta="réalisée")
        m3.metric("Sharpe B&H", fmt_num(stats.get("sharpe")), delta="price-only")
        m4.metric("Max drawdown", fmt_dd(stats.get("max_dd")), delta="asset")
        m5.metric("CVaR 95", fmt_pct(stats.get("cvar_95")), delta="daily")
        st.plotly_chart(
            make_market_chart(train, val, test, ticker_loaded),
            use_container_width=True,
            key="market_price_regimes",
        )

        selected_features = st.multiselect(
            "Features normalisées à afficher",
            [feature for feature in FEATURES if feature in data.columns],
            default=[feature for feature in FEATURES if feature in data.columns][:4],
        )
        st.plotly_chart(
            make_feature_chart(data, selected_features),
            use_container_width=True,
            key="market_features",
        )

        col_corr, col_stats = st.columns([1.0, 1.0], gap="large")
        with col_corr:
            section("Corrélation train", "Matrice sur les observations normalisées.")
            feature_cols = [feature for feature in FEATURES if feature in train.columns]
            if feature_cols:
                corr = train[feature_cols].corr()
                fig_corr = go.Figure(
                    go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        zmin=-1,
                        zmax=1,
                        zmid=0,
                        colorscale=[[0, "#ff5a6e"], [0.5, "#111822"], [1, "#18c99f"]],
                        text=np.round(corr.values, 2),
                        texttemplate="%{text}",
                        colorbar=dict(title="rho"),
                    )
                )
                st.plotly_chart(
                    style_fig(fig_corr, height=420),
                    use_container_width=True,
                    key="market_corr",
                )
            else:
                st.plotly_chart(
                    empty_fig("Aucune feature disponible."),
                    use_container_width=True,
                    key="market_corr_empty",
                )
        with col_stats:
            section("Splits", "Statistiques price-only par segment temporel.")
            split_df = split_stats_frame({"Train": train, "Validation": val, "Test": test})
            split_display = percent_points(split_df, ["return", "ann_vol", "max_dd"])
            st.dataframe(
                split_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "return": st.column_config.NumberColumn("return", format="%.1f%%"),
                    "ann_vol": st.column_config.NumberColumn("ann_vol", format="%.1f%%"),
                    "max_dd": st.column_config.NumberColumn("max_dd", format="%.1f%%"),
                    "sharpe": st.column_config.NumberColumn("sharpe", format="%.2f"),
                },
            )


with tab_strategy:
    section("Scorecard modèles", "Single AAPL vs entraînement multi-actifs, même protocole full split.")
    if model_df.empty:
        st.warning("Aucun fichier reports/metrics.json exploitable.")
    else:
        scorecard = model_df[
            [
                "model",
                "return",
                "bh",
                "alpha",
                "max_dd",
                "sharpe",
                "sortino",
                "calmar",
                "cvar_95",
                "n_trades",
                "terminated_early",
            ]
        ]
        scorecard = percent_points(
            scorecard,
            ["return", "bh", "alpha", "max_dd", "cvar_95"],
            invert_abs=["max_dd"],
        )
        st.dataframe(
            scorecard,
            use_container_width=True,
            hide_index=True,
            column_config={
                "return": st.column_config.NumberColumn("return", format="%.1f%%"),
                "bh": st.column_config.NumberColumn("buy_hold", format="%.1f%%"),
                "alpha": st.column_config.NumberColumn("alpha", format="%.1f%%"),
                "max_dd": st.column_config.NumberColumn("max_dd", format="%.1f%%"),
                "sharpe": st.column_config.NumberColumn("sharpe", format="%.2f"),
                "sortino": st.column_config.NumberColumn("sortino", format="%.2f"),
                "calmar": st.column_config.NumberColumn("calmar", format="%.2f"),
                "cvar_95": st.column_config.NumberColumn("cvar_95", format="%.2f%%"),
            },
        )

    col_1, col_2 = st.columns(2, gap="large")
    with col_1:
        section("Cross-ticker alpha", "Le test utile contre l'overfit single-stock.")
        st.plotly_chart(make_cross_chart(cross_df), use_container_width=True, key="strategy_cross_alpha")
    with col_2:
        section("Robustesse par seed", "Moyenne et dispersion sur sous-fenêtres aléatoires.")
        if robust_df.empty:
            st.plotly_chart(
                empty_fig("Aucune donnée seed disponible."),
                use_container_width=True,
                key="strategy_seed_empty",
            )
        else:
            fig_seed = go.Figure()
            fig_seed.add_trace(
                go.Bar(
                    x=robust_df["model"],
                    y=robust_df["alpha"],
                    name="Alpha moyen",
                    marker_color="#18c99f",
                    error_y=dict(type="data", array=robust_df["alpha_std"], color="#dce6f7"),
                    text=[fmt_pct(v) for v in robust_df["alpha"]],
                    textposition="outside",
                )
            )
            fig_seed.add_trace(
                go.Scatter(
                    x=robust_df["model"],
                    y=robust_df["return"],
                    mode="markers",
                    name="Return moyen",
                    marker=dict(color="#4da3ff", size=12),
                )
            )
            fig_seed.update_yaxes(tickformat=".0%")
            st.plotly_chart(
                style_fig(fig_seed, height=390),
                use_container_width=True,
                key="strategy_seed_chart",
            )

    seed_detail = seed_robustness.get("per_seed", {})
    if seed_detail:
        section("Seed tape", "Alpha cross-ticker moyen par seed d'entraînement.")
        seed_rows = []
        for seed, values in seed_detail.items():
            seed_rows.append(
                {
                    "seed": seed,
                    "aapl_alpha": values.get("aapl_alpha"),
                    "cross_alpha_mean": values.get("cross_alpha_mean"),
                    "cross_positive": values.get("cross_positive"),
                }
            )
        seed_df = pd.DataFrame(seed_rows)
        seed_display = percent_points(seed_df, ["aapl_alpha", "cross_alpha_mean"])
        st.dataframe(
            seed_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "aapl_alpha": st.column_config.NumberColumn("aapl_alpha", format="%.1f%%"),
                "cross_alpha_mean": st.column_config.NumberColumn("cross_alpha_mean", format="%.1f%%"),
            },
        )


with tab_walk:
    section("Walk-forward matrix", "Alpha par année de test et par ticker.")
    w1, w2 = st.columns([1.1, 1.0], gap="large")
    with w1:
        st.plotly_chart(
            make_walk_forward_heatmap(wf_df),
            use_container_width=True,
            key="walk_heatmap",
        )
    with w2:
        st.plotly_chart(
            make_walk_forward_year_chart(walk_forward),
            use_container_width=True,
            key="walk_years",
        )

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Mean alpha", fmt_pct(walk_agg.get("mean_alpha")))
    a2.metric("Median alpha", fmt_pct(walk_agg.get("median_alpha")))
    a3.metric("Positive cells", fmt_pct(walk_agg.get("pct_positive"), signed=False))
    a4.metric("Worst cell", fmt_pct(walk_agg.get("worst_cell_alpha")))
    a5.metric("Best cell", fmt_pct(walk_agg.get("best_cell_alpha")))

    if not wf_df.empty:
        section("Cellules détaillées", "Toutes les observations du walk-forward.")
        wf_display = percent_points(
            wf_df.sort_values(["year", "ticker"]),
            ["return", "bh", "alpha", "max_dd"],
            invert_abs=["max_dd"],
        )
        st.dataframe(
            wf_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "return": st.column_config.NumberColumn("return", format="%.1f%%"),
                "bh": st.column_config.NumberColumn("buy_hold", format="%.1f%%"),
                "alpha": st.column_config.NumberColumn("alpha", format="%.1f%%"),
                "sharpe": st.column_config.NumberColumn("sharpe", format="%.2f"),
                "max_dd": st.column_config.NumberColumn("max_dd", format="%.1f%%"),
            },
        )


with tab_risk:
    r1, r2 = st.columns(2, gap="large")
    with r1:
        section("Sensibilité aux frais", "Alpha après frais de transaction croissants.")
        st.plotly_chart(make_fee_chart(fees_df), use_container_width=True, key="risk_fees")
    with r2:
        section("Kill-switch drawdown", "Contribution du stop de risque au résultat final.")
        st.plotly_chart(
            make_kill_switch_chart(stress_tests),
            use_container_width=True,
            key="risk_kill_switch",
        )

    no_stop = stress_tests.get("no_killswitch", {})
    if no_stop:
        section("Risk ledger", "Chiffres bruts du stress test sans stop.")
        rows = []
        for model, values in no_stop.items():
            rows.append({"model": model, **values})
        risk_display = percent_points(
            pd.DataFrame(rows),
            ["return", "alpha", "max_dd", "alpha_with_stop", "stop_contribution"],
            invert_abs=["max_dd"],
        )
        st.dataframe(
            risk_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "return": st.column_config.NumberColumn("return", format="%.1f%%"),
                "alpha": st.column_config.NumberColumn("alpha", format="%.1f%%"),
                "max_dd": st.column_config.NumberColumn("max_dd", format="%.1f%%"),
                "alpha_with_stop": st.column_config.NumberColumn("alpha_with_stop", format="%.1f%%"),
                "stop_contribution": st.column_config.NumberColumn("stop_contribution", format="%.1f%%"),
            },
        )

    figure_candidates = [
        FIGURES_DIR / "fig_equity_test.png",
        FIGURES_DIR / "fig_drawdown_test.png",
    ]
    existing_figures = [path for path in figure_candidates if path.exists()]
    if existing_figures:
        section("Figures rapport", "Snapshots générés par le pipeline de reporting.")
        cols = st.columns(len(existing_figures))
        for col, path in zip(cols, existing_figures):
            with col:
                st.image(str(path), use_column_width=True)


with tab_gen:
    dv = diffusion_val
    ddpm_res = dv.get("ddpm") if isinstance(dv, dict) else None
    if not ddpm_res:
        section("Générateur de scénarios (DDPM)",
                "reports/diffusion_validation.json absent ou sans verdict — "
                "lancer validate_diffusion.py.")
        st.plotly_chart(empty_fig("Aucun verdict de génération disponible.", height=360),
                        use_container_width=True, key="gen_empty")
    else:
        verdict = ddpm_res.get("verdict", {})
        crit_labels = {
            "E2_queues": "É2 queues",
            "E3_acf_parasite": "É3 ACF parasite",
            "E4_clustering": "É4 clustering",
            "E5_discriminatif": "É5 discriminatif",
            "E6_memorisation": "É6 mémorisation",
        }
        protocole_ok = dv.get("protocole_E1", {}).get("ok", None)
        all_pass = verdict.get("all_pass", False)

        section(
            "Générateur de scénarios (DDPM sur rendements)",
            "Jugé par le protocole PRÉ-ENREGISTRÉ : bandes calibrées sur le réel "
            "et seuils figés avant tout entraînement — le verdict ci-dessous ne "
            "peut pas être négocié après coup.",
        )

        banner_cls = "go" if all_pass else "nogo"
        banner_txt = ("GO — échantillons validés, branchement RL Phase 2 autorisé"
                      if all_pass else
                      "NO-GO — pas d'entraînement RL sur ces échantillons "
                      "(2 itérations : ε-prediction puis v-prediction)")
        st.markdown(f'<div class="verdict-banner {banner_cls}">{banner_txt}</div>',
                    unsafe_allow_html=True)

        badges = []
        if protocole_ok is not None:
            badges.append(
                f'<span class="badge {"pass" if protocole_ok else "fail"}">'
                f'É1 protocole {"✓" if protocole_ok else "✗"}</span>')
        for key, label in crit_labels.items():
            if key in verdict:
                ok = verdict[key].get("pass", False)
                badges.append(
                    f'<span class="badge {"pass" if ok else "fail"}">'
                    f'{label} {"✓" if ok else "✗"}</span>')
        st.markdown(f'<div class="badge-row">{"".join(badges)}</div>',
                    unsafe_allow_html=True)

        moments = ddpm_res.get("summary", {}).get("moments", {})
        real_moments = dv.get("real", {}).get("summary", {}).get("moments", {})
        e2 = verdict.get("E2_queues", {})
        e4 = verdict.get("E4_clustering", {})
        e5 = verdict.get("E5_discriminatif", {})
        e6 = verdict.get("E6_memorisation", {})
        kpi_row([
            {"label": "σ générée / réelle",
             "value": f'{moments.get("std", float("nan")):.4f}',
             "sub": f'réel {real_moments.get("std", float("nan")):.4f} '
                    f'(×{(moments.get("std", 0) / max(real_moments.get("std", 1e-9), 1e-9)):.2f})',
             "tone": "dn" if not e2.get("pass") else "up"},
            {"label": "Kurtosis excès",
             "value": f'{moments.get("kurtosis_excess", float("nan")):.1f}',
             "sub": f'bande [{e2.get("band", [0, 0])[0]:.1f} ; {e2.get("band", [0, 0])[1]:.1f}]',
             "tone": "up" if e2.get("pass") else "dn"},
            {"label": "ACF|r| lag 1",
             "value": f'{e4.get("lag1", float("nan")):.3f}',
             "sub": f'bande [{e4.get("lag1_band", [0, 0])[0]:.3f} ; {e4.get("lag1_band", [0, 0])[1]:.3f}]',
             "tone": "up" if e4.get("pass") else "dn"},
            {"label": "Juge discriminatif",
             "value": f'{ddpm_res.get("disc", {}).get("acc", float("nan")):.3f}',
             "sub": f'seuil ≤ {e5.get("threshold", float("nan")):.3f} (GARCH+5pts)',
             "tone": "up" if e5.get("pass") else "dn"},
            {"label": "Distance NN médiane",
             "value": f'{ddpm_res.get("nn_median", float("nan")):.2f}',
             "sub": f'seuil ≥ {e6.get("threshold", float("nan")):.2f} (p10 réel)',
             "tone": "up" if e6.get("pass") else "dn"},
        ])

        fig_dir = FIGURES_DIR
        img_specs = [
            ("diffusion_acf.png",
             "ACF des rendements (É3) et de |r| (É4) — le clustering est capturé."),
            ("diffusion_window_stats.png",
             "Ce que l'env RL « verrait » d'une fenêtre : vol, max drawdown, terminal."),
            ("diffusion_trajectories.png",
             "30 trajectoires cumulées par générateur — l'œil confirme les chiffres."),
        ]
        for name, caption in img_specs:
            path = fig_dir / name
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)

        st.markdown(
            """
            <div class="note-box">
                <div class="compact-label">Lecture desk</div>
                Le volatility clustering (É4) et l'anti-mémorisation (É6) passent —
                la structure temporelle est apprise. L'échec porte sur
                l'<b>échelle</b> : sous-dispersion ×0.56 du mélange multi-tickers
                (queue de vol ×0.43), identique sous ε- et v-prediction ; le sampler
                a été disculpé par test-oracle analytique. Verdict v1 archivé
                (<span class="num">diffusion_validation_v1.json</span>) ; levier v3
                identifié : normalisation par ticker. Récit complet : rapport, Acte 4.
            </div>
            """,
            unsafe_allow_html=True,
        )


with tab_model:
    payload = st.session_state.get("market_payload")
    section("Évaluation locale PPO", "Charge un modèle entraîné et rejoue la politique sur le split test chargé.")

    if not DRL_AVAILABLE:
        st.warning(f"Évaluation live indisponible: {DRL_ERROR}")
    elif not payload:
        st.info("Charge d'abord un marché depuis la sidebar pour créer le split test.")
    else:
        model_paths = {
            "Single AAPL": ROOT / "models" / "ppo_single" / "best_model.zip",
            "Multi 5 tickers": ROOT / "models" / "ppo_multi" / "best_model.zip",
        }
        existing_models = {name: path for name, path in model_paths.items() if path.exists()}

        source = st.radio("Source modèle", ["Modèle local", "Upload"], horizontal=True)
        local_path = None
        uploaded_model = None
        uploaded_vecnorm = None
        if source == "Modèle local":
            if existing_models:
                selected_model = st.selectbox("Modèle", list(existing_models.keys()))
                local_path = existing_models[selected_model]
                st.caption("VecNormalize est lu automatiquement s'il existe dans le même dossier.")
            else:
                st.warning("Aucun best_model.zip local trouvé dans models/ppo_single ou models/ppo_multi.")
        else:
            uploaded_model = st.file_uploader("best_model.zip", type=["zip"])
            uploaded_vecnorm = st.file_uploader("vec_normalize.pkl", type=["pkl"])

        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        with cfg_col1:
            initial_capital = st.number_input("Capital initial", value=10_000.0, step=1_000.0)
        with cfg_col2:
            fee_bps = st.number_input("Frais bps", value=10.0, step=5.0)
        with cfg_col3:
            window_size = st.number_input("Window", min_value=5, max_value=30, value=10, step=1)
        with cfg_col4:
            deterministic = st.toggle("Full split", value=True)

        runs = 1 if deterministic else st.slider("Runs aléatoires", 2, 10, 5)
        ready = local_path is not None or uploaded_model is not None
        if st.button("Évaluer", type="primary", use_container_width=True, disabled=not ready):
            cfg_env = EnvConfig(
                initial_capital=float(initial_capital),
                transaction_cost=float(fee_bps) / 10_000,
                window_size=int(window_size),
                max_drawdown_pct=0.25,
                reward_scaling=100.0,
            )
            test_data = payload["test"]
            try:
                results = []
                with st.spinner("Évaluation du modèle..."):
                    if local_path is not None:
                        for seed in range(runs):
                            results.append(
                                run_drl_episode(local_path, test_data, cfg_env, deterministic, seed)
                            )
                    else:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            model_path = Path(tmpdir) / "best_model.zip"
                            with model_path.open("wb") as fh:
                                fh.write(uploaded_model.read())
                            if uploaded_vecnorm is not None:
                                vec_path = Path(tmpdir) / "vec_normalize.pkl"
                                with vec_path.open("wb") as fh:
                                    fh.write(uploaded_vecnorm.read())
                            for seed in range(runs):
                                results.append(
                                    run_drl_episode(model_path, test_data, cfg_env, deterministic, seed)
                                )
                st.session_state.live_eval = results
                st.success("Évaluation terminée.")
            except Exception as exc:
                st.error(f"Évaluation impossible: {exc}")

        results = st.session_state.get("live_eval")
        if results:
            returns = np.array([item["return"] for item in results], dtype=float)
            alphas = np.array([item["alpha"] for item in results], dtype=float)
            sharpes = np.array([item["sharpe"] for item in results], dtype=float)
            drawdowns = np.array([item["max_dd"] for item in results], dtype=float)
            trades = np.array([item["n_trades"] for item in results], dtype=float)

            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("Return agent", fmt_pct(np.nanmean(returns)))
            e2.metric("Alpha", fmt_pct(np.nanmean(alphas)))
            e3.metric("Sharpe", fmt_num(np.nanmean(sharpes)))
            e4.metric("Max DD", fmt_dd(np.nanmean(drawdowns)))
            e5.metric("Trades", f"{np.nanmean(trades):.0f}")

            first = results[0]
            portfolio = first["portfolio"]
            prices = first["prices"]
            bh_values = prices / prices[0] * portfolio[0] if len(prices) else []
            fig_live = go.Figure()
            fig_live.add_trace(
                go.Scatter(
                    y=portfolio,
                    mode="lines",
                    name="Agent",
                    line=dict(color="#18c99f", width=2.5),
                )
            )
            fig_live.add_trace(
                go.Scatter(
                    y=bh_values,
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(color="#4da3ff", width=2, dash="dash"),
                )
            )
            fig_live.add_hline(y=portfolio[0], line_dash="dot", line_color="#91a0b8")
            st.plotly_chart(
                style_fig(fig_live, height=430),
                use_container_width=True,
                key="live_equity",
            )

            action_map = {0: "Hold", 1: "Long", 2: "Flat", 3: "Short"}
            action_counts = pd.Series(first["actions"]).map(action_map).value_counts()
            fig_actions = go.Figure(
                go.Bar(
                    x=action_counts.index,
                    y=action_counts.values,
                    marker_color=[ACTION_COLORS.get(action, "#91a0b8") for action in action_counts.index],
                    text=action_counts.values,
                    textposition="outside",
                )
            )
            fig_actions.update_yaxes(title="Actions")
            st.plotly_chart(
                style_fig(fig_actions, height=320),
                use_container_width=True,
                key="live_actions",
            )


with tab_data:
    payload = st.session_state.get("market_payload")
    section("Données de session", "Export et inspection du dataset actuellement chargé.")
    if not payload:
        st.plotly_chart(
            empty_fig("Aucune donnée chargée."),
            use_container_width=True,
            key="data_empty",
        )
    else:
        train = payload["train"]
        val = payload["val"]
        test = payload["test"]
        ticker_loaded = payload["ticker"]
        data = pd.concat([train, val, test])
        split_name = st.radio(
            "Split",
            ["Train", "Validation", "Test", "Tout"],
            index=3,
            horizontal=True,
        )
        split_map = {"Train": train, "Validation": val, "Test": test, "Tout": data}
        df_view = split_map[split_name]

        d1, d2, d3 = st.columns(3)
        d1.metric("Rows", f"{len(df_view):,}")
        d2.metric("Début", df_view.index[0].date().isoformat())
        d3.metric("Fin", df_view.index[-1].date().isoformat())

        cols = ["price"] + [feature for feature in FEATURES if feature in df_view.columns]
        st.dataframe(df_view[cols].round(5), use_container_width=True, height=460)
        st.download_button(
            "Télécharger CSV",
            data=df_view.to_csv().encode("utf-8"),
            file_name=f"{ticker_loaded}_{split_name.lower()}.csv",
            mime="text/csv",
            use_container_width=True,
        )


st.markdown("---")
st.caption(
    "DRL Portfolio - Martin Chassaing - usage pédagogique uniquement, pas un conseil financier."
)
