# dashboard.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os

# streamlit-extras
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stoggle import stoggle

sys.path.append(os.path.dirname(__file__))
from feature_engineering import load_data, DataConfig, FEATURES


# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title  = "DRL Portfolio",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric label { color: #8892b0 !important; font-size: 0.85rem !important; }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.shields.io/badge/DRL-Portfolio-blue?style=for-the-badge")
    add_vertical_space(1)

    colored_header(
        label       = "Configuration",
        description = "Paramètres du pipeline",
        color_name  = "green-70"
    )

    ticker = st.selectbox(
        "📌 Actif",
        ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD", "SPY", "TSLA"],
        index=0
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Début", value=pd.to_datetime("2018-01-01"))
    with col2:
        end_date   = st.date_input("📅 Fin",   value=pd.to_datetime("2024-01-01"))

    add_vertical_space(1)

    colored_header(
        label       = "Modèle",
        description = "Hyperparamètres des features",
        color_name  = "blue-70"
    )

    vol_window  = st.slider("Fenêtre Volatilité", 5,  50,  20)
    rsi_window  = st.slider("Fenêtre RSI",        5,  30,  14)
    sma_window  = st.slider("Fenêtre SMA",        10, 100, 50)
    max_regimes = st.slider("Max Régimes GMM",    2,  6,   5)

    add_vertical_space(1)
    load_btn = st.button("🚀 Charger les données", use_container_width=True)
    add_vertical_space(1)

    # Explications cachables dans la sidebar
    stoggle("💡 C'est quoi les régimes GMM ?",
        """
        Le **Gaussian Mixture Model** détecte automatiquement
        les régimes de marché (Bull / Bear / Sideways) en
        clusterisant les log-returns et la volatilité.
        Le nombre optimal de régimes est choisi par **BIC**.
        """
    )
    stoggle("🔒 Pourquoi No Data Leakage ?",
        """
        Le GMM et le RobustScaler sont **fit uniquement sur
        le train set**, puis appliqués sur val/test.
        Sinon l'agent "verrait le futur" pendant l'entraînement.
        """
    )

    add_vertical_space(1)
    st.markdown("""
    <small style='color:#8892b0;'>
    Développé par <b>Martin Chassaing</b><br>
    IMT Atlantique × Paris Dauphine
    </small>
    """, unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
colored_header(
    label       = "DRL Portfolio Dashboard",
    description = "Deep Reinforcement Learning × Quantitative Finance",
    color_name  = "green-70"
)
add_vertical_space(1)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
@st.cache_data(show_spinner=False)
def get_data(ticker, start, end, vol_w, rsi_w, sma_w, max_r):
    cfg = DataConfig(
        ticker      = ticker,
        start_date  = str(start),
        end_date    = str(end),
        vol_window  = vol_w,
        rsi_window  = rsi_w,
        sma_window  = sma_w,
        max_regimes = max_r
    )
    return load_data(cfg)


# ============================================================
# ÉTAT SESSION
# ============================================================
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if load_btn:
    with st.spinner(f"⏳ Chargement de {ticker}..."):
        try:
            train, val, test, scaler, gmm = get_data(
                ticker, start_date, end_date,
                vol_window, rsi_window, sma_window, max_regimes
            )
            st.session_state.train       = train
            st.session_state.val         = val
            st.session_state.test        = test
            st.session_state.ticker      = ticker
            st.session_state.data_loaded = True
            st.success(f"✅ {ticker} chargé avec succès !")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")


# ============================================================
# MAIN CONTENT
# ============================================================
if not st.session_state.data_loaded:

    # Page d'accueil
    st.markdown("""
    <div style='text-align:center; padding:60px 0; color:#8892b0;'>
        <h2>👈 Configure et charge un actif dans la sidebar</h2>
        <p>Le pipeline complet sera exécuté : Download → Features → GMM → Split</p>
    </div>
    """, unsafe_allow_html=True)

    add_vertical_space(1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("📊 **5 Features Core**\nlog_returns, volatility, RSI, dist_to_SMA, GMM regime")
    with c2:
        st.info("🧠 **GMM Auto**\nSélection du nombre optimal de régimes par BIC")
    with c3:
        st.info("✂️ **Split 70/15/15**\nTrain / Validation / Test strict")
    with c4:
        st.info("🔒 **No Data Leakage**\nScaler et GMM fit sur train uniquement")

else:
    train  = st.session_state.train
    val    = st.session_state.val
    test   = st.session_state.test
    ticker = st.session_state.ticker
    data   = pd.concat([train, val, test])

    REGIME_COLORS = {
        0: '#2ecc71',
        1: '#e74c3c',
        2: '#3498db',
        3: '#f39c12',
        4: '#9b59b6',
    }

    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Prix & Régimes",
        "🔬 Features",
        "📊 Statistiques",
        "🗂️ Data"
    ])


    # ============================================================
    # TAB 1 : PRIX & RÉGIMES
    # ============================================================
    with tab1:

        colored_header(
            label       = f"{ticker} — Vue Générale",
            description = "Performance & régimes de marché détectés par GMM",
            color_name  = "green-70"
        )

        # --- KPIs ---
        total_return = (data['price'].iloc[-1] / data['price'].iloc[0] - 1) * 100
        log_ret      = data['log_returns']
        sharpe       = (log_ret.mean() / log_ret.std()) * np.sqrt(252)
        peak         = data['price'].cummax()
        max_dd       = ((data['price'] - peak) / peak).min() * 100
        ann_vol      = log_ret.std() * np.sqrt(252) * 100
        n_regimes    = data['market_regime'].nunique()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("💰 Return Total",   f"{total_return:+.1f}%")
        k2.metric("📐 Sharpe Ratio",   f"{sharpe:.2f}")
        k3.metric("📉 Max Drawdown",   f"{max_dd:.1f}%")
        k4.metric("🌊 Vol Annualisée", f"{ann_vol:.1f}%")
        k5.metric("🧠 Régimes GMM",    f"{n_regimes}")

        # Style des metric cards
        style_metric_cards(
            background_color  = "#1e2130",
            border_left_color = "#64ffda",
            border_color      = "#3d4463",
            box_shadow        = True
        )

        add_vertical_space(1)

        # Explications KPIs
        stoggle("💡 Comment lire ces métriques ?",
            """
            - **Return Total** : performance brute sur la période
            - **Sharpe Ratio** : rendement ajusté par le risque (>1 = bon, >2 = excellent)
            - **Max Drawdown** : perte maximale depuis un pic (plus proche de 0 = mieux)
            - **Vol Annualisée** : agitation du marché sur 1 an
            - **Régimes GMM** : nombre d'états de marché détectés automatiquement
            """
        )

        add_vertical_space(1)

        # --- Chart Prix + Régimes ---
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.05,
            subplot_titles=(
                f"{ticker} — Prix & Régimes de Marché",
                "Volatilité Rolling",
                "Distribution des Régimes"
            )
        )

        regime_names = {
            0: 'Bull', 1: 'Bear', 2: 'Sideways',
            3: 'Régime 3', 4: 'Régime 4'
        }

        for regime in sorted(data['market_regime'].unique()):
            mask = data['market_regime'] == regime
            fig.add_trace(
                go.Scatter(
                    x=data.index[mask],
                    y=data['price'][mask],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=REGIME_COLORS.get(regime, 'white'),
                        opacity=0.8
                    ),
                    name=f"Régime {regime} ({regime_names.get(regime, '')})",
                    legendgroup=f"regime_{regime}"
                ),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['price'],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.15)', width=1),
                showlegend=False, name='Prix'
            ),
            row=1, col=1
        )

        for split_date, label, color in [
            (train.index[-1], "Train | Val", "rgba(255,255,255,0.5)"),
            (val.index[-1],   "Val | Test",  "rgba(255,165,0,0.5)"),
        ]:
            for row in [1, 2, 3]:
                fig.add_vline(
                    x=split_date, line_dash="dash",
                    line_color=color, line_width=1.5,
                    row=row, col=1
                )
            fig.add_annotation(
                x=split_date, y=data['price'].max(),
                text=label, showarrow=False,
                font=dict(color=color, size=10),
                yshift=10, row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['volatility'],
                fill='tozeroy',
                fillcolor='rgba(230,126,34,0.2)',
                line=dict(color='#e67e22', width=1.5),
                name='Volatilité', showlegend=False
            ),
            row=2, col=1
        )

        regime_counts = data['market_regime'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f"Régime {r}" for r in regime_counts.index],
                y=regime_counts.values,
                marker_color=[
                    REGIME_COLORS.get(r, 'gray') for r in regime_counts.index
                ],
                showlegend=False, name='Distribution'
            ),
            row=3, col=1
        )

        fig.update_layout(
            height=700,
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            legend=dict(
                orientation='h', yanchor='bottom',
                y=1.02, xanchor='right', x=1
            ),
            margin=dict(l=50, r=30, t=60, b=30),
        )
        fig.update_xaxes(showgrid=True, gridcolor='#2d3250', gridwidth=0.5)
        fig.update_yaxes(showgrid=True, gridcolor='#2d3250', gridwidth=0.5)

        st.plotly_chart(fig, use_container_width=True)


    # ============================================================
    # TAB 2 : FEATURES
    # ============================================================
    with tab2:

        colored_header(
            label       = "Analyse des Features",
            description = "Visualisation et corrélation des 5 features core",
            color_name  = "blue-70"
        )

        stoggle("💡 À quoi servent ces features ?",
            """
            - **log_returns** : rendement journalier (hypothèse GBM : suit une loi Normale)
            - **volatility** : proxy GARCH, mesure l'agitation du marché sur 20j
            - **rsi** : momentum — surachat (>0.7) ou survente (<0.3)
            - **dist_to_sma** : signal Ornstein-Uhlenbeck, mesure l'écart à la moyenne
            - **market_regime** : état latent détecté par GMM (Bull/Bear/Sideways)
            """
        )

        add_vertical_space(1)

        selected_features = st.multiselect(
            "Features à afficher",
            options=FEATURES,
            default=['log_returns', 'volatility', 'rsi', 'dist_to_sma']
        )

        if selected_features:
            fig2 = make_subplots(
                rows=len(selected_features), cols=1,
                shared_xaxes=True,
                subplot_titles=selected_features,
                vertical_spacing=0.06
            )

            feature_colors = {
                'log_returns'  : '#2ecc71',
                'volatility'   : '#e67e22',
                'rsi'          : '#9b59b6',
                'dist_to_sma'  : '#1abc9c',
                'market_regime': '#3498db',
            }

            for i, feat in enumerate(selected_features, 1):

                if feat == 'rsi':
                    fig2.add_hrect(
                        y0=0.7, y1=1.0,
                        fillcolor="rgba(231,76,60,0.1)",
                        line_width=0, row=i, col=1
                    )
                    fig2.add_hrect(
                        y0=0.0, y1=0.3,
                        fillcolor="rgba(46,204,113,0.1)",
                        line_width=0, row=i, col=1
                    )
                    fig2.add_hline(
                        y=0.7, line_dash="dot",
                        line_color="rgba(231,76,60,0.5)",
                        row=i, col=1
                    )
                    fig2.add_hline(
                        y=0.3, line_dash="dot",
                        line_color="rgba(46,204,113,0.5)",
                        row=i, col=1
                    )

                if feat in ['dist_to_sma', 'log_returns']:
                    fig2.add_hline(
                        y=0,
                        line_color='rgba(255,255,255,0.2)',
                        line_width=1, row=i, col=1
                    )

                if feat == 'market_regime':
                    fig2.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data[feat],
                            marker_color=[
                                REGIME_COLORS.get(int(v), 'gray')
                                for v in data[feat]
                            ],
                            showlegend=False, name=feat
                        ),
                        row=i, col=1
                    )
                else:
                    fig2.add_trace(
                        go.Scatter(
                            x=data.index, y=data[feat],
                            mode='lines',
                            line=dict(
                                color=feature_colors.get(feat, 'white'),
                                width=1.2
                            ),
                            fill='tozeroy' if feat == 'volatility' else None,
                            fillcolor='rgba(230,126,34,0.1)',
                            showlegend=False, name=feat
                        ),
                        row=i, col=1
                    )

                for split_date in [train.index[-1], val.index[-1]]:
                    fig2.add_vline(
                        x=split_date, line_dash="dash",
                        line_color="rgba(255,255,255,0.2)",
                        row=i, col=1
                    )

            fig2.update_layout(
                height=250 * len(selected_features),
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                margin=dict(l=50, r=30, t=40, b=30),
                showlegend=False
            )
            fig2.update_xaxes(showgrid=True, gridcolor='#2d3250')
            fig2.update_yaxes(showgrid=True, gridcolor='#2d3250')

            st.plotly_chart(fig2, use_container_width=True)

        add_vertical_space(1)

        colored_header(
            label       = "Matrice de Corrélation",
            description = "Calculée sur le Train Set uniquement",
            color_name  = "orange-70"
        )

        stoggle("⚠️ Corrélations détectées dans ton projet",
            """
            - **volatility ↔ market_regime** : ~0.83 → le GMM est très influencé par la vol
            - **rsi ↔ dist_to_sma** : ~0.89 → les deux mesurent un écart à la moyenne

            Ce n'est pas forcément un problème pour le DRL (l'agent peut gérer
            la redondance), mais garde-le en tête si les performances sont décevantes.
            """
        )

        corr = train[FEATURES].corr()

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale='RdYlGn',
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorbar=dict(title="Corrélation")
        ))
        fig_corr.update_layout(
            height=450,
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)


    # ============================================================
    # TAB 3 : STATISTIQUES
    # ============================================================
    with tab3:

        colored_header(
            label       = "Statistiques Descriptives",
            description = "Distribution des rendements & analyse par régime",
            color_name  = "violet-70"
        )

        stoggle("💡 Pourquoi les log-returns ne suivent pas la Normale ?",
            """
            En finance réelle, les rendements ont des **queues épaisses** (fat tails) :
            les événements extrêmes (crashes, rallyes) sont bien plus fréquents
            que ce que prédit une loi Normale. C'est le problème fondamental de
            Black-Scholes que le DRL essaie de contourner en étant **model-free**.
            """
        )

        add_vertical_space(1)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Distribution des Log-Returns")

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=data['log_returns'],
                nbinsx=100,
                histnorm='probability density',
                marker_color='#2ecc71',
                opacity=0.7,
                name='Log Returns'
            ))

            mu  = data['log_returns'].mean()
            sig = data['log_returns'].std()
            x_range    = np.linspace(data['log_returns'].min(),
                                     data['log_returns'].max(), 300)
            normal_pdf = (
                1 / (sig * np.sqrt(2 * np.pi))
            ) * np.exp(-0.5 * ((x_range - mu) / sig) ** 2)

            fig_hist.add_trace(go.Scatter(
                x=x_range, y=normal_pdf,
                mode='lines',
                line=dict(color='#e74c3c', width=2),
                name='Normale théorique'
            ))
            fig_hist.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                height=350,
                legend=dict(orientation='h', y=1.02),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.markdown("##### Distribution par Régime")

            fig_regime = go.Figure()
            for regime in sorted(data['market_regime'].unique()):
                subset = data[data['market_regime'] == regime]['log_returns']
                fig_regime.add_trace(go.Violin(
                    y=subset,
                    name=f"Régime {regime}",
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=REGIME_COLORS.get(regime, 'gray'),
                    opacity=0.7,
                    line_color='white'
                ))
            fig_regime.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title='Log Returns'
            )
            st.plotly_chart(fig_regime, use_container_width=True)

        add_vertical_space(1)

        colored_header(
            label       = "Stats par Split",
            description = "Train / Validation / Test",
            color_name  = "green-70"
        )

        splits     = {'Train': train, 'Validation': val, 'Test': test}
        stats_rows = []
        for name, df_split in splits.items():
            lr = df_split['log_returns']
            stats_rows.append({
                'Split'           : name,
                'Jours'           : len(df_split),
                'Return Moyen'    : f"{lr.mean() * 252:.2%}",
                'Volatilité Ann.' : f"{lr.std() * np.sqrt(252):.2%}",
                'Sharpe'          : f"{(lr.mean() / lr.std()) * np.sqrt(252):.2f}",
                'Min Return'      : f"{lr.min():.3%}",
                'Max Return'      : f"{lr.max():.3%}",
            })

        st.dataframe(
            pd.DataFrame(stats_rows).set_index('Split'),
            use_container_width=True
        )


    # ============================================================
    # TAB 4 : DATA
    # ============================================================
    with tab4:

        colored_header(
            label       = "Données Brutes",
            description = "Exploration et export des données",
            color_name  = "orange-70"
        )

        split_view = st.radio(
            "Split à afficher",
            ["Train", "Validation", "Test", "Tout"],
            horizontal=True
        )

        split_map = {
            "Train"      : train,
            "Validation" : val,
            "Test"       : test,
            "Tout"       : data
        }
        df_view = split_map[split_view]

        # Métriques rapides du split sélectionné
        m1, m2, m3 = st.columns(3)
        m1.metric("📅 Nombre de jours", len(df_view))
        m2.metric("📌 Début",  str(df_view.index[0].date()))
        m3.metric("📌 Fin",    str(df_view.index[-1].date()))
        style_metric_cards(
            background_color  = "#1e2130",
            border_left_color = "#3498db",
            border_color      = "#3d4463",
            box_shadow        = True
        )

        add_vertical_space(1)

        st.dataframe(
            df_view[['price'] + FEATURES].round(4),
            use_container_width=True,
            height=400
        )

        csv = df_view.to_csv().encode('utf-8')
        st.download_button(
            label     = "⬇️ Télécharger CSV",
            data      = csv,
            file_name = f"{ticker}_{split_view.lower()}.csv",
            mime      = "text/csv"
        )


# ============================================================
# FOOTER
# ============================================================
add_vertical_space(2)
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8892b0; font-size:0.8rem; padding:10px;'>
    DRL Portfolio • Martin Chassaing • IMT Atlantique × Paris Dauphine<br>
    <em>For educational purposes only — Not financial advice</em>
</div>
""", unsafe_allow_html=True)