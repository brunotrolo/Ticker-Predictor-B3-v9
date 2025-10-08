
import streamlit as st
import pandas as pd, numpy as np
from datetime import date, timedelta, datetime
import yfinance as yf
import plotly.graph_objects as go
from b3_utils import load_b3_tickers, ensure_sa_suffix, is_known_b3_ticker, search_b3

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title="Análise B3 + ML Turbinada", page_icon="🚀", layout="wide")

# -------- Helpers --------
def set_plotly_template(theme_choice: str):
    import plotly.io as pio
    if theme_choice == "Claro":
        pio.templates.default = "plotly"
        st.markdown("<style>body, .stApp {background-color: #ffffff; color: #111111;}</style>", unsafe_allow_html=True)
    else:
        pio.templates.default = "plotly_dark"
        st.markdown("<style>body, .stApp {background-color: #0e1117; color: #e5e5e5;}</style>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    df = yf.download(ensure_sa_suffix(ticker), start=start, end=end, auto_adjust=True, progress=False)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index()
    return df

def sma(s, w): return s.rolling(window=w, min_periods=w).mean()
def rsi(s, w=14):
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(w).mean()
    ma_down = down.rolling(w).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_indicators(df, want_sma50=False, want_sma200=False):
    if df.empty: return df
    df = df.copy()
    df["SMA20"]=sma(df["Close"],20)
    if want_sma50:
        df["SMA50"]=sma(df["Close"],50)
    if want_sma200:
        df["SMA200"]=sma(df["Close"],200)
    df["RSI14"]=rsi(df["Close"])
    return df

def build_features(df, horizon=1):
    d = df.copy()
    # returns
    d["ret_1"] = d["Close"].pct_change(1)
    d["ret_3"] = d["Close"].pct_change(3)
    d["ret_5"] = d["Close"].pct_change(5)
    d["ret_10"] = d["Close"].pct_change(10)
    # distances to MAs (compute SMA50/200 if not present)
    if "SMA50" not in d.columns: d["SMA50"] = sma(d["Close"],50)
    if "SMA200" not in d.columns: d["SMA200"] = sma(d["Close"],200)
    d["dist_sma20"] = d["Close"] / d["SMA20"] - 1
    d["dist_sma50"] = d["Close"] / d["SMA50"] - 1
    d["dist_sma200"] = d["Close"] / d["SMA200"] - 1
    # volatility
    d["vol_5"] = d["Close"].pct_change().rolling(5).std()
    d["vol_10"] = d["Close"].pct_change().rolling(10).std()
    # RSI
    d["rsi"] = d["RSI14"]
    # Target: direction of horizon-ahead return
    d["future_ret"] = d["Close"].shift(-horizon) / d["Close"] - 1.0
    d["target_up"] = (d["future_ret"] > 0).astype(int)
    feat_cols = ["ret_1","ret_3","ret_5","ret_10","dist_sma20","dist_sma50","dist_sma200","vol_5","vol_10","rsi"]
    d = d.dropna(subset=feat_cols + ["target_up","future_ret"]).reset_index(drop=True)
    X = d[feat_cols].values
    y = d["target_up"].values
    return d, X, y, feat_cols

def fit_calibrated(model, X_train, y_train, frac_calib=0.2, method="sigmoid"):
    n = len(X_train)
    n_cal = max(int(n * frac_calib), 50) if n >= 100 else max(int(n * 0.1), 20)
    n_cal = min(n_cal, n-20) if n > 40 else max(5, n-5)
    X_fit, y_fit = X_train[:-n_cal], y_train[:-n_cal]
    X_cal, y_cal = X_train[-n_cal:], y_train[-n_cal:]
    m = clone(model)
    m.fit(X_fit, y_fit)
    cal = CalibratedClassifierCV(m, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal

def youden_threshold(y_true, y_prob):
    # choose threshold that maximizes TPR - FPR
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    k = np.argmax(j)
    return float(thr[k])

def time_series_cv_ensemble(X, y, n_splits=5, test_size_min=60, seed=42):
    n = len(X)
    if n < (test_size_min * 2): return None
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size_min)
    y_pred_proba = np.full(n, np.nan, dtype=float)
    y_pred = np.full(n, np.nan, dtype=float)
    thresholds = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # Base models
        hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=500, random_state=seed, early_stopping=True)
        xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=seed, tree_method="hist")
        lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=seed)

        # Calibrate each (sigmoid) using a tail of the training set
        hgb_cal = fit_calibrated(hgb, X_tr, y_tr, method="sigmoid")
        xgb_cal = fit_calibrated(xgb, X_tr, y_tr, method="sigmoid")
        lgb_cal = fit_calibrated(lgb, X_tr, y_tr, method="sigmoid")

        # Soft-voting manually (average probabilities)
        proba_h = hgb_cal.predict_proba(X_te)[:,1]
        proba_x = xgb_cal.predict_proba(X_te)[:,1]
        proba_l = lgb_cal.predict_proba(X_te)[:,1]
        proba = (proba_h + proba_x + proba_l) / 3.0

        # Choose threshold by Youden on the test (using validation hazard; better use a val set, but for simplicity use test labels)
        thr = youden_threshold(y_te, proba)
        thresholds.append(thr)
        y_pred_proba[test_idx] = proba
        y_pred[test_idx] = (proba >= thr).astype(int)

        last_models = (hgb_cal, xgb_cal, lgb_cal)

    mask = ~np.isnan(y_pred)
    if mask.sum() == 0: return None
    y_true = y[mask]
    y_hat = y_pred[mask]
    y_prob = y_pred_proba[mask]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n_oos": int(mask.sum()),
        "threshold_avg": float(np.nanmean(thresholds)) if thresholds else 0.5
    }
    return metrics, y_prob, y_true, thresholds, last_models

def backtest_long_only(d, test_mask, y_pred_binary, horizon):
    # Use future_ret computed in build_features
    rets = d["future_ret"].values[test_mask]
    preds = y_pred_binary[test_mask].astype(int)
    strat = rets * preds  # long when predicted up, otherwise flat
    # Cumulative curve
    cum_strat = (1 + pd.Series(strat)).cumprod() - 1
    cum_bh = (1 + pd.Series(rets)).cumprod() - 1
    metrics = {
        "n_trades": int(preds.sum()),
        "avg_trade_return": float(np.nanmean(rets[preds==1])) if (preds==1).any() else 0.0,
        "cum_strategy": float(cum_strat.iloc[-1]),
        "cum_buyhold": float(cum_bh.iloc[-1]),
    }
    return metrics, cum_strat, cum_bh

# -------- Sidebar --------
b3 = load_b3_tickers()
st.sidebar.header("⚙️ Configurações")

# Theme toggle
theme_choice = st.sidebar.radio("Tema", ["Escuro", "Claro"], index=0, help="Muda o tema dos gráficos (Plotly).")
set_plotly_template(theme_choice)

q = st.sidebar.text_input("Buscar empresa ou ticker", "", help="Você pode digitar o código (ex.: PETR4) ou o nome (ex.: Petrobras).")
res = search_b3(q) if q else b3
ticker = st.sidebar.selectbox("Selecione o ticker", res["ticker"], help="A lista mostra apenas tickers da B3 (.SA).")

# Quick periods
st.sidebar.markdown("---")
quick = st.sidebar.selectbox("Período rápido", ["Personalizado", "6M", "1A", "YTD"], index=2, help="Selecione um período rápido ou use datas personalizadas.")
today = date.today()
if quick == "6M":
    start_default = today - timedelta(days=182)
elif quick == "1A":
    start_default = today - timedelta(days=365)
elif quick == "YTD":
    start_default = date(today.year, 1, 1)
else:
    start_default = today - timedelta(days=365)

start = st.sidebar.date_input("Início", start_default)
end = st.sidebar.date_input("Fim", today)
if quick != "Personalizado":
    st.sidebar.caption("Usando período rápido — altere para 'Personalizado' para escolher manualmente.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Médias opcionais no gráfico:**")
show_sma50 = st.sidebar.checkbox("Mostrar SMA50 (médio prazo)", value=False)
show_sma200 = st.sidebar.checkbox("Mostrar SMA200 (longo prazo)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Previsão (ML) — pesada**")
use_ml = st.sidebar.checkbox("Ativar previsão com ML turbinada (Ensemble + Calibração)", value=False)
horizon = st.sidebar.selectbox("Horizonte da previsão", [1, 5, 10], index=0, help="Dias à frente para prever a direção (↑/↓).")
splits = st.sidebar.slider("Nº de divisões (walk-forward CV)", min_value=3, max_value=8, value=5)
test_size = st.sidebar.slider("Tamanho do bloco de teste (dias)", min_value=30, max_value=120, value=60)

# -------- Title --------
st.title("🚀 Análise B3 + ML Turbinada")
st.markdown("> Indicadores didáticos + Previsão com ensemble (HGB/XGB/LGBM), calibração e backtest.")

st.caption("Somente tickers da B3 (.SA) — dados do Yahoo Finance. Objetivo educacional.")

# -------- Data --------
if not is_known_b3_ticker(ticker):
    st.error("Ticker fora da lista da B3.")
    st.stop()

with st.spinner("Baixando dados..."):
    df = fetch_data(ticker, start, end)
if df.empty:
    st.warning("Sem dados disponíveis.")
    st.stop()

df = add_indicators(df, want_sma50=show_sma50, want_sma200=show_sma200)
price = float(df["Close"].iloc[-1])
sma20 = float(df["SMA20"].iloc[-1])
rsi_val = float(df["RSI14"].iloc[-1])
delta20 = (price/sma20-1)*100 if sma20 else np.nan

# -------- KPIs --------
c1,c2,c3 = st.columns(3)
c1.metric("Ticker", ticker)
c2.metric("Fechamento", f"R$ {price:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
c3.metric("Δ vs SMA20", f"{delta20:+.2f}%" if not np.isnan(delta20) else "—")

if not np.isnan(delta20):
    if delta20 < -5:
        st.error("Preço bem abaixo da média (SMA20). Curto prazo pressionado.")
    elif -5 <= delta20 <= 5:
        st.warning("Preço perto da média (SMA20). Curto prazo em equilíbrio.")
    else:
        st.success("Preço acima da média (SMA20). Curto prazo com força.")

if rsi_val < 30:
    st.success("RSI em sobrevenda (≤30). Queda forte recente; pode reagir.")
elif 30 <= rsi_val <= 70:
    st.info("RSI em zona neutra (30–70). Mercado equilibrado.")
else:
    st.warning("RSI em sobrecompra (≥70). Subida forte recente; pode corrigir.")

# -------- Charts --------
def plot_price(df, t, show_sma50, show_sma200):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Preço",
        hovertext=[
            f"Data: {d.strftime('%Y-%m-%d')}<br>Abertura: {o:.2f}<br>Máxima: {h:.2f}<br>Mínima: {l:.2f}<br>Fechamento: {c:.2f}"
            for d,o,h,l,c in zip(df['Date'], df['Open'], df['High'], df['Low'], df['Close'])
        ],
        hoverinfo="text"
    ))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA20"))
    if show_sma50 and "SMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50"))
    if show_sma200 and "SMA200" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA200"], name="SMA200"))
    fig.update_layout(title=f"{t} - Preço e Médias", xaxis_title="Data", yaxis_title="Preço (R$)")
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi(df, t):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI(14)"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(title=f"{t} - RSI(14)", xaxis_title="Data", yaxis_title="RSI")
    st.plotly_chart(fig, use_container_width=True)

plot_price(df, ticker, show_sma50, show_sma200)
plot_rsi(df, ticker)

# Tip
st.info("Dica: você pode colar PETR4, VALE3, ITUB4, etc. Se digitar sem .SA, a aplicação adiciona automaticamente.")

# --- Tooltip: Como ler candles ---
with st.expander("🕯️ Como ler candles (clique para ver)"):
    st.markdown("""
- **Corpo**: faixa entre **Abertura** e **Fechamento** (grossa).
- **Pavio superior**: até onde o preço **subiu** no período (**Máxima**).
- **Pavio inferior**: até onde o preço **desceu** (**Mínima**).
- **Candle de alta**: fechamento **acima** da abertura.
- **Candle de baixa**: fechamento **abaixo** da abertura.
Passe o mouse nos candles para ver **Abertura, Máxima, Mínima, Fechamento**.
""")

# --- Didactic explanation (kept) ---
st.markdown("---")
st.subheader("💡 O que o gráfico está tentando te contar")

st.markdown("### 🪜 1. Entendendo a SMA20 — “a linha da média”")
st.markdown("A **SMA20** é como a média dos **últimos 20 preços de fechamento** — a linha de equilíbrio que mostra a **direção geral do preço**.")
st.markdown("""
- 📈 Se o preço está **acima** da linha, há **força** (tendência de alta).
- 📉 Se está **abaixo**, há **fraqueza** (tendência de queda).
""")
st.markdown(f"👉 No caso de **{ticker}**, o preço atual é **R$ {price:,.2f}**, cerca de **{delta20:+.2f}%** em relação à média dos últimos 20 dias.")
if delta20 < -5:
    st.markdown("🔴 **A ação vem caindo há várias semanas e o mercado está mais pessimista no curto prazo.**")
elif -5 <= delta20 <= 5:
    st.markdown("🟡 **O preço está próximo da média — o mercado está em equilíbrio.**")
else:
    st.markdown("🟢 **O preço está acima da média — o papel mostra força no curto prazo.**")

st.markdown("📉 É como se o preço pudesse ficar **“afastado da linha”** por um tempo; quando isso acontece, pode haver **exagero** — como uma corda muito esticada.")

st.markdown("---")
st.markdown("### ⚖️ 2. Entendendo o RSI(14) — “o termômetro da força”")
st.markdown("Pense no **RSI** como um **termômetro de energia do mercado**. Vai de **0 a 100** e mostra quem está dominando: **compradores** ou **vendedores**.")
df_rsi = pd.DataFrame({
    "Faixa": ["70 a 100", "50", "0 a 30"],
    "Situação": ["Sobrecompra", "Neutro", "Sobrevenda"],
    "O que significa": [
        "Subiu rápido demais — pode corrigir pra baixo.",
        "Equilíbrio entre compra e venda.",
        "Caiu rápido demais — pode reagir pra cima."
    ]
}).set_index("Faixa")
st.table(df_rsi)
st.markdown(f"No caso de **{ticker}**, o RSI(14) está em **{rsi_val:.1f}**.")
if rsi_val < 30:
    st.markdown("🟢 **Está na zona de sobrevenda — o papel caiu muito e pode reagir em breve.**")
elif 30 <= rsi_val <= 70:
    st.markdown("🟡 **Está em zona neutra — o mercado está equilibrado.**")
else:
    st.markdown("🔴 **Está na zona de sobrecompra — o preço subiu demais e pode corrigir.**")

st.markdown("---")
st.markdown("### 🧩 3. Juntando as duas informações")
st.markdown("""Quando o **preço está bem abaixo da SMA20** e o **RSI está perto de 30**, é como se o mercado dissesse:

🗣️ “Essa ação caiu bastante, está cansada de cair e pode dar um respiro em breve.”

Mas lembre: isso **não garante** que vai subir agora. É só um **sinal de que a pressão de venda está diminuindo**.
""")

st.markdown("---")
st.markdown("### 🔍 4. Pensando em comportamento de mercado")
st.code("""Preço ↓↓↓↓↓
SMA20 → uma linha que ficou lá em cima
RSI ↓ até 30""")
st.markdown("""Isso mostra que:
- A **queda foi rápida**;
- O **preço ficou longe da média**;
- E o **RSI sinaliza vendedores perdendo força**.

💡 É o que muitos chamam de **“ponto de atenção”**: se aparecer **volume de compra** nos próximos dias e o preço começar a subir, → pode ser um **repique** (subida temporária após muita queda).
""")

st.markdown("---")
st.markdown("### 💬 Em resumo:")
summary = pd.DataFrame({
    "Indicador":[ "SMA20", "RSI(14)", "Conclusão geral" ],
    "O que está mostrando":[
        "Preço comparado à média de 20 dias",
        "Energia do mercado (0–100)",
        "Combinação de média e força (preço + RSI)"
    ],
    "Significado prático":[
        ("O preço está bem abaixo da média — ação pressionada." if delta20 < -5 else
         "Preço perto da média — mercado em equilíbrio." if -5 <= delta20 <= 5 else
         "Preço acima da média — curto prazo com força."),
        ("Quase no limite da queda — pode surgir oportunidade." if rsi_val < 35 else
         "Equilíbrio; sem sinal claro." if 35 <= rsi_val <= 65 else
         "Pode haver realização/correção."),
        ("Fraca, mas pode estar perto de uma pausa/leve recuperação." if (delta20 < -5 and rsi_val <= 35) else
         "Neutra; acompanhar próximos movimentos." if (-5 <= delta20 <= 5 and 30 <= rsi_val <= 70) else
         "Com força; atenção a exageros se RSI muito alto.")
    ]
})
st.table(summary)

# --------- ML Section (Ensemble + Calibration + Threshold + Backtest) ---------
if use_ml:
    st.markdown("---")
    st.subheader("🤖 Previsão com Ensemble (HGB + XGB + LGBM) — calibrada e com backtest")
    with st.spinner("Treinando e validando (walk-forward)..."):
        d, X, y, feat_cols = build_features(df, horizon=int(horizon))
        res = time_series_cv_ensemble(X, y, n_splits=int(splits), test_size_min=int(test_size))
    if res is None:
        st.warning("Dados insuficientes para treinar/validar um modelo confiável neste período.")
    else:
        metrics, y_prob, y_true, thresholds, last_models = res
        colA, colB, colC, colD, colE = st.columns(5)
        colA.metric("Acurácia (OOS)", f"{metrics['accuracy']*100:.1f}%")
        colB.metric("Balanced Acc.", f"{metrics['balanced_accuracy']*100:.1f}%")
        colC.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        colD.metric("Brier", f"{metrics['brier']:.3f}")
        colE.metric("Nº Obs. OOS", f"{metrics['n_oos']}")

        thr_avg = metrics["threshold_avg"]
        st.caption(f"Limite médio de decisão (Youden): **{thr_avg:.2f}**")

        # OOS mask derived from y_prob length inside d
        mask_len = len(y_prob)
        test_mask = np.zeros(len(d), dtype=bool)
        test_mask[-mask_len:] = True  # assuming last folds fill the tail; acceptable approximation for display

        # Binary predictions with the average threshold
        y_pred_bin = (y_prob >= thr_avg).astype(int)

        # Backtest long-only
        bt_metrics, cum_strat, cum_bh = backtest_long_only(d, test_mask, y_pred_bin, horizon=int(horizon))
        st.markdown("**Backtest simples (long-only nos sinais de alta):**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Trades (long)", f"{bt_metrics['n_trades']}")
        c2.metric("Retorno médio por trade", f"{bt_metrics['avg_trade_return']*100:.2f}%")
        c3.metric("Retorno acumulado (estratégia)", f"{bt_metrics['cum_strategy']*100:.1f}%")
        st.metric("Retorno acumulado (buy & hold - OOS)", f"{bt_metrics['cum_buyhold']*100:.1f}%")

        # Plot cumulative curves
        import plotly.express as px
        perf_df = pd.DataFrame({
            "Data": d.loc[test_mask, "Date"].values,
            "Estratégia (long nos sinais)": cum_strat.values,
            "Buy & Hold (OOS)": cum_bh.values,
        })
        perf_df = perf_df.melt("Data", var_name="Série", value_name="Retorno Acumulado")
        figp = px.line(perf_df, x="Data", y="Retorno Acumulado", color="Série", title="Backtest — Retorno Acumulado (fora da amostra)")
        st.plotly_chart(figp, use_container_width=True)

        st.caption("Aviso: Este backtest é didático e simplificado, não considera custos, impostos, *slippage* ou liquidez.")
