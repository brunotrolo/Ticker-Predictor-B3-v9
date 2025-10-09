import streamlit as st
import pandas as pd, numpy as np
from datetime import date, timedelta
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from b3_utils import load_b3_tickers, ensure_sa_suffix, is_known_b3_ticker, search_b3

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title="B3 + ML Turbinada v9.2", page_icon="🚀", layout="wide")

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
    if want_sma50: df["SMA50"]=sma(df["Close"],50)
    if want_sma200: df["SMA200"]=sma(df["Close"],200)
    df["RSI14"]=rsi(df["Close"])
    return df

def build_features(df, horizon=1):
    d = df.copy()
    d["ret_1"] = d["Close"].pct_change(1)
    d["ret_3"] = d["Close"].pct_change(3)
    d["ret_5"] = d["Close"].pct_change(5)
    d["ret_10"] = d["Close"].pct_change(10)
    if "SMA50" not in d.columns: d["SMA50"] = sma(d["Close"],50)
    if "SMA200" not in d.columns: d["SMA200"] = sma(d["Close"],200)
    d["dist_sma20"] = d["Close"]/d["SMA20"] - 1
    d["dist_sma50"] = d["Close"]/d["SMA50"] - 1
    d["dist_sma200"] = d["Close"]/d["SMA200"] - 1
    d["vol_5"] = d["Close"].pct_change().rolling(5).std()
    d["vol_10"] = d["Close"].pct_change().rolling(10).std()
    d["rsi"] = d["RSI14"]
    d["future_ret"] = d["Close"].shift(-horizon)/d["Close"] - 1.0
    d["target_up"] = (d["future_ret"] > 0).astype(int)
    feat_cols = ["ret_1","ret_3","ret_5","ret_10","dist_sma20","dist_sma50","dist_sma200","vol_5","vol_10","rsi"]
    d = d.dropna(subset=feat_cols + ["target_up","future_ret"]).reset_index(drop=True)
    X = d[feat_cols].values; y = d["target_up"].values; future_ret = d["future_ret"].values
    return d, X, y, future_ret, feat_cols

from sklearn.metrics import roc_curve
def fit_calibrated(model, X_train, y_train, frac_calib=0.2, method="sigmoid"):
    n = len(X_train)
    if n < 40:
        m = clone(model); m.fit(X_train, y_train); return m
    n_cal = max(int(n * frac_calib), 50) if n >= 100 else max(int(n * 0.1), 20)
    n_cal = min(n_cal, n-20) if n > 40 else max(5, n-5)
    m = clone(model); m.fit(X_train[:-n_cal], y_train[:-n_cal])
    cal = CalibratedClassifierCV(m, method=method, cv="prefit")
    cal.fit(X_train[-n_cal:], y_train[-n_cal:])
    return cal

def safe_tscv_params(n_samples, n_splits, test_size_min):
    max_splits = max(1, n_samples // max(1, test_size_min) - 1)
    adj_splits = min(n_splits, max_splits)
    adj_test = test_size_min
    while adj_splits < 2 and adj_test > 20:
        adj_test = max(20, adj_test // 2)
        max_splits = max(1, n_samples // max(1, adj_test) - 1)
        adj_splits = min(n_splits, max_splits)
    return adj_splits, adj_test

def best_threshold_by_return(proba, rets):
    if len(proba) != len(rets) or len(proba) == 0:
        return 0.5
    grid = np.linspace(0.4, 0.7, 61)
    best_thr, best_ret = 0.5, -1e9
    for thr in grid:
        sig = (proba >= thr).astype(int)
        cum = (1 + pd.Series(rets * sig)).prod() - 1
        if cum > best_ret:
            best_ret = float(cum); best_thr = float(thr)
    return best_thr

def time_series_cv_ensemble(X, y, future_ret, n_splits=5, test_size_min=60, seed=42, thr_method="youden"):
    n = len(X)
    if n < 80:
        return {"note": "Poucos dados para CV robusta (mín. ~80 amostras)."}, None, None, None, None, None
    n_splits_safe, test_size_safe = safe_tscv_params(n, n_splits, test_size_min)
    if n_splits_safe < 2:
        return {"note": f"Amostra insuficiente para dividir {n_splits}x com teste={test_size_min}. Reduza o período, o 'test_size' ou os 'splits'."}, None, None, None, None, None

    tscv = TimeSeriesSplit(n_splits=n_splits_safe, test_size=test_size_safe)
    y_pred_proba = np.full(n, np.nan, dtype=float)
    thresholds = []
    last_models = None

    for train_idx, test_idx in tscv.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        rets_te = future_ret[test_idx]

        hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=500, random_state=seed, early_stopping=True)
        xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=seed, tree_method="hist")
        lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1)

        use_lgb = (len(X_tr) >= 150)

        hgb_cal = fit_calibrated(hgb, X_tr, y_tr, method="sigmoid")
        xgb_cal = fit_calibrated(xgb, X_tr, y_tr, method="sigmoid")
        models = [hgb_cal, xgb_cal]
        if use_lgb:
            lgb_cal = fit_calibrated(lgb, X_tr, y_tr, method="sigmoid")
            models.append(lgb_cal)

        probs = [m.predict_proba(X_te)[:,1] for m in models]
        proba = np.mean(probs, axis=0)

        if thr_method == "retorno":
            thr_fold = best_threshold_by_return(proba, rets_te)
        else:
            from sklearn.metrics import roc_curve
            fpr, tpr, thr = roc_curve(y_te, proba); j = tpr - fpr
            thr_fold = thr[int(np.argmax(j))]

        thresholds.append(float(thr_fold))
        y_pred_proba[test_idx] = proba
        last_models = models

    mask = ~np.isnan(y_pred_proba)
    if mask.sum() == 0:
        return {"note": "Falha ao gerar previsões OOS."}, None, None, None, None, None

    y_true = y[mask]; y_prob = y_pred_proba[mask]
    metrics = {
        "accuracy": float(accuracy_score(y_true, (y_prob>=0.5).astype(int))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, (y_prob>=0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n_oos": int(mask.sum()),
        "threshold_avg": float(np.nanmean(thresholds)) if thresholds else 0.5,
        "adj_splits": int(n_splits_safe),
        "adj_test_size": int(test_size_safe)
    }
    return metrics, y_prob, y_true, thresholds, last_models, mask

def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    equity = (1 + pd.Series(returns)).cumprod()
    peak = equity.cummax()
    dd = equity/peak - 1.0
    return float(dd.min())

# Sidebar
b3 = load_b3_tickers()
st.sidebar.header("⚙️ Configurações")
theme_choice = st.sidebar.radio("Tema", ["Escuro", "Claro"], index=0)
set_plotly_template(theme_choice)

q = st.sidebar.text_input("Buscar empresa ou ticker", "")
res = search_b3(q) if q else b3
ticker = st.sidebar.selectbox("Selecione o ticker", res["ticker"])

st.sidebar.markdown("---")
quick = st.sidebar.selectbox("Período rápido", ["Personalizado", "6M", "1A", "YTD"], index=2)
from datetime import date, timedelta
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

st.sidebar.markdown("---")
show_sma50 = st.sidebar.checkbox("Mostrar SMA50 (médio prazo)", value=False)
show_sma200 = st.sidebar.checkbox("Mostrar SMA200 (longo prazo)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Previsão (ML) — pesada**")
use_ml = st.sidebar.checkbox("Ativar previsão com ML", value=False)
horizon = st.sidebar.selectbox("Horizonte da previsão", [1,5,10], index=0)
splits = st.sidebar.slider("Nº de divisões (walk-forward CV)", 3, 8, 5)
test_size = st.sidebar.slider("Tamanho do bloco de teste (dias)", 20, 120, 60)
thr_method = st.sidebar.selectbox("Método do limiar", ["Youden (acerto)", "Retorno OOS (backtest)"], index=1)
min_prob = st.sidebar.slider("Filtro de confiança — mín. prob. para entrar (long)", 0.50, 0.70, 0.55, 0.01)
neutral_band = st.sidebar.slider("Banda neutra (sem trade) em torno de 50%", 0.00, 0.10, 0.05, 0.01)

st.sidebar.markdown("**Filtro de tendência**")
use_trend = st.sidebar.checkbox("Operar long apenas se Preço > SMA200", value=True)
allow_contrarian = st.sidebar.checkbox("Permitir contrarian em sobrevenda (RSI<30)", value=True)
contrarian_max_dist = st.sidebar.slider("Limite de distância à SMA20 para contrarian (negativo = abaixo)", -0.15, 0.00, -0.05, 0.01)

st.title("🚀 Análise B3 + ML Turbinada — v9.2")
st.markdown("> Ensemble calibrado + threshold configurável + filtros de estratégia + backtest com risco, e explicação dinâmica dos resultados.")
st.caption("Somente tickers da B3 (.SA) — dados do Yahoo Finance. Objetivo educacional.")

# Data
if not is_known_b3_ticker(ticker):
    st.error("Ticker fora da lista da B3."); st.stop()
with st.spinner("Baixando dados..."):
    df = fetch_data(ticker, start, end)
if df.empty:
    st.warning("Sem dados disponíveis."); st.stop()
df = add_indicators(df, want_sma50=show_sma50, want_sma200=show_sma200)
price = float(df["Close"].iloc[-1]); sma20 = float(df["SMA20"].iloc[-1]); rsi_val = float(df["RSI14"].iloc[-1])
delta20 = (price/sma20-1)*100 if sma20 else np.nan

c1,c2,c3 = st.columns(3)
c1.metric("Ticker", ticker)
c2.metric("Fechamento", f"R$ {price:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
c3.metric("Δ vs SMA20", f"{delta20:+.2f}%" if not np.isnan(delta20) else "—")

def plot_price(df, t, show_sma50, show_sma200):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Preço"))
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
    fig.add_hline(y=70, line_dash="dash"); fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(title=f"{t} - RSI(14)", xaxis_title="Data", yaxis_title="RSI")
    st.plotly_chart(fig, use_container_width=True)

plot_price(df, ticker, show_sma50, show_sma200)
plot_rsi(df, ticker)
st.info("Dica: cole PETR4, VALE3, ITUB4 etc. Sem .SA? A app adiciona automaticamente.")

if use_ml:
    st.markdown("---")
    st.subheader("🤖 Previsão e Estratégia")
    with st.spinner("Treinando e validando (walk-forward)..."):
        d, X, y, future_ret, feat_cols = build_features(df, horizon=int(horizon))
        finite_rows = np.isfinite(X).all(axis=1)
        d, X, y, future_ret = d.loc[finite_rows].reset_index(drop=True), X[finite_rows], y[finite_rows], future_ret[finite_rows]
        if len(X) < 80:
            st.warning("Poucos dados úteis após sanitização (NaN/Inf). Aumente o período, reduza o horizonte ou ajuste test_size/splits.")
        else:
            method_key = "retorno" if thr_method.startswith("Retorno") else "youden"
            metrics, y_prob, y_true, thresholds, last_models, oos_mask = time_series_cv_ensemble(
                X, y, future_ret, n_splits=int(splits), test_size_min=int(test_size), thr_method=method_key
            )
            if isinstance(metrics, dict) and "note" in metrics and y_prob is None:
                st.warning(metrics["note"] + " — Tente **um período maior**, **test_size menor** ou **menos splits**.")
            else:
                colA, colB, colC, colD, colE = st.columns(5)
                colA.metric("Acurácia (OOS)", f"{metrics['accuracy']*100:.1f}%")
                colB.metric("Balanced Acc.", f"{metrics['balanced_accuracy']*100:.1f}%")
                colC.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
                colD.metric("Brier", f"{metrics['brier']:.3f}")
                colE.metric("OOS", f"{metrics['n_oos']}")
                st.caption(f"CV ajustada: splits={metrics['adj_splits']} • test_size={metrics['adj_test_size']} • Limiar: {thr_method}")
                proba_next = None
                if last_models is not None and len(d) > 0:
                    x_next = d[feat_cols].values[-1:].copy()
                    proba_next = float(np.mean([m.predict_proba(x_next)[:,1] for m in last_models]))
                    st.metric(f"Prob. de alta em {horizon} dia(s)", f"{proba_next*100:.1f}%")
                thr_avg = metrics["threshold_avg"]
                prob_oos = y_prob
                rets_oos = future_ret[oos_mask]
                sig = (prob_oos >= thr_avg).astype(int)
                low_b = 0.5 - neutral_band; high_b = 0.5 + neutral_band
                neutral = (prob_oos >= low_b) & (prob_oos <= high_b)
                sig[neutral] = 0
                sig[prob_oos < min_prob] = 0
                if use_trend:
                    px = d.loc[oos_mask, "Close"].values
                    sma200 = d.loc[oos_mask, "SMA200"].values
                    rsi_oos = d.loc[oos_mask, "rsi"].values
                    dist20 = d.loc[oos_mask, "dist_sma20"].values
                    above_trend = np.isfinite(sma200) & (px > sma200)
                    contrarian = (rsi_oos < 30) & (dist20 <= contrarian_max_dist)
                    allow = above_trend | (allow_contrarian & contrarian)
                    sig = sig * allow.astype(int)
                strat = rets_oos * sig
                cum_strat = (1 + pd.Series(strat)).cumprod() - 1
                cum_bh = (1 + pd.Series(rets_oos)).cumprod() - 1
                def max_drawdown(returns):
                    if len(returns) == 0:
                        return 0.0
                    equity = (1 + pd.Series(returns)).cumprod()
                    peak = equity.cummax()
                    dd = equity/peak - 1.0
                    return float(dd.min())
                dd_strat = max_drawdown(strat); dd_bh = max_drawdown(rets_oos)
                vol_strat = float(np.nanstd(strat)); vol_bh = float(np.nanstd(rets_oos))
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Trades (long)", f"{int(sig.sum())}")
                c2.metric("Ret. médio/trade", f"{(np.nanmean(rets_oos[sig==1])*100 if (sig==1).any() else 0.0):.2f}%")
                c3.metric("Ret. acumulado (estratégia)", f"{float(cum_strat.iloc[-1])*100:.1f}%")
                c4.metric("Máx. drawdown (estratégia)", f"{dd_strat*100:.1f}%")
                c5.metric("Vol (estratégia, por passo)", f"{vol_strat*100:.2f}%")
                st.metric("Ret. acumulado (buy & hold - OOS)", f"{float(cum_bh.iloc[-1])*100:.1f}%")
                st.caption(f"Vol (buy & hold, por passo): {vol_bh*100:.2f}%  •  Banda neutra: ±{neutral_band*100:.0f}p.p.  •  Filtro de confiança: {min_prob*100:.0f}%")
                perf_df = pd.DataFrame({
                    "Data": d.loc[oos_mask, "Date"].values,
                    "Estratégia (long nos sinais)": cum_strat.values,
                    "Buy & Hold (OOS)": cum_bh.values,
                }).melt("Data", var_name="Série", value_name="Retorno Acumulado")
                figp = px.line(perf_df, x="Data", y="Retorno Acumulado", color="Série", title="Backtest — Retorno Acumulado (fora da amostra)")
                st.plotly_chart(figp, use_container_width=True)
                st.markdown("---")
                st.subheader("🧠 O que os resultados dizem (dinâmico)")
                auc = metrics['roc_auc']; brier = metrics['brier']; acc = metrics['accuracy']
                msg_auc = "vantagem forte" if auc >= 0.65 else "vantagem moderada" if auc >= 0.60 else "vantagem pequena" if auc >= 0.53 else "sinal fraco (perto do acaso)"
                msg_brier = "probabilidades bem informativas/calibradas" if brier < 0.23 else "probabilidades razoáveis" if brier < 0.26 else "probabilidades pouco informativas"
                if proba_next is None: proba_next = 0.5
                bias = "ligeiramente **altista**" if proba_next >= 0.55 else "indefinido" if 0.45 <= proba_next < 0.55 else "ligeiramente **baixista**"
                rel = float(cum_strat.iloc[-1]) - float(cum_bh.iloc[-1])
                if float(cum_strat.iloc[-1]) >= 0 and float(cum_bh.iloc[-1]) < 0:
                    msg_bt = "protegeu nas quedas **e** terminou **positivo**, melhor que o buy & hold."
                elif rel > 0:
                    msg_bt = "venceu o buy & hold no período OOS (melhor desempenho relativo)."
                else:
                    msg_bt = "ficou **abaixo** do buy & hold neste recorte — tente outro período/horizonte ou ajuste filtros."
                st.markdown(f"""
- **AUC {auc:.3f} / Brier {brier:.3f}** → {msg_auc}; {msg_brier}.
- **Acurácia {acc*100:.1f}%** → acima de 50% indica alguma vantagem, mas observe o AUC/Brier.
- **Próximo passo:** prob. de alta = **{proba_next*100:.1f}%** → {bias}.
- **Backtest:** {msg_bt}
- **Risco:** Máx. drawdown da estratégia = **{dd_strat*100:.1f}%**; Volatilidade por passo = **{vol_strat*100:.2f}%**.
- **CV** usada: **splits={metrics['adj_splits']}**, **test_size={metrics['adj_test_size']}**, limiar **{thr_method}**.
                """)
                st.caption("Observação: resultados mudam com período, horizonte, filtros e o próprio regime do mercado. Objetivo educacional.")
