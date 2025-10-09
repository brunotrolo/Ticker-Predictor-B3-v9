
import streamlit as st
import pandas as pd, numpy as np
from datetime import date, timedelta
import yfinance as yf
import plotly.graph_objects as go
from b3_utils import load_b3_tickers, ensure_sa_suffix, is_known_b3_ticker, search_b3

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title="B3 + ML Turbinada v9.1", page_icon="🚀", layout="wide")

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
    # Skip calibration for very short training blocks
    if n < 40:
        m = clone(model)
        m.fit(X_train, y_train)
        return m
    n_cal = max(int(n * frac_calib), 50) if n >= 100 else max(int(n * 0.1), 20)
    n_cal = min(n_cal, n-20) if n > 40 else max(5, n-5)
    m = clone(model); m.fit(X_train[:-n_cal], y_train[:-n_cal])
    cal = CalibratedClassifierCV(m, method=method, cv="prefit")
    cal.fit(X_train[-n_cal:], y_train[-n_cal:])
    return cal

def safe_tscv_params(n_samples, n_splits, test_size_min):
    # Ensure TimeSeriesSplit won't raise for short series
    max_splits = max(1, n_samples // max(1, test_size_min) - 1)
    adj_splits = min(n_splits, max_splits)
    adj_test = test_size_min
    while adj_splits < 2 and adj_test > 20:
        adj_test = max(20, adj_test // 2)
        max_splits = max(1, n_samples // max(1, adj_test) - 1)
        adj_splits = min(n_splits, max_splits)
    return adj_splits, adj_test

def time_series_cv_ensemble(X, y, n_splits=5, test_size_min=60, seed=42):
    n = len(X)
    if n < 80:
        return {"note": "Poucos dados para CV robusta (mín. ~80 amostras)."}, None, None, None, None
    n_splits_safe, test_size_safe = safe_tscv_params(n, n_splits, test_size_min)
    if n_splits_safe < 2:
        return {"note": f"Amostra insuficiente para dividir {n_splits}x com teste={test_size_min}. Reduza o período, o 'test_size' ou os 'splits'."}, None, None, None, None

    tscv = TimeSeriesSplit(n_splits=n_splits_safe, test_size=test_size_safe)
    y_pred_proba = np.full(n, np.nan, dtype=float)
    y_pred = np.full(n, np.nan, dtype=float)
    thresholds = []

    last_models = None
    for train_idx, test_idx in tscv.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # Base models
        hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=500, random_state=seed, early_stopping=True)
        xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=seed, tree_method="hist")
        # LightGBM silenciado e um pouco mais estável em janela curta
        lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                             subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1)

        # Se a janela de treino for muito pequena, podemos pular o LGB para robustez
        use_lgb = (len(X_tr) >= 150)

        hgb_cal = fit_calibrated(hgb, X_tr, y_tr, method="sigmoid")
        xgb_cal = fit_calibrated(xgb, X_tr, y_tr, method="sigmoid")
        models = [hgb_cal, xgb_cal]

        if use_lgb:
            lgb_cal = fit_calibrated(lgb, X_tr, y_tr, method="sigmoid")
            models.append(lgb_cal)

        # Soft voting (média das probabilidades)
        probs = [m.predict_proba(X_te)[:,1] for m in models]
        proba = np.mean(probs, axis=0)

        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y_te, proba)
        j = tpr - fpr
        thr_fold = thr[int(np.argmax(j))]
        thresholds.append(thr_fold)

        y_pred_proba[test_idx] = proba
        y_pred[test_idx] = (proba >= thr_fold).astype(int)

        last_models = models

    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"note": "Falha ao gerar previsões OOS."}, None, None, None, None

    y_true = y[mask]; y_hat = y_pred[mask]; y_prob = y_pred_proba[mask]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n_oos": int(mask.sum()),
        "threshold_avg": float(np.nanmean(thresholds)) if thresholds else 0.5,
        "adj_splits": int(n_splits_safe),
        "adj_test_size": int(test_size_safe)
    }
    return metrics, y_prob, y_true, thresholds, last_models

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
test_size = st.sidebar.slider("Tamanho do bloco de teste (dias)", min_value=20, max_value=120, value=60)

# -------- Title --------
st.title("🚀 Análise B3 + ML Turbinada — v9.1")
st.markdown("> Indicadores didáticos + Previsão com ensemble (HGB/XGB/LGBM), calibração, threshold e backtest — com correções de estabilidade.")

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
        name="Preço"
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

# --- Didactic explanation (short) ---
with st.expander("💡 Como interpretar rapidamente (clique para abrir)"):
    st.markdown(f"- **Δ vs SMA20**: {delta20:+.2f}% → abaixo = fraqueza; acima = força.")
    st.markdown(f"- **RSI(14)**: {rsi_val:.1f} → ≤30: sobrevenda; ≥70: sobrecompra; meio: neutro.")

# --------- ML Section (Ensemble + Calibration + Threshold + Backtest) ---------
if use_ml:
    st.markdown("---")
    st.subheader("🤖 Previsão com Ensemble (HGB + XGB + LGBM) — calibrada e com backtest")
    with st.spinner("Treinando e validando (walk-forward)..."):
        d, X, y, feat_cols = build_features(df, horizon=int(horizon))

        # SANITIZAÇÃO: manter apenas linhas totalmente finitas (evita NaN/Inf no treino)
        finite_rows = np.isfinite(X).all(axis=1)
        X, y = X[finite_rows], y[finite_rows]
        d = d.loc[finite_rows].reset_index(drop=True)

        if len(X) < 80:
            st.warning("Poucos dados úteis após sanitização (NaN/Inf). Aumente o período, reduza o horizonte ou ajuste o test_size/splits.")
        else:
            metrics, y_prob, y_true, thresholds, last_models = time_series_cv_ensemble(
                X, y, n_splits=int(splits), test_size_min=int(test_size)
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
                st.caption(f"CV ajustada: splits={metrics['adj_splits']} • test_size={metrics['adj_test_size']}")

                # Probabilidade de alta (próximo passo)
                if last_models is not None and len(d) > 0:
                    x_next = d[feat_cols].values[-1:].copy()
                    probs = [m.predict_proba(x_next)[:,1] for m in last_models]
                    proba_next = float(np.mean(probs))
                    st.metric(f"Prob. de alta em {horizon} dia(s)", f"{proba_next*100:.1f}%")

                # Backtest simples (long-only nos sinais de alta)
                if y_prob is not None:
                    thr_avg = metrics["threshold_avg"]
                    y_pred_bin = (y_prob >= thr_avg).astype(int)

                    # Alinha máscara OOS aproximando com tail
                    mask_len = len(y_prob)
                    test_mask = np.zeros(len(d), dtype=bool)
                    test_mask[-mask_len:] = True

                    rets = d["future_ret"].values[test_mask]
                    preds = y_pred_bin.astype(int)
                    strat = rets * preds

                    cum_strat = (1 + pd.Series(strat)).cumprod() - 1
                    cum_bh = (1 + pd.Series(rets)).cumprod() - 1

                    bt_metrics = {
                        "n_trades": int(preds.sum()),
                        "avg_trade_return": float(np.nanmean(rets[preds==1])) if (preds==1).any() else 0.0,
                        "cum_strategy": float(cum_strat.iloc[-1]),
                        "cum_buyhold": float(cum_bh.iloc[-1]),
                    }

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trades (long)", f"{bt_metrics['n_trades']}")
                    c2.metric("Retorno médio/trade", f"{bt_metrics['avg_trade_return']*100:.2f}%")
                    c3.metric("Retorno acumulado (estratégia)", f"{bt_metrics['cum_strategy']*100:.1f}%")
                    st.metric("Retorno acumulado (buy & hold - OOS)", f"{bt_metrics['cum_buyhold']*100:.1f}%")
