# B3 + ML Turbinada — v9.3

App didático para análise de ações da B3 com indicadores e previsão por ensemble (HGB/XGB/LGBM) + calibração, threshold configurável, filtros de estratégia e backtest com métricas de risco.

> ⚠️ Uso educacional. Não é recomendação de investimento.

## Como publicar (Streamlit Cloud)
1. Repositório com: `streamlit_app.py`, `requirements.txt`, `b3_utils.py`, `data/b3_tickers.csv`, `.streamlit/config.toml`.
2. Em https://streamlit.io/cloud → **New app** → selecione repo/branch → **Main file** = `streamlit_app.py` → **Deploy`.
3. (Opcional) `runtime.txt` com `python-3.11`.

## Como usar (passo a passo)
1. **Ticker e Período**: escolha um ativo da B3 e selecione 6M, 1A, YTD ou datas personalizadas.
2. **Gráfico**: ative **SMA50/SMA200** para tendência de médio/longo prazo.
3. **Previsão (ML)**: marque “Ativar previsão” e ajuste os parâmetros (explicados abaixo). Veja métricas OOS, probabilidade do próximo passo e backtest.
4. **“O que os resultados dizem (dinâmico)”**: interpretação em linguagem simples com base nas métricas e no backtest.

## Parâmetros — o que são e quando ajustar
**Tema (Escuro/Claro)**  
Só muda o visual dos gráficos.

**Buscar/Selecionar ticker**  
Filtra e escolhe ativos da B3 (.SA).

**Período rápido (6M, 1A, YTD/Personalizado)**  
- Períodos maiores → modelos mais estáveis.  
- Períodos curtos → úteis para cenários recentes, porém mais ruidosos.

**SMA50/SMA200 no gráfico**  
- SMA50 = tendência intermediária; SMA200 = tendência principal.  
- Úteis para contexto visual (não alteram o modelo).

### Seção: Previsão (ML) — pesada
**Ativar previsão com ML**  
Liga o ensemble (HGB, XGB, LGBM) com **calibração** das probabilidades e validação temporal (*walk‑forward*).

**Horizonte da previsão (1/5/10 dias)**  
- 1 dia: muito sensível a ruído, reage rápido.  
- 5 dias: compromisso entre ruído e tendência.  
- 10 dias: movimentos mais “lentos”, precisa de período maior para treinar.

**Nº de divisões (splits)**  
Quantas janelas de treino/teste serão feitas em sequência temporal.  
- 3: mais rápido, útil em 6M.  
- 5+: mais robusto (use com ≥1A).

**Tamanho do bloco de teste (dias)**  
Tamanho do pedaço reservado para teste em cada *split*.  
- 30–40: bom para 6M.  
- 60–80: bom para ≥1A.  
O app **ajusta automaticamente** se faltar histórico para evitar erros.

**Método do limiar**  
Como escolher o ponto de corte da probabilidade **em cada split**:  
- **Youden (acerto)**: otimiza acerto/ROC. Útil para avaliar “qualidade de classificação”.  
- **Retorno OOS (backtest)**: busca o limiar que **maximiza retorno** no teste. Alinha o classificador ao objetivo financeiro.

**Filtro de confiança — mín. prob. para entrar (long)**  
Define o piso para entrar comprado (ex.: 0.55 = só opera quando prob.≥55%). **Aumente** para evitar trades “fracos”.

**Banda neutra em torno de 50%**  
Zona morta onde **não há trade** (ex.: 0.05 → sem trade se prob.∈[45%,55%]). Reduz sinais ambíguos.

**Filtro de tendência (Preço>SMA200)**  
Opera **a favor** da tendência de longo prazo. Útil em mercados direcionais.

**Permitir contrarian em sobrevenda (RSI<30)** + **Limite de distância à SMA20**  
Permite comprar **contra** a tendência apenas em **exageros de queda** (RSI<30 e preço X% abaixo da SMA20). Ex.: −0,05 = 5% abaixo.

## Métricas e explicação dinâmica
- **AUC**: 0.5=aleatório; 0.53–0.60 pequena vantagem; ≥0.60 moderada; ≥0.65 forte.  
- **Brier**: qualidade/calibração das probabilidades (quanto mais baixo, melhor; ~0.25 ≈ moeda).  
- **Acurácia / Balanced Acc.**: acerto global e balanceado.  
- **Prob. do próximo passo**: leitura (↑/↓) imediata do ensemble.  
- **Backtest (OOS)**: retorno acumulado, **max drawdown**, **volatilidade**, nº de trades e retorno médio/trade.  
A seção “**O que os resultados dizem (dinâmico)**” traduz tudo isso em linguagem simples com base nos valores exibidos.

## Dicas rápidas
- Para **6M**: `splits=3`, `test_size=30–40`.  
- Para **≥1A**: `splits=5`, `test_size=60–80`.  
- Para **horizonte=10**, prefira períodos ≥1–2 anos.  
- Se as probabilidades estiverem “fracas” (Brier ≈ 0.25), aumente período e/ou use filtro de confiança/banda neutra.

## Limitações
- Sem custos transacionais, impostos ou slippage no backtest.  
- Resultados variam com regime de mercado e parâmetros.

## Licença
Ajuste conforme sua necessidade (ex.: MIT).
