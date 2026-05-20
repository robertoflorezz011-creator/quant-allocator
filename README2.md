# Quant Allocator

**Projeto Final · IA Aplicada ao Mercado Financeiro · FGV EAESP · 2026.1**  
Grupo: Beatriz Vieira, Gustavo Liang, Luiza Chammas, Roberto Florez

Motor de otimização de portfólio com **Hierarchical Risk Parity** para wealth managers, family offices e asset managers.

---

## Como subir no GitHub e publicar (sem terminal)

### 1. Criar o repositório no GitHub
1. Acesse [github.com](https://github.com) e faça login
2. Clique em **New repository** (botão verde)
3. Nome: `quant-allocator`
4. Visibilidade: **Public** *(obrigatório para o Streamlit Cloud gratuito)*
5. Marque **Add a README file**
6. Clique em **Create repository**

### 2. Fazer upload dos arquivos
1. No repositório criado, clique em **Add file → Upload files**
2. Arraste os 3 arquivos: `app.py`, `requirements.txt`, `README.md`
3. Clique em **Commit changes**

### 3. Deploy no Streamlit Cloud
1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Login com GitHub → **New app**
3. Preencha:
   - **Repository:** `SEU_USUARIO/quant-allocator`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Clique em **Deploy!**
5. Em ~3 minutos seu app estará em: `https://quant-allocator-XXXXX.streamlit.app`

---

## Métodos implementados

| Método | Referência | Papel |
|---|---|---|
| Equal Weight (1/N) | DeMiguel et al. (2009) | Benchmark naive |
| Min Variance | Markowitz (1952) | Baseline clássico |
| Max Sharpe + Ledoit-Wolf | Markowitz + Ledoit & Wolf (2004) | Markowitz com shrinkage |
| **HRP** | **López de Prado (2016)** | **Método ML recomendado** |

## Funcionalidades

- Sidebar com capital, perfil de risco (Conservador/Moderado/Arrojado), horizonte e universo de ativos
- Recomendação HRP com métricas esperadas (Sharpe, vol, drawdown, prêmio CDI)
- Gráfico de composição (tabela + pizza)
- Comparativo defensivo HRP vs Markowitz vs Equal Weight
- Backtest walk-forward com anti-look-ahead bias e custos de transação
- Performance em janelas de stress (COVID, alta de juros 2022, eleições 2022)
- Projeção Monte Carlo (12 meses, 1.000 simulações)
- Exportação de proposta em PDF para apresentação ao cliente
- Análise técnica detalhada (fronteira eficiente, drawdowns, sensibilidade, correlação)
