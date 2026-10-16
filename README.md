# 📊 Quantitative Risk Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![NumPy](https://img.shields.io/badge/NumPy-1.26-orange?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![CI](https://img.shields.io/github/actions/workflow/status/alaaeddinethr/quant-risk-engine/ci.yml?style=flat-square&label=CI)

**Portfolio risk analytics engine — VaR, CVaR, Monte Carlo simulation, Markowitz optimization, and stress testing in one unified pipeline.**

[Live Demo](#) · [Documentation](#architecture) · [Quickstart](#quickstart)

</div>

---

## 📌 Context & Motivation

Modern portfolio management requires more than tracking returns — it demands a rigorous understanding of **downside risk** under normal and extreme conditions. This project implements a full quantitative risk pipeline used in professional asset management, from raw price data to actionable risk reports.

Built as a practical application of quantitative finance methods studied in the **MSc Financial Engineering (WorldQuant University)**.

---

## 🏗️ Architecture

```
quant-risk-engine/
│
├── src/
│   ├── data/
│   │   └── market_data.py         # yfinance downloader, returns computation
│   │
│   ├── risk/
│   │   ├── var.py                 # VaR: Historical, Parametric (Normal/t), Monte Carlo
│   │   ├── monte_carlo.py         # Full portfolio simulation engine
│   │   └── stress_test.py         # Historical scenario replay (2008, COVID, 2022)
│   │
│   ├── portfolio/
│   │   ├── metrics.py             # Sharpe, Sortino, Calmar, Max Drawdown
│   │   └── optimizer.py           # Markowitz efficient frontier + SLSQP
│   │
│   └── dashboard/
│       └── app.py                 # Streamlit interactive dashboard
│
├── tests/
│   ├── test_var.py
│   ├── test_portfolio.py
│   └── test_monte_carlo.py
│
├── .github/workflows/ci.yml
├── config.yaml
└── requirements.txt
```

---

## 🎯 Core Capabilities

### Risk Measurement
| Metric | Method | Confidence Level |
|--------|--------|-----------------|
| VaR | Historical Simulation | 95%, 99% |
| VaR | Parametric (Normal) | 95%, 99% |
| VaR | Parametric (Student-t) | 95%, 99% |
| VaR | Monte Carlo | 95%, 99% |
| CVaR / ES | All methods above | 95%, 99% |

### Portfolio Optimization
- **Minimum Variance Portfolio** — lowest risk for a given return
- **Maximum Sharpe Ratio** — best risk-adjusted return
- **Efficient Frontier** — full Pareto curve of optimal portfolios
- **Risk Parity** — equal risk contribution per asset

### Stress Testing
| Scenario | Period | Market Drop |
|----------|--------|-------------|
| Global Financial Crisis | Sep 2008 – Mar 2009 | −54% S&P 500 |
| COVID Crash | Feb 2020 – Mar 2020 | −34% S&P 500 |
| 2022 Rate Shock | Jan 2022 – Oct 2022 | −25% S&P 500 |
| Dot-com Bubble | Mar 2000 – Oct 2002 | −49% NASDAQ |

---

## 📈 Results

On a sample portfolio (AAPL, MSFT, GOOGL, JPM, GS, BRK-B — equal weight):

| Metric | Value |
|--------|-------|
| Annual Return | 18.4% |
| Annual Volatility | 16.2% |
| Sharpe Ratio | 1.14 |
| Max Drawdown | −22.3% |
| 95% 1-day VaR (hist.) | −1.83% |
| 95% CVaR | −2.61% |
| 99% VaR (MC, 10k sim.) | −3.47% |

*Evaluated on 5 years of daily data (2019–2024)*

---

## 🚀 Quick Start

```bash
git clone https://github.com/alaaeddinethr/quant-risk-engine.git
cd quant-risk-engine
pip install -r requirements.txt

# Download market data
python src/data/market_data.py --tickers AAPL MSFT GOOGL JPM GS --years 5

# Run full risk report
python src/risk/var.py --portfolio config.yaml

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## 📚 References

- [1] Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.
- [2] Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance, 7(1), 77–91.
- [3] Rockafellar, R.T. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk.
- [4] McNeil, A.J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management*. Princeton University Press.

---

## 👤 Author

**Alaa Eddine TAHIR** — Engineering student at Arts et Métiers ParisTech | MSc Financial Engineering (WorldQuant University)
[LinkedIn](https://linkedin.com/in/alaaeddine-tahir) · [GitHub](https://github.com/alaaeddinethr)
