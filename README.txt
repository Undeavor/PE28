# 📈 MLP-ARCH Portfolio Dashboard

Application interactive développée avec Streamlit pour :
- modéliser la volatilité avec des réseaux de neurones (MLP-ARCH)
- backtester des stratégies de portefeuille
- visualiser les performances et risques

---

## Fonctionnalités

- Modèles de volatilité :
  - MLP-ARCH basique (AR + ARCH)
  - MLP-ARCH amélioré (avec volume + effet GARCH)

- Visualisation :
  - Prix normalisés
  - Volatilité réalisée
  - Corrélations
  - Courbes de loss

- Backtesting :
  - Optimisation de portefeuille (Max Sharpe)
  - Allocation dynamique
  - Benchmark equal-weight

- Analyse :
  - Drawdown
  - Distribution des rendements
  - Volatilité prédite vs réalisée
  - Sharpe par actif

---

## Modèle

Le modèle combine :
- composante AR(1)
- réseau de neurones MLP
- structure ARCH / GARCH

Implémenté en **PyTorch** pour flexibilité et performance.

---

## ⚙️ Installation

Clone le repo :

```bash
git clone <repo_url>
cd mlp-arch-dashboard
