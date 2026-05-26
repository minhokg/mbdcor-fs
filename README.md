# mbdcor-fs

Feature selection using **Markov Boundary Discovery with Distance Correlation (MBDcor)** for nonlinear binary classification problems.

---

## Overview

`mbdcor-fs` is a feature selection framework inspired by **Markov Boundary discovery** and **distance correlation** theory.

The repository implements the methodology proposed in the master thesis:

> *Markov Boundary Discovery with Distance Correlation for Feature Selection in Classification Problems*  
> Minho Kang, Hertie School

The goal of the project is to identify a compact and informative subset of features that approximates the **Markov Boundary** of a target variable while remaining computationally efficient in high-dimensional settings.

The method combines:

- marginal dependence screening,
- residual-based conditional dependence testing,
- random subspace exploration,
- stability-based aggregation,
- and adaptive pruning.

Unlike traditional correlation-based feature selection methods, MBDcor is designed to capture both **linear and nonlinear dependencies** using **distance correlation**.

The repository also includes experiments comparing MBDcor against **Boruta Random Forest** feature selection on both synthetic and real-world datasets.

---

## Motivation

In high-dimensional machine learning problems, many variables may be:

- irrelevant,
- redundant,
- noisy,
- or highly correlated.

The Markov Boundary provides a theoretically grounded target for feature selection because it represents the minimal set of variables that contains all predictive information about the target.

However, exact Markov Boundary recovery is computationally difficult, especially in nonlinear settings.

MBDcor provides a practical approximation approach using distance-correlation-based dependence testing and stochastic stability selection.

---


## Repository Structure

```text
mbdcor-fs/
│
├── src/
│   └── mbdcor_fs/
│       ├── analysis/                   # Analysis scripts and experiments
│       └── utils/
│           ├── base/                  # Base classes and shared components
│           ├── correlation_functions/ # Distance correlation implementations
│           ├── feature_selection/     # Feature selection algorithms
│           ├── helper/                # Utility/helper functions
│           └── train_evaluate/        # Model training and evaluation utilities
│
├── requirements.txt                   # Python dependencies
└── README.md
```


## Installation

Clone the repository:

```bash
git clone https://github.com/minhokg/mbdcor-fs.git
cd mbdcor-fs
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

Run pre-commit checks manually:

```bash
pre-commit run --all-files
```

---


## Experimental Design

The repository includes experiments from the thesis involving:

- Synthetic nonlinear classification datasets
- Correlated feature structures
- Monte Carlo simulations
- Real-world Breast Cancer Wisconsin dataset

MBDcor is compared against:

- Boruta Random Forest Feature Selection

Evaluation metrics include:

- Predictive log-loss
- Feature recall
- Runtime
- Number of selected features

---





## Thesis

This repository accompanies the thesis:

> **Markov Boundary Discovery with Distance Correlation for Feature Selection in Classification Problems**  
> Minho Kang  
> Hertie School — Data Science for Public Policy

---


## License

This project is licensed under the MIT License.

---

## Author

Developed by Minho Kang.

GitHub Repository: https://github.com/minhokg/mbdcor-fs