# mbdcor-fs

Feature selection using **Modified Brownian Distance Correlation (MBDCor)** for high-dimensional machine learning problems.

---

## Overview

`mbdcor-fs` is a lightweight feature selection framework based on distance correlation methods.  
The repository provides tools to identify informative variables by measuring nonlinear dependence between features and target variables.

The project is designed for:

- high-dimensional datasets,
- nonlinear feature-target relationships,
- preprocessing before machine learning models,
- interpretable feature ranking.

The method extends classical distance correlation approaches with a modified Brownian distance correlation framework to improve robustness and feature screening performance.

---

## Features

- Nonlinear dependency-based feature selection
- Works with regression and classification tasks
- Feature ranking and screening utilities
- Simple and modular Python implementation
- Compatible with common ML workflows

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

---

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


## Methodology

The framework uses Modified Brownian Distance Correlation (MBDCor) to evaluate dependence between input variables and target responses.

Compared with traditional correlation-based methods, MBDCor can:

- capture nonlinear relationships,
- detect non-monotonic dependencies,
- reduce sensitivity to linearity assumptions,
- improve feature screening in complex datasets.

Typical pipeline:

1. Load dataset
2. Compute MBDCor scores
3. Rank features
4. Select top-k variables
5. Train downstream ML models



----

## Contributing

Contributions are welcome.

To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## License

This project is licensed under the MIT License.

---

## Author

Developed by Minho Kang.

GitHub Repository: https://github.com/minhokg/mbdcor-fs