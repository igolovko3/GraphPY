# GraphPY

MCMC sampler for Graphical Pitman–Yor process mixture model with a pluggable observation models (1-D Gaussian, multivariate Gaussian, etc.) Includes utilities for consensus clustering (PSM + minVI) and posterior predictive plots.

## Features
- **One engine, many models**
- **Explicit, testable helpers**
- **Posterior predictive**: compute vs. plot decoupled (`compute_posterior_predictive`, `plot_posterior`).

## Install
Clone the repository and run from source:
```bash
pip install -e ".[dev]"
```

Alternatively, to directly install the library run:
```bash
pip install git+https://github.com/igolovko3/GraphPY
```

