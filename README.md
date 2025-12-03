# Bioprocess GP Modeling Tool

A user-friendly Python tool for modeling bioprocess data using Gaussian Processes (GPyTorch). Designed for engineers to easily define process parameters, handle bolus feeds, and train probabilistic models on time-series data.

## Features

- **Declarative API**: Define your process structure using `Parameter`, `Feed`, and `Output` objects.
- **Smart Feed Handling**: Automatically processes bolus/batch feeds into cumulative features with sigmoid smoothing to handle sampling timing nuances.
- **Probabilistic Modeling**: Built on GPyTorch to provide mean predictions with uncertainty estimates (confidence intervals).
- **Automated Normalization**: Handles data standardization and inverse transformation automatically.
- **Immutable Design**: The `fit` method returns a new `FittedBioprocessModel`, preserving your original model definition.

## Installation

This project uses `uv` for dependency management.

```bash
pip install .
# or with uv
uv pip install .
```

For NixOS users, a `shell.nix` is provided:

```bash
nix-shell
```

## Usage

Define your model structure and fit it to your data:

```python
import pandas as pd
from bioprocess_gp import BioprocessModel, Parameter, Feed, Output, Normal

# 1. Define the model
model = BioprocessModel(
    parameters=[
        Parameter("ph", bounds=(6.0, 8.0), prior=Normal(7.0, 0.5)),
        Parameter("temperature", bounds=(35.0, 39.0))
    ],
    feeds=[
        Feed("glucose_feed")
    ],
    outputs=[
        Output("titer")
    ]
)

# 2. Fit the model (returns a fitted instance)
# data should be a DataFrame with columns for time, run_id, parameters, feeds, and outputs
fitted_model = model.fit(train_df, time_col="time", run_col="run_id")

# 3. Predict on new conditions
predictions = fitted_model.predict(test_df)

print(predictions[["time", "titer_mean", "titer_std"]])
```

## Running the Demo

A complete example with synthetic data generation is available in `demo.py`.

```bash
python demo.py
```

## AI Useage Note

This project was created with the assistance of AI tools (Gemini 3 Pro and Antigravity).
