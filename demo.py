import pandas as pd
import numpy as np
import torch
import gpytorch
from bioprocess_gp import BioprocessModel, Normal, Parameter, Output, Feed, GaussianSmoothing

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_synthetic_data(n_runs=5, time_points=10):
    data = []
    for run_id in range(n_runs):
        # Random initial conditions
        ph = np.random.normal(7.0, 0.2)
        temp = np.random.normal(37.0, 0.5)
        
        cumulative_feed = 0
        current_titer = 0.1
        
        for t in range(time_points):
            # Feed event at t=5
            feed = 0.0
            if t == 5:
                feed = 5.0
            
            cumulative_feed += feed
            
            # Simple dynamics: Titer grows with time, affected by pH, Temp, and Feed
            growth_rate = 0.1 * (1 - abs(ph - 7.0)) * (1 - abs(temp - 37.0)/10)
            feed_effect = 0.05 * cumulative_feed
            
            current_titer += growth_rate + feed_effect + np.random.normal(0, 0.01)
            
            data.append({
                "run_id": run_id,
                "time": float(t),
                "ph": ph,
                "temperature": temp,
                "glucose_feed": feed,
                "titer": current_titer
            })
            
    return pd.DataFrame(data)

def main():
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_runs=10, time_points=20)
    print(df.head())
    
    print("\nInitializing Model...")
    model = BioprocessModel(
        parameters=[
            Parameter("ph", bounds=(6.0, 8.0), prior=Normal(7.0, 0.5)),
            Parameter("temperature", bounds=(35.0, 39.0), prior=Normal(37.0, 1.0))
        ],
        feeds=[
            Feed("glucose_feed", smoothing=GaussianSmoothing(sigma=2.0))
        ],
        outputs=[
            Output("titer")
        ]
    )
    # Note: Since I updated core.py to have Feed class, I should use it.
    # But I need to import it in demo.py first.
    # Let's fix imports in next step or assume I can use Parameter with is_feed=True for backward compat if I didn't remove it?
    # I removed is_feed from Parameter in core.py. So I MUST use Feed class.
    pass
    
    print("\nFitting Model...")
    # Training on first 8 runs
    train_df = df[df["run_id"] < 8]
    test_df = df[df["run_id"] >= 8]
    
    fitted_model = model.fit(train_df, training_iter=50)
    print("Training complete.")
    
    print("\nPredicting on Test Run (Run 8)...")
    # We need to provide the conditions for Run 8
    # We can just pass the test_df (it has the inputs)
    # The predict method will ignore the target column if present, or we can drop it
    conditions = test_df[test_df["run_id"] == 8].copy()
    ground_truth = conditions["titer"].values
    
    predictions = fitted_model.predict(conditions)
    
    print("\nResults (First 10 timepoints):")
    result_df = pd.DataFrame({
        "Time": conditions["time"],
        "Feed": conditions["glucose_feed"],
        "Actual Titer": ground_truth,
        "Predicted Mean": predictions["titer_mean"],
        "Predicted Std": predictions["titer_std"]
    })
    print(result_df.head(10))
    
    # Calculate MSE
    mse = np.mean((result_df["Actual Titer"] - result_df["Predicted Mean"])**2)
    print(f"\nMean Squared Error: {mse:.4f}")

if __name__ == "__main__":
    main()
