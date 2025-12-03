import torch
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
from .core import Parameter, Feed, Output

class DataProcessor:
    def __init__(self, parameters: Dict[str, Parameter], feeds: Dict[str, Feed], outputs: Dict[str, Output]):
        self.parameters = parameters
        self.feeds = feeds
        self.outputs = outputs
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        
    def process_feeds(self, df: pd.DataFrame, time_col: str, run_col: str) -> pd.DataFrame:
        df = df.copy()
        # Sort by run and time
        df = df.sort_values([run_col, time_col])
        
        for feed_name in self.feeds:
            # We want to smooth the feed events to create a continuous feature.
            # 1. Extract the feed column (spikes)
            # 2. Apply Gaussian smoothing to spread the spike
            # 3. Calculate cumulative sum of the smoothed signal
            
            def smooth_and_cumsum(group):
                feed_signal = group[feed_name].values.astype(float)
                smoothed_feed = gaussian_filter1d(feed_signal, sigma=1.0)
                return pd.Series(np.cumsum(smoothed_feed), index=group.index)

            # Use include_groups=False to avoid deprecation warning and potential issues
            # We need to ensure the index aligns correctly
            smoothed_series = df.groupby(run_col, group_keys=False).apply(smooth_and_cumsum)
            
            # Ensure smoothed_series is a Series not a DataFrame
            if isinstance(smoothed_series, pd.DataFrame):
                 # If it's a DataFrame, take the first column
                 smoothed_series = smoothed_series.iloc[:, 0]
            
            # Ensure it's a Series with the same index
            if not isinstance(smoothed_series, pd.Series):
                smoothed_series = pd.Series(smoothed_series, index=df.index)
            else:
                smoothed_series = smoothed_series.reindex(df.index)
            
            # Fill NaNs if any (reindexing might introduce them if indices don't match)
            if smoothed_series.isna().any():
                # print(f"DEBUG: NaNs found in smoothed_series for {feed_name}")
                smoothed_series = smoothed_series.fillna(0.0)

            df[f"cumulative_{feed_name}"] = smoothed_series
            
        return df

    def fit_transform(self, df: pd.DataFrame, time_col: str = "time", run_col: str = "run_id") -> Tuple[torch.Tensor, torch.Tensor]:
        # Process feeds
        df_processed = self.process_feeds(df, time_col, run_col)
        
        # Extract X and y
        x_cols = [time_col] + list(self.parameters.keys()) + [f"cumulative_{f}" for f in self.feeds]
        y_cols = list(self.outputs.keys())
        
        X = df_processed[x_cols].values
        y = df_processed[y_cols].values
        
        # Normalize
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0) + 1e-6 # Avoid division by zero
        
        self.y_mean = np.mean(y, axis=0)
        self.y_std = np.std(y, axis=0) + 1e-6
        
        X_norm = (X - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        
        # Return 1D tensor for y if it's a single output
        return torch.tensor(X_norm).float(), torch.tensor(y_norm).float().squeeze()
        
    def transform(self, df: pd.DataFrame, time_col: str = "time", run_col: str = "run_id") -> torch.Tensor:
        # Used for prediction
        df_processed = self.process_feeds(df, time_col, run_col)
        
        x_cols = [time_col] + list(self.parameters.keys()) + [f"cumulative_{f}" for f in self.feeds]
        
        X = df_processed[x_cols].values
        X_norm = (X - self.x_mean) / self.x_std
        
        return torch.tensor(X_norm).float()
        
    def inverse_transform_y(self, y_norm: torch.Tensor) -> np.ndarray:
        y_norm_np = y_norm.detach().numpy()
        return y_norm_np * self.y_std + self.y_mean
