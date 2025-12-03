from __future__ import annotations
import pandas as pd
import torch
import gpytorch
import numpy as np
from typing import Optional, List, Dict, Union
from dataclasses import dataclass, field
from .core import Parameter, Output, Feed, Normal, LogNormal, Uniform
from .model import ManagedGP, train_gp
from .data import DataProcessor

@dataclass
class FittedBioprocessModel:
    definition: "BioprocessModel"
    processor: DataProcessor
    model: ManagedGP
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    outputs: List[Output]

    def predict(self, conditions: Union[pd.DataFrame, List[Dict]], time_col: str = "time", run_col: str = "run_id"):
        if isinstance(conditions, list):
            conditions = pd.DataFrame(conditions)
            
        # Transform input
        test_x = self.processor.transform(conditions, time_col, run_col)
        
        # Predict
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            
        # Get mean and std (normalized)
        mean_norm = observed_pred.mean
        std_norm = observed_pred.stddev
        
        # Inverse transform mean
        # Note: Inverse transform assumes y was shape (N, D), but GP output is (N,) if 1D.
        # We need to handle dimensions carefully.
        # For now assuming 1 output.
        mean_real = self.processor.inverse_transform_y(mean_norm.unsqueeze(-1))
        # Std is scaled by y_std
        std_real = std_norm.detach().numpy() * self.processor.y_std
        
        # Construct result DataFrame
        result = conditions.copy()
        output_name = self.outputs[0].name # Assuming single output for now
        result[f"{output_name}_mean"] = mean_real.flatten()
        result[f"{output_name}_std"] = std_real.flatten()
        
        return result

@dataclass
class BioprocessModel:
    parameters: List[Parameter] = field(default_factory=list)
    feeds: List[Feed] = field(default_factory=list)
    outputs: List[Output] = field(default_factory=list)
    
    def fit(self, data: Union[pd.DataFrame, List[Dict]], time_col: str = "time", run_col: str = "run_id", training_iter=50) -> FittedBioprocessModel:
        if isinstance(data, list):
            data = pd.DataFrame(data)
            
        # Convert lists to dicts for DataProcessor
        param_dict = {p.name: p for p in self.parameters}
        feed_dict = {f.name: f for f in self.feeds}
        output_dict = {o.name: o for o in self.outputs}
            
        processor = DataProcessor(param_dict, feed_dict, output_dict)
        train_x, train_y = processor.fit_transform(data, time_col, run_col)
        
        # Initialize Likelihood and Model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ManagedGP(train_x, train_y, likelihood)
        
        # Train
        model, likelihood = train_gp(model, likelihood, train_x, train_y, training_iter=training_iter)
        
        return FittedBioprocessModel(
            definition=self,
            processor=processor,
            model=model,
            likelihood=likelihood,
            outputs=self.outputs
        )
