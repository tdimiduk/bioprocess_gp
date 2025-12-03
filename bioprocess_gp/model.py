import torch
import gpytorch

class ManagedGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ManagedGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(model, likelihood, train_x, train_y, training_iter=50, lr=0.1):
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        # Ensure target is 1D for ExactGP if it's not multitask
        if train_y.dim() > 1:
            train_y_flat = train_y.squeeze()
        else:
            train_y_flat = train_y
            
        loss = -mll(output, train_y_flat)
        loss.backward()
        optimizer.step()
        
    model.eval()
    likelihood.eval()
    return model, likelihood
