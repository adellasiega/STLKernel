import torch
from typing import List
import numpy as np
from .kernel import STLKernel
from sklearn.model_selection import KFold
from itertools import product


class STLKernelRidgeRegression:
    """
    Scikit-learn style Kernel Ridge Regression for STL formulae.
    """
    
    def __init__(
        self,
        trajectories: torch.Tensor,
        alpha: float = 1e-3,
        normalize_robustness: bool = True,
        timed: bool = True,
        dt: float = 1.0,
        kernel_type: str = 'gaussian',
        sigma: float = 1.0,
        device: str = 'cpu',
        cache_dir: str = None,
        verbose: bool = True,
    ):
        """
        Initialize STL Kernel Ridge Regression.
        
        Args:
            trajectories: Trajectories to compute kernels
            alpha: Regularization parameter
            normalize_robustness: Whether to normalize robustness values
            timed: Whether to use timed semantics
            dt: time step of trajectories
            kernel_type: Type of kernel ('k_prime', 'k0', 'gaussian')
            sigma: Bandwidth for Gaussian kernel
            device: 'cpu' or 'cuda'
            cache_dir: Directory in which robustness matrixes are stored
            verbose: Show progress bars
        """
        self.alpha = alpha
        self.device = torch.device(device)
        
        self.stl_kernel = STLKernel(
            trajectories,
            normalize_robustness,
            timed,
            dt,
            kernel_type,
            sigma,
            device,
            cache_dir,
            verbose,
        )
        
        self.weights = None

    
    def fit(
        self, 
        formulae_train: List, 
        y_train: torch.Tensor,
    ) -> 'STLKernelRidgeRegression':
        """
        Fit the model.
        
        Args:
            formulae_train: List of n training STL formulae
            y_train: Target values of shape (n,) or (n, n_outputs)
            
        Returns:
            self: Fitted estimator
        """

        # Convert y_train to torch tensor
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()
        y_train = y_train.to(self.device)
        
        # Ensure correct shape
        if y_train.dim() == 1:
            y_train = y_train.unsqueeze(1)  # [n, 1]
        
        # Compute kernel matrix for training data
        K_train = self.stl_kernel.compute_kernel_train(formulae_train)
        n_formulae_train = len(formulae_train)
        
        K_reg = K_train + self.alpha * torch.eye(n_formulae_train, device=self.device)
        self.weights = torch.linalg.solve(K_reg, y_train)
        
        return self

    def remove_cache(self):
        self.stl_kernel.remove_cache()
    
    
    def predict(
        self, 
        formulae_test: List,
    ) -> torch.Tensor:
        """
        Predict target values for test formulae.
        
        Args:
            formulae_test: List of m test STL formulae
            
        Returns:
            y_pred: Predictions of shape (m,) or (m, n_outputs)
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        K_test_train = self.stl_kernel.compute_kernel_cross(formulae_test)
        
        y_pred = K_test_train @ self.weights
        
        # Remove extra dimension if single output
        if y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(1)
        
        return y_pred

def kfoldcv(
        trajectories, 
        formulae, 
        y,
        kernel_type,
        alphas, 
        sigmas, 
        n_splits=3,
        cache_dir = None,
        device='cpu'
    ):
    """
    Performs K-Fold Cross Validation to tune alpha and sigma.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_loss = float('inf')
    best_params = None

    # Ensure y is accessible by index
    if isinstance(y, list):
        y = np.array(y)

    print(f"Starting Grid Search on {len(sigmas)*len(alphas)} combinations...")

    if len(formulae):
        formulae=formulae[:500]

    for sigma, alpha in product(sigmas, alphas):
        fold_losses = []
        
        for train_idx, val_idx in kf.split(formulae):
            # Split Data
            f_train = [formulae[i] for i in train_idx]
            f_val   = [formulae[i] for i in val_idx]
            y_train = y[train_idx]
            y_val   = torch.as_tensor(y[val_idx], device=device).float()

            # Instantiate and Fit
            model = STLKernelRidgeRegression(
                trajectories=trajectories,
                alpha=alpha,
                sigma=sigma,
                device=device,
                kernel_type = kernel_type,
                cache_dir=cache_dir,
                verbose=False,
            )
            model.fit(f_train, y_train)
            
            # Validate
            preds = model.predict(f_val)
            
            # Handle dimension mismatch if necessary
            if preds.shape != y_val.shape:
                preds = preds.view_as(y_val)
                
            loss = torch.nn.functional.mse_loss(preds, y_val).item()
            fold_losses.append(loss)

            model.remove_cache()

        avg_loss = np.mean(fold_losses)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {'alpha': alpha, 'sigma': sigma}
            
    print(f"Best Params: {best_params} | MSE: {best_loss:.5f}")
    return best_params