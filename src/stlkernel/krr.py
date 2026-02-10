import torch
from typing import List, Optional, Callable
import numpy as np
from .kernel import STLKernel


class STLKernelRidgeRegression:
    """
    Scikit-learn style Kernel Ridge Regression for STL formulae.
    
    Example:
        >>> sampler = lambda n: (torch.randn(n, 2, 100), torch.linspace(0, 10, 100))
        >>> krr = STLKernelRidgeRegression(
        ...     trajectory_sampler=sampler,
        ...     alpha=0.1,
        ...     kernel_type='gaussian',
        ...     sigma=1.0
        ... )
        >>> krr.fit(formulae_train, y_train)
        >>> y_pred = krr.predict(formulae_test)
    """
    
    def __init__(
        self,
        trajectory_sampler: Callable,
        alpha: float = 1.0,
        kernel_type: str = 'gaussian',
        sigma: float = 1.0,
        n_trajectories: int = 1000,
        timed: bool = True,
        normalize: bool = True,
        device: str = 'cpu',
        verbose: bool = True,
    ):
        """
        Initialize STL Kernel Ridge Regression.
        
        Args:
            trajectory_sampler: Function that returns (trajectories, time_points)
            alpha: Regularization parameter (λ)
            kernel_type: Type of kernel ('k_prime', 'k0', 'gaussian')
            sigma: Bandwidth for Gaussian kernel
            n_trajectories: Number of trajectories to sample
            timed: Whether to use timed semantics
            normalize: Whether to normalize robustness values
            device: 'cpu' or 'cuda'
            verbose: Show progress bars
        """
        self.alpha = alpha
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.verbose = verbose
        self.device = torch.device(device)
        
        # Initialize STL kernel computer
        self.stl_kernel = STLKernel(
            trajectory_sampler=trajectory_sampler,
            n_trajectories=n_trajectories,
            timed=timed,
            normalize=normalize,
            device=device,
        )
        
        # Fitted attributes (set during fit)
        self.formulae_train_ = None
        self.y_train_ = None
        self.weights_ = None
        self.K_train_ = None
        self.R_train_ = None
        
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
        # Store training data
        self.formulae_train_ = formulae_train
        
        # Convert y_train to torch tensor
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()
        self.y_train_ = y_train.to(self.device)
        
        # Ensure correct shape
        if self.y_train_.dim() == 1:
            self.y_train_ = self.y_train_.unsqueeze(1)  # [n, 1]
        
        n_train = len(formulae_train)
        
        if self.verbose:
            print(f"Fitting KRR with {n_train} training formulae...")
        
        # Compute kernel matrix for training data
        self.K_train_ = self.stl_kernel.compute_kernel_matrix(
            formulae_train,
            kernel_type=self.kernel_type,
            sigma=self.sigma,
            verbose=self.verbose,
        ).to(self.device)
        
        # Solve (K + αI)w = y
        K_reg = self.K_train_ + self.alpha * torch.eye(n_train, device=self.device)
        self.weights_ = torch.linalg.solve(K_reg, self.y_train_)
        
        if self.verbose:
            print(f"Training complete. Weights shape: {self.weights_.shape}")
        
        return self
    
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
        if self.formulae_train_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        n_test = len(formulae_test)
        
        if self.verbose:
            print(f"Predicting for {n_test} test formulae...")
        
        # Sample trajectories
        trajectories, time_points = self.stl_kernel.trajectory_sampler(
            self.stl_kernel.n_trajectories
        )
        
        # Compute robustness for train and test formulae
        if self.verbose:
            print("Computing robustness matrices...")
        
        R_train = self.stl_kernel.compute_robustness_matrix(
            self.formulae_train_, 
            trajectories, 
            verbose=self.verbose
        )
        
        R_test = self.stl_kernel.compute_robustness_matrix(
            formulae_test,
            trajectories,
            verbose=self.verbose
        )
        
        # Compute cross-kernel K_test_train[i,j] between test[i] and train[j]
        if self.verbose:
            print("Computing test-train kernel matrix...")
        
        K_test_train = self._compute_cross_kernel(R_test, R_train)
        
        # Predictions: y_pred = K_test_train @ weights
        y_pred = K_test_train @ self.weights_
        
        # Remove extra dimension if single output
        if y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(1)
        
        return y_pred
    
    def _compute_cross_kernel(
        self,
        R_test: torch.Tensor,
        R_train: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-kernel matrix between test and train formulae.
        
        Args:
            R_test: Robustness matrix for test formulae [n_test, n_traj, time]
            R_train: Robustness matrix for train formulae [n_train, n_traj, time]
            
        Returns:
            K_cross: Cross-kernel matrix [n_test, n_train]
        """
        n_test = R_test.shape[0]
        n_train = R_train.shape[0]
        n_trajectories = R_test.shape[1]
        
        # Compute K_prime (base kernel)
        K_prime = torch.zeros(n_test, n_train, device=self.device)
        
        if self.stl_kernel.timed:
            for i in range(n_test):
                for j in range(n_train):
                    # Find common time length
                    len_i = (R_test[i].abs().sum(dim=0) > 0).sum().item()
                    len_j = (R_train[j].abs().sum(dim=0) > 0).sum().item()
                    common_len = min(len_i, len_j) if len_i > 0 and len_j > 0 else R_test.shape[2]
                    
                    # Product over time, mean over samples
                    product = R_test[i, :, :common_len] * R_train[j, :, :common_len]
                    K_prime[i, j] = product.sum(dim=1).mean().item()
        else:
            # Untimed: only use t=0
            R_test_t0 = R_test[:, :, 0]  # [n_test, n_traj]
            R_train_t0 = R_train[:, :, 0]  # [n_train, n_traj]
            K_prime = torch.mm(R_test_t0, R_train_t0.t()) / n_trajectories
        
        if self.kernel_type == 'k_prime':
            return K_prime
        
        # Compute normalization factors
        # For test formulae
        K_prime_test_diag = torch.zeros(n_test, device=self.device)
        for i in range(n_test):
            if self.stl_kernel.timed:
                len_i = (R_test[i].abs().sum(dim=0) > 0).sum().item()
                product = R_test[i, :, :len_i] * R_test[i, :, :len_i]
                K_prime_test_diag[i] = product.sum(dim=1).mean().item()
            else:
                K_prime_test_diag[i] = (R_test[i, :, 0] ** 2).mean().item()
        
        # For train formulae (use diagonal of K_train if available)
        K_prime_train_diag = torch.diag(self.K_train_) if self.kernel_type == 'k_prime' else None
        
        if K_prime_train_diag is None:
            K_prime_train_diag = torch.zeros(n_train, device=self.device)
            for j in range(n_train):
                if self.stl_kernel.timed:
                    len_j = (R_train[j].abs().sum(dim=0) > 0).sum().item()
                    product = R_train[j, :, :len_j] * R_train[j, :, :len_j]
                    K_prime_train_diag[j] = product.sum(dim=1).mean().item()
                else:
                    K_prime_train_diag[j] = (R_train[j, :, 0] ** 2).mean().item()
        
        # Normalize
        normalizer = torch.sqrt(
            K_prime_test_diag.unsqueeze(1) * K_prime_train_diag.unsqueeze(0)
        ).clamp(min=1e-10)
        K0 = K_prime / normalizer
        
        if self.kernel_type == 'k0':
            return K0
        
        # Gaussian kernel
        exponent = -(1 - 2 * K0) / (self.sigma ** 2)
        exponent = torch.clamp(exponent, -500, 500)
        K_gauss = torch.exp(exponent)
        
        return K_gauss
    
    def score(
        self, 
        formulae_test: List, 
        y_test: torch.Tensor,
    ) -> float:
        """
        Compute R² score on test data.
        
        Args:
            formulae_test: List of test formulae
            y_test: True target values
            
        Returns:
            score: R² score
        """
        y_pred = self.predict(formulae_test)
        
        # R² = 1 - SS_res / SS_tot
        ss_res = torch.sum((y_test - y_pred) ** 2)
        ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2