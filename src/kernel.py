import torch
from typing import Callable, List
from tqdm import tqdm


class STLKernel:
    """STL kernel for measuring formula similarity via robustness semantics."""
    
    def __init__(
        self,
        trajectory_sampler: Callable,
        n_trajectories: int = 1000, 
        timed: bool = True,
        normalize: bool = True,
        device: str = 'cpu',
    ):
        self.trajectory_sampler = trajectory_sampler
        self.n_trajectories = n_trajectories
        self.timed = timed
        self.normalize = normalize
        self.device = torch.device(device)
    
    def _to_device(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        
        return tensor.to(self.device) if tensor.device != self.device else tensor
    
    def compute_robustness_matrix(
        self,
        formulae: List,
        trajectories: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Compute robustness matrix 
            R[i, j, t] = rho(phi_i, xi_j, t)
        for all formulae phi and trajectories xi.
        
        Args:
            formulae: List of n STL formulae
            trajectories: Tensor of shape [n_trajectories, n_vars, n_timesteps]
            verbose: Show progress bar
            
        Returns:
            R: Tensor of shape [n_formulae, n_trajectories, max_time_length]
        """
        n_formulae = len(formulae)
        n_trajectories = trajectories.shape[0]
        
        # First pass: compute all robustness values and find max time length
        robustness_list = []
        max_time_length = 0

        for formula in tqdm(formulae, disable=not verbose, desc="Computing robustness matrix"):
            with torch.no_grad():
                rho = formula.quantitative(
                    trajectories, 
                    evaluate_at_all_times=True,
                    normalize=self.normalize,
                )
                
            rho = self._to_device(rho).squeeze(1)  # [n_trajectories, time_steps]
            robustness_list.append(rho)
            max_time_length = max(max_time_length, rho.shape[1])
        
        # Second pass: pad to uniform size and stack
        R = torch.zeros(n_formulae, n_trajectories, max_time_length, device=self.device)
        
        for i, rho in enumerate(robustness_list):
            time_len = rho.shape[1]
            R[i, :, :time_len] = rho
        
        return R
    
    def compute_kernel_from_robustness(
        self, 
        R: torch.Tensor, 
        kernel_type: str = 'gaussian',
        sigma: float = 1.0,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Compute kernel matrix from pre-computed robustness matrix.
        
        Args:
            R: Robustness matrix of shape [n_formulae, n_trajectories, time_length]
            kernel_type: 'k_prime', 'k0', or 'gaussian'
            sigma: Bandwidth for Gaussian kernel
            verbose: Show progress
            
        Returns:
            K: Kernel matrix of shape [n_formulae, n_formulae]
        """
        n_formulae = R.shape[0]
        n_trajectories = R.shape[1]
        
        # Compute k' (base kernel) matrix
        if self.timed:
            # k'(φ_i, φ_j) = (1/n_trajectories) * Σ_samples Σ_time ρ_i * ρ_j * dt
            # R[i]: [n_trajectories, time] -> expand to [n_trajectories, time, 1]
            # R[j]: [n_trajectories, time] -> expand to [n_trajectories, 1, time]
            # Product: [n_trajectories, time, time] but we only need diagonal
            
            # Simpler: for each pair (i,j), compute dot product over time and average over samples
            K_prime = torch.zeros(n_formulae, n_formulae, device=self.device)
            
            for i in tqdm(range(n_formulae), disable=not verbose, desc="K' matrix"):
                for j in range(i, n_formulae):
                    # Find common time length
                    # Get non-zero lengths for each formula
                    len_i = (R[i].abs().sum(dim=0) > 0).sum().item()
                    len_j = (R[j].abs().sum(dim=0) > 0).sum().item()
                    common_len = min(len_i, len_j) if len_i > 0 and len_j > 0 else R.shape[2]
                    
                    # R[i, :, :common_len] * R[j, :, :common_len]: [n_trajectories, common_len]
                    product = R[i, :, :common_len] * R[j, :, :common_len]
                    # Sum over time, mean over samples
                    k_val = product.sum(dim=1).mean().item()
                    
                    K_prime[i, j] = K_prime[j, i] = k_val
        else:
            # Untimed: only use t=0
            # k'(φ_i, φ_j) = (1/n_trajectories) * Σ_samples ρ_i(t=0) * ρ_j(t=0)
            R_t0 = R[:, :, 0]  # [n_formulae, n_trajectories]
            # K'[i,j] = mean(R[i,:,0] * R[j,:,0])
            K_prime = torch.mm(R_t0, R_t0.t()) / n_trajectories  # [n_formulae, n_formulae]
        
        if kernel_type == 'k_prime':
            return K_prime.cpu()
        
        # Compute k0[i,j] = k'[i,j] / sqrt(k'[i,i] * k'[j,j])
        diag = torch.diag(K_prime).clamp(min=1e-10)
        normalizer = torch.sqrt(torch.outer(diag, diag))  # [n_formulae, n_formulae]
        K0 = K_prime / normalizer
        
        if kernel_type == 'k0':
            return K0.cpu()
        
        # Compute Gaussian kernel
        # k[i,j] = exp(-(1 - 2*k0[i,j]) / σ²)
        exponent = -(1 - 2 * K0) / (sigma ** 2)
        exponent = torch.clamp(exponent, -500, 500)  # Prevent overflow
        K_gauss = torch.exp(exponent)
        
        return K_gauss.cpu()
    
    def compute_kernel_matrix(
        self,
        formulae: List,
        kernel_type: str = 'gaussian',
        sigma: float = 1.0,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Compute kernel matrix efficiently by first computing robustness matrix.
        
        Args:
            formulae: List of STL formulae
            kernel_type: 'k_prime', 'k0', or 'gaussian'
            sigma: Bandwidth for Gaussian kernel
            verbose: Show progress
            
        Returns:
            K: Kernel matrix of shape [n_formulae, n_formulae]
        """

        trajectories, time_points = self.trajectory_sampler(self.n_trajectories)
        R = self.compute_robustness_matrix(formulae, trajectories, verbose=verbose)
        K = self.compute_kernel_from_robustness(R, kernel_type, sigma, verbose=verbose)
        
        return K
