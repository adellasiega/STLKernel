import os 
import torch
import shutil
from tqdm import tqdm
from typing import Optional, List

class STLKernel:
    """
    STL kernel for measuring formula similarity via robustness semantics.
    """
    
    def __init__(
        self,
        trajectories: torch.Tensor,
        normalize_robustness: bool = True,
        timed: bool = True,
        dt: float = 1.0,
        kernel_type: str = 'gaussian',
        sigma: float = 1.0,
        device: str = 'cpu',
        cache_dir: str = None,
        verbose: bool = True,
    ):
        self.trajectories = trajectories.to(torch.device(device))
        self.normalize_robustness = normalize_robustness
        self.timed = timed
        self.dt = dt
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.device = torch.device(device)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.verbose = verbose
     
    
    def _compute_robustness(
        self,
        formulae: List,
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
        robustness_list = []
        max_time_length = 0
        
        for formula in tqdm(formulae, disable=not self.verbose, desc=f"Computing robustness matrix, timed: {self.timed}"):
            with torch.no_grad():
                rho = formula.quantitative(
                    self.trajectories, 
                    evaluate_at_all_times=self.timed,
                    normalize=self.normalize_robustness,
                )

            if self.timed: # rho = [n_trajectories, 1, n_timesteps] 
                rho = rho.squeeze(1)  # -> [n_trajectories, n_timesteps]
                robustness_list.append(rho)
                max_time_length = max(max_time_length, rho.shape[1])
            
            else: # rho = [n_trajectories]
                robustness_list.append(rho)

        if self.timed: # Pad to uniform size and stack
            n_trajectories = self.trajectories.shape[0]
            R_pad = torch.zeros(n_formulae, n_trajectories, max_time_length, device=self.device)
            
            for i, rho in enumerate(robustness_list):
                time_len = rho.shape[1]
                R_pad[i, :, :time_len] = rho
            
            return R_pad
        
        else:
            R = torch.stack(robustness_list)
            return R


    def apply_gaussian(
            self, 
            K0: torch.Tensor, 
            sigma: float
        ) -> torch.Tensor:
        """
        Applies Gaussian transformation to a precomputed kernel matrix K0.
        
        Args:
            K0: Normalized kernel matrix
            sigma: Bandwidth parameter
            
        Returns:
            K_gauss: Gaussian kernel matrix
        """
        return torch.exp(-(2 - 2 * K0) / (sigma ** 2))
    

    def _compute_kernel(
            self, 
            R1: torch.Tensor,
            R2: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Compute kernel matrix 
            k[i, j] = integral_xi integral_t rho(phi_i, xi, t) rho(phi_j, xi, t) dt dmu0
        
        Args:
            R1: Robustness tensor of shape [n1_formulae, n_trajectories, n1_timesteps]
                If only R1 is specified, the Gram matrix is computed.

            R2: Optional robustness tensor of shape [n2_formulae, n_trajectories, n2_timesteps]
                If R2 is specified, the Cross Kernel is computed.
                        
        Returns:
            K: Tensor of shape [n1_formulae, n2_formulae]
        """
        
        is_symmetric = R2 is None
        if is_symmetric:
            R2 = R1
        
        n_trajectories = R1.shape[1]

        # Compute K_prime
        if self.timed:
            # R1 [n_formulae, n_trajectories, n1_timesteps]
            # R2 [n_formulae, n_trajectories, n2_timesteps] 
            # Pad to match R1 and R2 time dimensions for matrix multiplication
            max_t = max(R1.shape[2], R2.shape[2])
            R1_p = torch.nn.functional.pad(R1, (0, max_t - R1.shape[2]))
            R2_p = torch.nn.functional.pad(R2, (0, max_t - R2.shape[2]))
            
            # Flatten to (n_formulae, n_trajectories * max_t)
            X1 = R1_p.reshape(R1.shape[0], -1)
            X2 = R2_p.reshape(R2.shape[0], -1)
            
            # K_prime[i,j] = (sum over time and trajectories) * dt / n_trajectories
            K_prime = (X1 @ X2.T) / n_trajectories * self.dt
        
        else:  # untimed R [n_formulae, n_trajectories] 
            K_prime = (R1 @ R2.T) / n_trajectories

        if self.kernel_type == 'k_prime':
            return K_prime

        # Compute K0
        if is_symmetric:
            d1 = torch.diag(K_prime)
            d2 = d1
        else:
            if self.timed:
                d1 = (R1_p.reshape(R1.shape[0], -1)**2).sum(dim=1) / n_trajectories * self.dt
                d2 = (R2_p.reshape(R2.shape[0], -1)**2).sum(dim=1) / n_trajectories * self.dt
            else:
                d1 = (R1**2).mean(dim=1)
                d2 = (R2**2).mean(dim=1)

        normalizer = torch.sqrt(torch.outer(d1, d2)).clamp(min=1e-10)
        K0 = K_prime / normalizer
        
        if self.kernel_type == 'k0':
            return K0
        
        # Compute K_gaussian
        K_gauss = self.apply_gaussian(K0, self.sigma)
        
        return K_gauss


    def compute_kernel_train(
        self,
        formulae_train: List,
    ) -> torch.Tensor:
        """
        Compute training kernel matrix.
        
        Args:
            formulae_train: List of training formulae
            
        Returns:
            Kernel matrix
        """
        
        cache_path = os.path.join(self.cache_dir, "R_train.pt")
        
        if os.path.isfile(cache_path):
            R_train = torch.load(cache_path, map_location=self.device)
    
        else:
            R_train = self._compute_robustness(formulae_train)
            torch.save(R_train, cache_path)

        gram_matrix = self._compute_kernel(R_train)
        return gram_matrix
    
    
    def compute_kernel_cross(
        self,
        formulae_test: List,
    ) -> torch.Tensor:
        """
        Compute cross kernel matrix between test and training formulae.
        
        Args:
            formulae_test: List of test formulae
            
        Returns:
            Cross kernel matrix
        """
        
        file_R_train = os.path.join(self.cache_dir, "R_train.pt")
        if os.path.isfile(file_R_train):
            R_train = torch.load(file_R_train, map_location=self.device)
        else:
            raise FileNotFoundError(f"No training cache found at {self.cache_dir}. Compute kernel for train formulae first!")
        
        file_R_test = os.path.join(self.cache_dir, "R_test.pt")
        if os.path.isfile(file_R_test):
            R_test = torch.load(file_R_test, map_location=self.device)
        else:
            R_test = self._compute_robustness(formulae_test)
            torch.save(R_test, file_R_test)
        
        cross_kernel = self._compute_kernel(R_test, R_train)
        
        return cross_kernel
    
    def remove_cache(self):
        shutil.rmtree(self.cache_dir)