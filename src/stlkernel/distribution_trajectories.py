import torch
import pandas as pd
from typing import Tuple


class Mu0:
    def __init__(
        self,
        a: float = 0.0,
        b: float = 100.0,
        delta: float = 1.0,
        m_prime: float = 0.0,
        sigma_prime: float = 1.0,
        m_double_prime: float = 0.0,
        sigma_double_prime: float = 1.0,
        q: float = 0.1,
        standardize: bool = True,
        device: str = "cuda",
    ):
        self.a = a
        self.b = b
        self.delta = delta
        self.m_prime = m_prime
        self.sigma_prime = sigma_prime
        self.m_double_prime = m_double_prime
        self.sigma_double_prime = sigma_double_prime
        self.q = q
        self.standardize = standardize
        self.device = device

        self.N = int((b - a) / delta)
        self.time_points = torch.linspace(a, b, self.N + 1, device=device)

    def sample(
        self,
        n_trajectories: int,
        n_vars: int,
        partition: str = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            n_trajectories: number of trajectories
            n_vars: dimension of each trajectory (number of variables)

        Returns:
            trajectories: [n_trajectories, n_vars, N+1]
            time_points: [N+1]
        """

        device = self.device
        N = self.N

        xi_0 = (
            torch.randn(n_trajectories, n_vars, device=device)
            * self.sigma_prime
            + self.m_prime
        )

        K_sample = (
            torch.randn(n_trajectories, n_vars, device=device)
            * self.sigma_double_prime
            + self.m_double_prime
        )
        K = K_sample.pow(2)

        y = torch.rand(n_trajectories, n_vars, N - 1, device=device) * K[..., None]
        y_sorted, _ = torch.sort(y, dim=-1)

        y_all = torch.cat(
            [
                torch.zeros(n_trajectories, n_vars, 1, device=device),
                y_sorted,
                K[..., None],
            ],
            dim=-1,
        )

        flips = torch.where(
            torch.rand(n_trajectories, n_vars, N, device=device) < self.q,
            -1.0,
            1.0,
        )

        s0 = torch.where(
            torch.rand(n_trajectories, n_vars, 1, device=device) > 0.5,
            1.0,
            -1.0,
        )

        signs = torch.cat([s0, flips], dim=-1)
        signs = torch.cumprod(signs, dim=-1)[..., :N]

        increments = signs * (y_all[..., 1:] - y_all[..., :-1])

        trajectories = torch.zeros(
            n_trajectories, n_vars, N + 1, device=device
        )
        trajectories[..., 0] = xi_0
        trajectories[..., 1:] = xi_0[..., None] + torch.cumsum(
            increments, dim=-1
        )

        if self.standardize:
            mean = torch.mean(trajectories, dim=(0,2), keepdim=True)
            std = torch.std(trajectories, dim=(0,2), keepdim=True) + 10e-6
            trajectories = (trajectories - mean)/std

        return trajectories, self.time_points


class SDE:
    def __init__(
        self,
        drift,
        diffusion,
        a: float = 0.0,
        b: float = 100.0,
        delta: float = 1.0,
        m_prime: float = 0.0,
        sigma_prime: float = 1.0,
        standardize: bool = True,
        device: str = "cuda",
    ):
        """
        Args:
            drift(x, t): drift function f(x, t)
            diffusion(x, t): diffusion function g(x, t)
        """
        self.drift = drift
        self.diffusion = diffusion
        self.a = a
        self.b = b
        self.delta = delta
        self.m_prime = 0.0
        self.sigma_prime = 1.0
        self.standardize = standardize
        self.device = device

        self.N = int((b - a) / delta)
        self.time_points = torch.linspace(a, b, self.N + 1, device=device)

    def sample(
        self,
        n_trajectories: int,
        n_vars: int,
        partition: str = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            trajectories: [n_trajectories, n_vars, N+1]
            time_points: [N+1]
        """
        device = self.device
        dt = self.delta
        
        x = (
            torch.randn(n_trajectories, n_vars, device=device)
            * self.sigma_prime
            + self.m_prime
        )
        
        trajectories = torch.zeros(
            n_trajectories, n_vars, self.N + 1, device=device
        )
        trajectories[..., 0] = x

        for i in range(self.N):
            t = self.time_points[i]
            noise = torch.randn_like(x)

            drift_term = self.drift(x, t) * dt
            diffusion_term = self.diffusion(x, t) * torch.sqrt(
                torch.tensor(dt, device=device)
            ) * noise

            x = x + drift_term + diffusion_term
            trajectories[..., i + 1] = x

        if self.standardize:
            mean = torch.mean(trajectories, dim=(0,2), keepdim=True)
            std = torch.std(trajectories, dim=(0,2), keepdim=True) + 10e-6
            trajectories = (trajectories - mean)/std
        
        return trajectories, self.time_points


class WhiteNoise:
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        a: float = 0.0,
        b: float = 100.0,
        n_steps: int = 101,
        standardize: bool = True,
        device: str = "cuda",
    ):
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b
        self.n_steps = n_steps
        self.standardize = standardize
        self.device = device

        # time_points: [101] from 0 to 100
        self.time_points = torch.linspace(a, b, n_steps, device=device)

    def sample(
        self,
        n_trajectories: int,
        n_vars: int,
        partition: str = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            trajectories: [n_trajectories, n_vars, 101]
            time_points: [101]
        """
        # Directly sample the full tensor: i.i.d. Gaussian noise
        trajectories = (
            torch.randn(n_trajectories, n_vars, self.n_steps, device=self.device) 
            * self.std 
            + self.mean
        )

        if self.standardize:
            mean = torch.mean(trajectories, dim=(0, 2), keepdim=True)
            std = torch.std(trajectories, dim=(0, 2), keepdim=True) + 1e-6
            trajectories = (trajectories - mean) / std
        
        return trajectories, self.time_points


class RealDataDistribution:
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        a: float = 0.0,
        b: float = 100.0,
        delta: float = 1.0,
        standardize: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        self.standardize = standardize
        self.target_T = int((b-a)/delta)+1
        
        # 1. Load data as list of tensors if lengths vary, or a single tensor if padded
        # Assuming df.values works, but we apply interpolation to ensure target_T
        data_train_raw = torch.tensor(df_train.values, dtype=torch.float32, device=device).unsqueeze(1)
        data_test_raw = torch.tensor(df_test.values, dtype=torch.float32, device=device).unsqueeze(1)

        # Upsample/Downsample to target_T (101)
        # linear mode is best for time series; align_corners=True preserves endpoints 0 and 100
        self.data_train = torch.nn.functional.interpolate(data_train_raw, size=self.target_T, mode='linear', align_corners=True)
        self.data_test = torch.nn.functional.interpolate(data_test_raw, size=self.target_T, mode='linear', align_corners=True)

        self.n_trajectories_train = self.data_train.shape[0]
        self.n_trajectories_test = self.data_test.shape[0]
        self.n_vars = 1

        # 3. Time points from 0 to 100 with exactly target_T steps
        self.time_points = torch.linspace(0, 100, self.target_T, device=device)

    def sample(
        self,
        n_trajectories: int,
        n_vars: int = 1,
        partition: str = "train"
        ):
        
        data = self.data_train if partition == 'train' else self.data_test
        n_partition = self.n_trajectories_train if partition == 'train' else self.n_trajectories_test
        
        idx = torch.randint(0, n_partition, (n_trajectories,), device=self.device)
        trajectories = data[idx] # shape: [n_trajectories, 1, 101]

        if self.standardize:
            mean = trajectories.mean(dim=(0,2), keepdim=True)
            std = trajectories.std(dim=(0,2), keepdim=True) + 1e-6
            trajectories = (trajectories - mean) / std

        return trajectories, self.time_points


class FourierDistribution:
    def __init__(
        self,
        a: float = 0.0,
        b: float = 100.0,
        delta: float = 1.0,
        n_harmonics: int = 5,   
        omega_mean: float = 0.1,
        omega_std: float = 0.02,
        a0_mean: float = 0.0,
        a0_std: float = 0.5,
        coef_std: float = 1.0,
        standardize: bool = True,
        device: str = "cuda",
    ):
        self.a = a
        self.b = b
        self.delta = delta
        self.n_harmonics = n_harmonics
        self.omega_mean = omega_mean
        self.omega_std = omega_std
        self.a0_mean = a0_mean
        self.a0_std = a0_std
        self.coef_std = coef_std
        self.standardize = standardize
        self.device = device

        self.N = int((b - a) / delta)
        self.time_points = torch.linspace(a, b, self.N + 1, device=device)

    def sample(
        self,
        n_trajectories: int,
        n_vars: int,
        partition: str = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            trajectories: [n_trajectories, n_vars, N+1]
            time_points: [N+1]
        """

        device = self.device
        t = self.time_points  # [N+1]

        # ---- Sample parameters ----
        # base frequencies (positive)
        omega = torch.abs(
            torch.randn(n_trajectories, n_vars, device=device) * self.omega_std
            + self.omega_mean
        )  # [T, V]

        # DC offsets
        a0 = (
            torch.randn(n_trajectories, n_vars, device=device) * self.a0_std
            + self.a0_mean
        )  # [T, V]

        # Fourier coefficients
        a_n = torch.randn(
            n_trajectories, n_vars, self.n_harmonics, device=device
        ) * self.coef_std

        b_n = torch.randn(
            n_trajectories, n_vars, self.n_harmonics, device=device
        ) * self.coef_std

        # ---- Build series ----
        # shape handling
        t = t[None, None, None, :]                          # [1,1,1,N+1]
        omega = omega[..., None, None]                      # [T,V,1,1]
        n = torch.arange(1, self.n_harmonics + 1, device=device)[None, None, :, None]
        # n shape: [1,1,K,1]

        phase = n * omega * t                                # [T,V,K,N+1]

        cos_terms = a_n[..., None] * torch.cos(phase)        # [T,V,K,N+1]
        sin_terms = b_n[..., None] * torch.sin(phase)        # [T,V,K,N+1]

        series = cos_terms + sin_terms                       # [T,V,K,N+1]
        x = a0[..., None] + series.sum(dim=2)                # [T,V,N+1]

        if self.standardize:
            mean = torch.mean(x, dim=(0,2), keepdim=True)
            std = torch.std(x, dim=(0,2), keepdim=True) + 1e-6
            x = (x - mean) / std

        return x, self.time_points
    

class Mixture:
    def __init__(
        self,
        a: float = 0.0,
        b: float = 100.0,
        delta: float = 1.0,
        # Mu0 parameters
        mu0_m_prime: float = 0.0,
        mu0_sigma_prime: float = 1.0,
        mu0_m_double_prime: float = 0.0,
        mu0_sigma_double_prime: float = 1.0,
        mu0_q: float = 0.1,
        # Fourier parameters
        fourier_n_harmonics: int = 5,
        fourier_omega_mean: float = 0.1,
        fourier_omega_std: float = 0.02,
        fourier_a0_mean: float = 0.0,
        fourier_a0_std: float = 0.5,
        fourier_coef_std: float = 1.0,
        # SDE (Ornstein-Uhlenbeck) parameters
        ou_drift_coef: float = -0.1,
        ou_diffusion_coef: float = 0.1,
        ou_m_prime: float = 0.0,
        ou_sigma_prime: float = 1.0,
        # Common parameters
        standardize: bool = True,
        device: str = "cuda",
    ):
        """
        Mixture distribution sampling 33% from Mu0, 33% from Fourier, 33% from Ornstein-Uhlenbeck SDE.
        
        Args:
            a, b, delta: Time domain parameters
            mu0_*: Parameters for Mu0 distribution
            fourier_*: Parameters for Fourier distribution
            ou_*: Parameters for Ornstein-Uhlenbeck SDE
            standardize: Whether to standardize trajectories
            device: Device to use ('cuda' or 'cpu')
        """
        self.a = a
        self.b = b
        self.delta = delta
        self.standardize = standardize
        self.device = device
        
        self.N = int((b - a) / delta)
        self.time_points = torch.linspace(a, b, self.N + 1, device=device)
        
        # Initialize Mu0 distribution
        self.mu0 = Mu0(
            a=a,
            b=b,
            delta=delta,
            m_prime=mu0_m_prime,
            sigma_prime=mu0_sigma_prime,
            m_double_prime=mu0_m_double_prime,
            sigma_double_prime=mu0_sigma_double_prime,
            q=mu0_q,
            standardize=False,  # We'll standardize at the end
            device=device,
        )
        
        # Initialize Fourier distribution
        self.fourier = FourierDistribution(
            a=a,
            b=b,
            delta=delta,
            n_harmonics=fourier_n_harmonics,
            omega_mean=fourier_omega_mean,
            omega_std=fourier_omega_std,
            a0_mean=fourier_a0_mean,
            a0_std=fourier_a0_std,
            coef_std=fourier_coef_std,
            standardize=False,  # We'll standardize at the end
            device=device,
        )
        
        # Initialize Ornstein-Uhlenbeck SDE
        # drift(x, t) = -0.1 * x
        # diffusion(x, t) = 0.1
        def ou_drift(x, t):
            return ou_drift_coef * x
        
        def ou_diffusion(x, t):
            return ou_diffusion_coef * torch.ones_like(x)
        
        self.ou = SDE(
            drift=ou_drift,
            diffusion=ou_diffusion,
            a=a,
            b=b,
            delta=delta,
            m_prime=ou_m_prime,
            sigma_prime=ou_sigma_prime,
            standardize=False,  # We'll standardize at the end
            device=device,
        )
    
    def sample(
        self,
        n_trajectories: int,
        n_vars: int,
        partition: str = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample trajectories from mixture distribution.
        
        Args:
            n_trajectories: Total number of trajectories to sample
            n_vars: Dimension of each trajectory
            partition: 'train' or 'test' (not used, for compatibility)
            
        Returns:
            trajectories: [n_trajectories, n_vars, N+1]
            time_points: [N+1]
        """
        device = self.device
        
        # Calculate how many samples from each distribution
        # 33% each, handle remainder by giving extras to first distribution
        n_mu0 = n_trajectories // 3
        n_fourier = n_trajectories // 3
        n_ou = n_trajectories - n_mu0 - n_fourier  # Gets remainder
        
        # Sample from each distribution
        traj_mu0, _ = self.mu0.sample(n_mu0, n_vars, partition)
        traj_fourier, _ = self.fourier.sample(n_fourier, n_vars, partition)
        traj_ou, _ = self.ou.sample(n_ou, n_vars, partition)
        
        # Concatenate all trajectories
        trajectories = torch.cat([traj_mu0, traj_fourier, traj_ou], dim=0)
        
        # Shuffle to mix the distributions
        perm = torch.randperm(n_trajectories, device=device)
        trajectories = trajectories[perm]
        
        # Apply standardization if requested
        if self.standardize:
            mean = torch.mean(trajectories, dim=(0, 2), keepdim=True)
            std = torch.std(trajectories, dim=(0, 2), keepdim=True) + 1e-6
            trajectories = (trajectories - mean) / std
        
        return trajectories, self.time_points


class VAEDistribution:
    def __init__(
        self,
        model_path: str,
        model_class,
        a: float = 0.0,
        b: float = 100.0,
        delta: float = 1.0,
        latent_dim: int = 16,
        standardize: bool = True,
        device: str = "cuda"
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.seq_len = int((b-a)/delta)+1
        self.standardize = standardize
        
        # Load model
        self.model = model_class(latent_dim=latent_dim).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Match time points to [a,b] range
        self.time_points = torch.linspace(a, b, self.seq_len, device=device)

    def sample(
        self,
        n_trajectories: int,
        n_vars: int = 1,
        partition = None, 
    ) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(n_trajectories * n_vars, self.latent_dim).to(self.device)
            
            # Map latent to hidden state and repeat for target sequence length
            # .repeat ensures the LSTM runs for exactly self.seq_len steps
            z_ext = self.model.decoder_fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
            
            samples, _ = self.model.decoder_lstm(z_ext)
            x = self.model.output_layer(samples) # [batch, seq_len, 1]
            
            x = x.view(n_trajectories, n_vars, self.seq_len)
            
            if self.standardize:
                mean = x.mean(dim=(0,2), keepdim=True)
                std = x.std(dim=(0,2), keepdim=True) + 1e-6
                x = (x - mean) / std
            
        return x, self.time_points