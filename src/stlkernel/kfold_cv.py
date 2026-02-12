from .krr import STLKernelRidgeRegression
from sklearn.model_selection import KFold
from itertools import product
import torch
import numpy as np

def tune_stl_krr(
        trajectories, 
        formulae, 
        y,
        kernel_type,
        alphas, 
        sigmas, 
        n_splits=5,
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

    if len(formulae > 500):
        formulae=formulae[:500]

    for sigma, alpha in product(sigmas, alphas):
        fold_losses = []
        
        for train_idx, val_idx in kf.split(formulae):
            # 1. Split Data
            f_train = [formulae[i] for i in train_idx]
            f_val   = [formulae[i] for i in val_idx]
            y_train = y[train_idx]
            y_val   = torch.as_tensor(y[val_idx], device=device).float()

            # 2. Instantiate and Fit
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
            
            # 3. Validate
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

# Usage Example:
# best_p = tune_stl_krr(
#     trajectories=my_trajs,
#     formulae=my_formulae, 
#     y=my_scores, 
#     alphas=[1e-3, 1e-2, 0.1, 1.0], 
#     sigmas=[0.1, 0.5, 1.0, 2.0]
# )