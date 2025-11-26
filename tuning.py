import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

# ==============================================================================
# 1. THE NEW OPTIMIZER: AMKGH (Adaptive Momentum Kurtosis Gated Hybrid)
# ==============================================================================
class AMKGH(torch.optim.Optimizer):
    """
    AMKGH: Adaptive Momentum Kurtosis Gated Hybrid.
    
    Mechanism:
    Uses Kurtosis (K_t) to dynamically interpolate between:
    1. Smart Direction (Adam-like) -> Used when K_t is Low (Stable)
    2. Dumb Direction (SGD-Momentum) -> Used when K_t is High (Unstable/Spiky)
    
    The mixing is controlled by the Gate Signal s_t = 1 / (1 + lambda * K).
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        beta4=None,           
        lambda_k=0.1,         # Gate Sensitivity (Controls how easily we switch to SGD)
        kurtosis_max=10.0,    # Clipping threshold for stability
        eps=1e-8,
    ):
        if beta4 is None:
            beta4 = betas[1] 

        defaults = dict(
            lr=lr,
            betas=betas,
            beta4=beta4,
            lambda_k=lambda_k,
            kurtosis_max=kurtosis_max,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            beta4 = group["beta4"]
            lambda_k = group["lambda_k"]
            kurtosis_max = group["kurtosis_max"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)  
                    state["v"] = torch.zeros_like(p.data)  
                    state["E4"] = torch.zeros_like(p.data) 

                m = state["m"]
                v = state["v"]
                E4 = state["E4"]

                state["step"] += 1
                t = state["step"]

                # --- 1. Update Moments ---
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Efficient fourth moment
                g2 = grad.pow(2)
                g4 = g2.pow(2)
                E4.mul_(beta4).add_(g4, alpha=1 - beta4)

                # --- 2. Bias Correction ---
                bias_corr1 = 1 - beta1**t
                bias_corr2 = 1 - beta2**t
                
                m_hat = m / bias_corr1
                v_hat = v / bias_corr2

                # --- 3. Kurtosis Gating Signal ---
                # K_t measures "spikiness" of the gradient
                K_t = E4 / (v_hat.pow(2) + eps)
                
                # Clip to prevent numerical explosion in the gate
                K_clipped = torch.clamp(K_t, max=kurtosis_max)

                # Gate Factor s_t:
                # if K is small (Stable) -> s_t approx 1.0 -> Use Adam
                # if K is large (Unstable) -> s_t approx 0.0 -> Use SGD
                s_t = 1.0 / (1.0 + lambda_k * K_clipped)

                # --- 4. Dual-Channel Directions ---
                
                # Channel A: Smart (Adam-like)
                denom = torch.sqrt(v_hat) + eps
                smart_dir = m_hat / denom

                # Channel B: Dumb (SGD-Momentum)
                # We ignore curvature here to push through noise/valleys
                dumb_dir = m 

                # --- 5. Gated Hybrid Fusion ---
                # Interpolate based on stability
                hybrid_dir = (s_t * smart_dir) + ((1.0 - s_t) * dumb_dir)

                # --- 6. Apply Update ---
                p.data.add_(hybrid_dir, alpha=-lr)

        return loss

# ==============================================================================
# 2. MODEL & DATA SETUP
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR = os.getcwd()
EPOCHS = 10
BATCH_SIZE = 128

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def get_mnist_loaders():
    # Using basic transform
    transform = transforms.ToTensor()
    
    # Check if data exists to avoid redownloading constantly
    train_dataset = datasets.MNIST(DIR, train=True, download=True, transform=transform)
    valid_dataset = datasets.MNIST(DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, valid_loader

# ==============================================================================
# 3. OPTUNA OBJECTIVE FUNCTION
# ==============================================================================
def objective(trial):
    # --- A. Define Search Space for AMKGH ---
    
    # 1. Learning Rate
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    
    # 2. Gate Sensitivity (lambda_k) - CRITICAL for AMKGH
    # Low lambda (0.01) means the gate stays open (Adam) even with some noise.
    # High lambda (10.0) means the gate shuts to SGD very easily.
    lambda_k = trial.suggest_float("lambda_k", 0.01, 10.0, log=True)
    
    # 3. Kurtosis Max (Safety Cap)
    kurtosis_max = trial.suggest_float("kurtosis_max", 5.0, 100.0)
    
    # 4. Betas (Adam-like momentum)
    beta1 = trial.suggest_float("beta1", 0.85, 0.99)
    beta2 = trial.suggest_float("beta2", 0.90, 0.999)
    
    # --- B. Setup Training ---
    model = MNISTNet().to(DEVICE)
    
    # Using AMKGH
    optimizer = AMKGH(
        model.parameters(), 
        lr=lr, 
        betas=(beta1, beta2),
        lambda_k=lambda_k,
        kurtosis_max=kurtosis_max
    )
    
    train_loader, valid_loader = get_mnist_loaders()

    # --- C. Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        # --- D. Validation & Pruning ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(valid_loader.dataset)

        # Report intermediate accuracy to Optuna
        trial.report(accuracy, epoch)

        # Handle Pruning (Stop training early if the curve looks bad)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print(f"--- Starting AMKGH (New Gated Hybrid) Hyperparameter Optimization on {DEVICE} ---")
    print("Searching for best Gate Sensitivity (lambda) and LR...")
    
    # Number of hours to run (e.g., 2 hours)
    HOURS_TO_RUN = 2
    TIMEOUT_SECONDS = 60 * 60 * HOURS_TO_RUN 

    study = optuna.create_study(
        study_name="amkgh_mnist_tuning", 
        storage="sqlite:///amkgh_tuning.db", 
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    try:
        study.optimize(objective, timeout=TIMEOUT_SECONDS)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Saving results...")

    # --- 5. Report Results ---
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "="*40)
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    if len(complete_trials) > 0:
        print("\nBest trial:")
        trial = study.best_trial

        print(f"  Value (Accuracy): {trial.value:.4f}")
        print("  Best Hyperparameters for AMKGH:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    print("="*40)