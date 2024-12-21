import torch
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from model import SimpleNet, SimpleNet2
import torch
import numpy as np


# -------------------------
# Data Generation
# -------------------------
# Mixture of Gaussians: 50% N(-2,1), 50% N(2,1)
def sample_x(batch_size):
    return torch.randn(batch_size).unsqueeze(1)
    mix = torch.rand(batch_size)
    means = torch.where(mix < 0.5, torch.full_like(mix, -2.0), torch.full_like(mix, 2.0))
    x = means + torch.randn(batch_size)
    return x.unsqueeze(1)  # shape [batch_size, 1]


def sample_x(batch_size):
    """
    Draws samples from a mixture of Gaussians (MoG) distribution.

    The MoG has 8 components with the following parameters:
      - Component 1: mean = -5.0, std = 0.8, weight = 0.1
      - Component 2: mean = -3.5, std = 0.7, weight = 0.1
      - Component 3: mean = -2.0, std = 0.5, weight = 0.15
      - Component 4: mean = -0.5, std = 0.6, weight = 0.15
      - Component 5: mean = 1.0, std = 0.4, weight = 0.1
      - Component 6: mean = 2.5, std = 0.9, weight = 0.1
      - Component 7: mean = 4.0, std = 1.0, weight = 0.15
      - Component 8: mean = 5.5, std = 1.2, weight = 0.15

    Args:
        batch_size (int): Number of samples to generate.

    Returns:
        torch.Tensor: Samples of shape (batch_size, 1).
    """
    # Define the parameters of the MoG
    #means = torch.tensor([-5.0, -3.5, -2.0, -0.5, 1.0, 2.5, 4.0, 5.5])  # Means of the Gaussians
    #stds = torch.tensor([0.8, 0.7, 0.5, 0.6, 0.4, 0.9, 1.0, 1.2])      # Standard deviations of the Gaussians
    #weights = torch.tensor([0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.15, 0.15])  # Weights of the Gaussians

    means = torch.tensor([-5.0, -3.5, -2.0, -0.5, 1.0, 2.5, 4.0, 5.5])  # Means of the Gaussians
    stds = torch.tensor([0.8, 0.7, 0.5, 0.6, 0.4, 0.9, 1.0, 1.2])      # Standard deviations of the Gaussians
    weights = torch.tensor([0.1, 0.1, 0.15*2, 0.15*2, 0.1, 0.1, 0, 0])  # Weights of the Gaussians

    # Normalize weights to ensure they sum to 1
    weights = F.softmax(weights, dim=0)

    # Sample component indices according to the weights
    component_indices = torch.multinomial(weights, batch_size, replacement=True)

    # Draw samples for each component
    samples = torch.normal(means[component_indices], stds[component_indices])

    # Reshape to (B, 1)
    return samples.unsqueeze(1)

def sample_e(batch_size):
    # standard normal noise
    return torch.randn(batch_size, 1)

def sample_t(batch_size):
    # uniform in [0, 1]
    batch = torch.rand(batch_size, 1)
    return batch

plt.hist(sample_x(10000), bins=100, density=True, alpha=0.7, color='blue')
plt.xlabel("x_0")
plt.ylabel("Probability Density")
plt.title("Samples from the Diffusion Model (Reconstructed from Noise)")
plt.grid(True)
#plt.show()
plt.savefig('ori_sample.png')

def sample_t(batch_size):
    timesteps = torch.linspace(0, 1, 50)

    # Parameters
    #batch_size = 400

    # Randomly sample indices
    indices = torch.randint(0, len(timesteps), (batch_size,))

    # Draw random samples
    batch = timesteps[indices].unsqueeze(1)
    #print(batch.shape)

    return batch

# -------------------------
# Training Utilities
# -------------------------
def gaussian_pdf(x, mean=0.0, std=1.0):
    # PDF of a Gaussian N(mean, std^2)
    return (1/(std * (2*torch.pi)**0.5))*torch.exp(-0.5*((x-mean)/std)**2)

def train_baseline(model, optimizer, steps=1000, batch_size=6400):
    model.train()
    losses = []
    criterion = nn.MSELoss()
    for step in tqdm(range(steps)):
        optimizer.zero_grad()

        x = sample_x(batch_size)   # x0
        e = sample_e(batch_size)
        t = sample_t(batch_size)


        x = x.cuda()
        e = e.cuda()
        t = t.cuda()

        xt = (1 - t) * x + t * e

        # Compute v = e - x0
        v = e - x

        # Predict v
        v_pred = model(xt, t)

        loss = criterion(v_pred, v)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses

# -------------------------
# Improved Training
# -------------------------
def train_improved(model, optimizer, steps=1000, batch_size=640):
    model.train()
    losses = []
    criterion = nn.MSELoss()
    for step in range(steps):
        optimizer.zero_grad()

        # Sample the minibatch
        x_batch = sample_x(batch_size)  # x0 for each sample
        e_batch = sample_e(batch_size)
        t_batch = sample_t(batch_size)

        x_batch = x.cuda()
        e_batch = e.cuda()
        t_batch = t.cuda()

        # Compute xt for each sample
        xt_batch = (1 - t_batch)*x_batch + t_batch*e_batch

        # For each xt in the batch, find the candidate e and corresponding x that
        # yields the highest pdf under N(0,1).

        # Expand for broadcasting:
        xt_expanded = xt_batch.unsqueeze(1)  # [B, 1, 1]
        x_expanded = x_batch.unsqueeze(0)    # [1, B, 1]
        t_expanded = t_batch.unsqueeze(1)    # [B, 1, 1]

        # Candidate e for (xt[i], x[j]):
        # e_candidate[i, j] = (xt[i] - (1-t[i])*x[j]) / t[i]
        e_candidates = (xt_expanded - (1 - t_expanded)*x_expanded) / t_expanded

        # Compute pdf for these candidates under N(0,1)
        pdf_values = gaussian_pdf(e_candidates)  # [B, B, 1]

        pdf_values = pdf_values.squeeze(-1) # [B, B]

        # Chosen e for each sample i
        chosen_e = e_candidates[torch.arange(batch_size), max_indices] # [B, 1]
        # Corresponding x0 for each chosen pair:
        chosen_x = x_batch[max_indices] # [B, 1]

        # Now we want v = e - x0. In this chosen scenario:
        # chosen_v = chosen_e - chosen_x
        chosen_v = chosen_e - chosen_x

        # Predict v
        v_pred = model(xt_batch, t_batch)
        loss = criterion(v_pred, chosen_v)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses

def train_improved_weighted(model, optimizer, steps=1000, batch_size=6400):
    model.train()
    losses = []
    criterion = nn.MSELoss()
    for step in range(steps):
        optimizer.zero_grad()

        # Sample the minibatch
        x_batch = sample_x(batch_size)  # x0 for each sample
        e_batch = sample_e(batch_size)
        t_batch = sample_t(batch_size)

        # Compute xt for each sample
        xt_batch = (1 - t_batch)*x_batch + t_batch*e_batch

        # For each xt in the batch, find the candidate e and corresponding x that
        # yields the highest pdf under N(0,1).

        # Expand for broadcasting:
        xt_expanded = xt_batch.unsqueeze(1)  # [B, 1, 1]
        x_expanded = x_batch.unsqueeze(0)    # [1, B, 1]
        t_expanded = t_batch.unsqueeze(1)    # [B, 1, 1]

        # Candidate e for (xt[i], x[j]):
        # e_candidate[i, j] = (xt[i] - (1-t[i])*x[j]) / t[i]
        e_candidates = (xt_expanded - (1 - t_expanded)*x_expanded) / t_expanded
        pdf_values = gaussian_pdf(e_candidates)  # [B,B,1]
        pdf_values = pdf_values.squeeze(-1)  # [B,B]
        # v_candidates = e_candidates - x_j
        x_for_v = x_batch.unsqueeze(0).expand_as(e_candidates) # [B,B,1]
        v_candidates = e_candidates - x_for_v  # [B,B,1]

        # Weighted average of v
        weights = pdf_values
        weights_sum = torch.sum(weights, dim=1, keepdim=True) + 1e-8
        chosen_v = torch.sum(weights.unsqueeze(-1)*v_candidates, dim=1) / weights_sum.unsqueeze(-1) # [B,1]

        # Predict v
        v_pred = model(xt_batch, t_batch)
        loss = criterion(v_pred, chosen_v)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


def train_refined_improved_detached_alpha(model, optimizer, steps=1000, batch_size=64, lambda_uncertainty=1.0):
    model.train()
    losses = []
    criterion = nn.MSELoss(reduction='none')  # We'll handle weighting manually
    for step in range(steps):
        optimizer.zero_grad()

        # Sample the minibatch
        x_batch = sample_x(batch_size)  # x0
        e_batch = sample_e(batch_size)
        t_batch = sample_t(batch_size)

        x_batch = x_batch.cuda()
        e_batch = e_batch.cuda()
        t_batch = t_batch.cuda()

        xt_batch = (1 - t_batch)*x_batch + t_batch*e_batch
        v_true = e_batch - x_batch  # Ground truth v

        # Generate candidate e's and pdf weights as before
        xt_expanded = xt_batch.unsqueeze(1)  # [B,1,1]
        x_expanded = x_batch.unsqueeze(0)    # [1,B,1]
        t_expanded = t_batch.unsqueeze(1)    # [B,1,1]

        e_candidates = (xt_expanded - (1 - t_expanded)*x_expanded) / t_expanded  # [B,B,1]
        pdf_values = gaussian_pdf(e_candidates)  # [B,B,1]
        pdf_values = pdf_values.squeeze(-1)      # [B,B]

        x_for_v = x_batch.unsqueeze(0).expand_as(e_candidates) # [B,B,1]
        v_candidates = e_candidates - x_for_v  # [B,B,1]

        # Weighted average of v:
        weights_sum = torch.sum(pdf_values, dim=1, keepdim=True) + 1e-8
        v_avg = torch.sum(pdf_values.unsqueeze(-1)*v_candidates, dim=1) / weights_sum.unsqueeze(-1)  # [B,1]

        # Compute model prediction
        v_pred = model(xt_batch, t_batch)  # [B,1]

        # Compute the base MSE for each sample
        base_mse = criterion(v_pred, v_true)  # [B,1]

        # Compute uncertainty-based weights
        diff = torch.abs(v_true - v_avg)  # [B,1]

        # Weight: alpha = 1/(1+lambda*diff)
        alpha = 1.0 / (1.0 + lambda_uncertainty*diff)
        # Detach alpha so it doesn't affect gradients
        alpha = alpha.detach()

        # Weighted loss
        weighted_loss = (alpha * base_mse).mean()
        weighted_loss.backward()
        optimizer.step()

        losses.append(weighted_loss.item())

    return losses

# -------------------------
# Run the comparison
# -------------------------
model_baseline = SimpleNet2(input_dim=1, hidden_dim=256, time_embed_dim=256)
model_baseline.cuda()
optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=1e-5)
#baseline_losses = train_baseline(model_baseline, optimizer_baseline, steps=4000, batch_size=1280)
baseline_losses = train_baseline(model_baseline, optimizer_baseline, steps=40000, batch_size=128)

model_improved = SimpleNet()
model_improved.cuda()
optimizer_improved = optim.Adam(model_improved.parameters(), lr=1e-3)
#improved_losses = train_improved_weighted(model_improved, optimizer_improved, steps=2000)
improved_losses = train_refined_improved_detached_alpha(model_improved, optimizer_improved, steps=200)

# -------------------------
# Plot the results
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(baseline_losses, label='Baseline')
plt.plot(improved_losses, label='Improved')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.savefig('loss.png')


# Parameters for visualization
num_x = 50
num_t = 50
delta_t = 1/ num_t
x_vals = torch.linspace(-5, 5, num_x).unsqueeze(1) # vertical axis
t_vals = torch.linspace(0, 1, num_t).unsqueeze(1)  # horizontal axis

X, T = torch.meshgrid(x_vals.squeeze(), t_vals.squeeze(), indexing='ij')
X_flat = X.reshape(-1, 1)
T_flat = T.reshape(-1, 1)

# For visualization, e=0: x_t = (1-t)*x
XT_flat = (1 - T_flat)*X_flat

with torch.no_grad():
    XT_flat = XT_flat.cuda()
    T_flat = T_flat.cuda()
    v_pred_baseline = model_baseline(XT_flat, T_flat)  # [N, 1]
    v_pred_improved = model_improved(XT_flat, T_flat)  # [N, 1]

V_baseline = v_pred_baseline.reshape(num_x, num_t).detach().cpu().numpy()
V_improved = v_pred_improved.reshape(num_x, num_t).detach().cpu().numpy()

X_np = X.detach().cpu().numpy()
T_np = T.detach().cpu().numpy()

# Vector field components for a small increment delta_t:
U_baseline = np.full_like(V_baseline, delta_t)
W_baseline = delta_t * V_baseline

U_improved = np.full_like(V_improved, delta_t)
W_improved = delta_t * V_improved

# Let's generate some trajectories. We start at t=0 and pick several initial x0's.
initial_xs = np.linspace(-3, 3, 50)  # for instance 5 trajectories starting from x = -4 to 4

def integrate_trajectory(model, x0, step=50, t_start=0.0, t_end=1.0):
    # Integrate forward using Euler's method:
    # x_{new} = x_old + dt * v(t, x_old)
    # but remember we must evaluate v at (x_t, t) where x_t=(1-t)*x (with e=0)
    t_current = t_start
    x_current = x0
    traj_t = [t_current]
    traj_x = [x_current]
    model.eval()
    timesteps = torch.linspace(0, 1, step)
    timesteps1 = timesteps[:-1]
    timesteps2 = timesteps[1:]
    dts = timesteps2 - timesteps1
    with torch.no_grad():
        #while t_current < t_end:
        for ind, t in enumerate(timesteps1):
            # Compute x_t for input to model
            # Given t and x, we have x_t = (1 - t)*x
            # But we have an implicit equation: v is defined at (x_t, t), so to find v:
            # Actually, we have chosen x itself as the "state" we are tracking.
            # We must invert the relationship: if x is actually the original x0 value.
            # Wait, in previous steps we used x_t = (1-t)*x. For visualization, we treated x as the vertical axis variable.
            # Here, let's keep consistent: our model is trained to predict v from (x_t, t) where x_t=(1-t)*x.
            # If we consider "X_np" as the original x. Actually, we've been mixing notation:
            # The model expects (x_t, t) as input and outputs v = e - x0. Here we have x as the vertical axis.
            # For trajectory integration, we must be consistent with how we sampled x_t in the visualization.

            # We'll interpret the vertical axis (X_np) as the original x (x0).
            # The model requires x_t for input. Given x0 and t, x_t=(1-t)*x0 (since we set e=0).
            # So for integration:
            # At time t_current, x0 = x_current is actually the original x.
            # x_t = (1 - t_current)*x_current
            #x_t_input = (1 - t_current)*x_current
            dt = dts[ind]
            t_current = t

            x_t_input = x_current

            # Evaluate v
            xt_tensor = torch.tensor([[x_t_input]], dtype=torch.float32)
            t_tensor = torch.tensor([[t_current]], dtype=torch.float32)
            xt_tensor = xt_tensor.cuda()
            t_tensor = t_tensor.cuda()
            v_pred = model(xt_tensor, t_tensor).item()

            # Now update x using x_{new} = x_{old} + dt * v
            # v is defined as v = e - x0, but in this setup, we considered that we are just plotting the field.
            # Actually, we want to follow the "flow" given by v in terms of x0.
            # If we consider x as the original coordinate, and v as directions in that coordinate:
            # The vertical axis x we are plotting is effectively x0. The model's v = e - x0 suggests a direction in x0-space.
            # So we can directly update x_current by dt * v.

            x_current = x_current + dt * v_pred
            t_current = t_current + dt

            traj_t.append(t_current)
            traj_x.append(x_current)
    return np.array(traj_t), np.array(traj_x)

# Plot Baseline vector field + trajectories
plt.figure(figsize=(10,5))
plt.title("Baseline Vector Field + Trajectories (t horizontal, x vertical)")
plt.xlabel("t")
plt.ylabel("x")
# Vector field
step = 4
plt.quiver(T_np[::step, ::step], X_np[::step, ::step],
           U_baseline[::step, ::step], W_baseline[::step, ::step],
           angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5)

# Plot trajectories
for x0 in initial_xs:
    traj_t, traj_x = integrate_trajectory(model_baseline, x0, step=50, t_start=0.0, t_end=1.0)
    plt.plot(traj_t, traj_x, color='black')

plt.grid(True)
#plt.show()
plt.savefig('baseline_traj.png')

# Plot Improved vector field + trajectories
plt.figure(figsize=(10,5))
plt.title("Improved Vector Field + Trajectories (t horizontal, x vertical)")
plt.xlabel("t")
plt.ylabel("x")
plt.quiver(T_np[::step, ::step], X_np[::step, ::step],
           U_improved[::step, ::step], W_improved[::step, ::step],
           angles='xy', scale_units='xy', scale=1, color='red', alpha=0.5)

for x0 in initial_xs:
    traj_t, traj_x = integrate_trajectory(model_improved, x0, step=50, t_start=0.0, t_end=1.0)
    plt.plot(traj_t, traj_x, color='black')

plt.grid(True)
#plt.show()
plt.savefig('improved_traj.png')


# Assume model_improved is trained and available

def plot_model(model, name):
    model.eval()

    # Set parameters
    num_samples = 10000
    #delta_t = 0.02
    #steps = int(1.0 / delta_t)  # 50 steps if delta_t=0.02
    steps = 50
    delta_t = 1 / steps
    t_values = torch.linspace(1.0, 0.0, steps)  # from t=1 to t=0

    # Generate initial noise samples at t=1
    x_t = torch.randn(num_samples, 1)  # x(t=1) = e ~ N(0,1)

    # We'll integrate backward in time:
    for i in range(steps):
        t_current = t_values[i]
        # The model expects input shape [batch, 1], so we have x_t: [num_samples, 1], t: [num_samples, 1]
        t_tensor = torch.full((num_samples, 1), t_current, dtype=torch.float32)

        # Predict v = f(x_t, t)
        with torch.no_grad():
            x_t = x_t.cuda()
            t_tensor = t_tensor.cuda()
            v_pred = model(x_t, t_tensor)  # shape [num_samples, 1]

        # Update x_t: x_{t - delta_t} = x_t - v_pred * delta_t
        x_t = x_t - v_pred * delta_t

    # After the loop, x_t should correspond to x0 samples
    x_0_samples = x_t.detach().cpu().numpy().flatten()

    # Plot a histogram of the resulting samples x_0
    plt.figure(figsize=(10,5))
    plt.hist(x_0_samples, bins=100, density=True, alpha=0.7, color='blue')
    plt.xlabel("x_0")
    plt.ylabel("Probability Density")
    plt.title("Samples from the Diffusion Model (Reconstructed from Noise)")
    plt.grid(True)
    #plt.show()
    plt.savefig(f'{name}_sample.png')

plot_model(model_baseline, 'baseline')
#plot_model(model_improved, 'improved')

