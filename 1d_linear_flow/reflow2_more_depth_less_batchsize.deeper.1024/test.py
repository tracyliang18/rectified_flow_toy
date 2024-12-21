import torch

timesteps = torch.linspace(0, 1, 1000)

# Parameters
batch_size = 400

# Randomly sample indices
indices = torch.randint(0, len(timesteps), (batch_size,))

# Draw random samples
batch = timesteps[indices]

# Print the batch
print(f"Random Batch: {batch[:30]}")
