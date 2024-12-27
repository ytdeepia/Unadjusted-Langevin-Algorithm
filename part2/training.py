# %%
import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
from deepinv.models import DRUNet
import numpy as np
from deepinv.utils.demo import load_image, get_data_home
from deepinv.physics.noise import GaussianNoise
from torchvision import datasets
from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "ckpts"
ORIGINAL_DATA_DIR = get_data_home()

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
train_test_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=True, transform=train_test_transform, download=True
)

sigma = 0.1
physics = dinv.physics.Denoising(noise_model=GaussianNoise(sigma))

x = train_dataset[0][0].unsqueeze(0)
y = physics.forward(x)

plot([x, y])

# %%
model = DRUNet(in_channels=1, out_channels=1)
loss = dinv.loss.metric.MSE(train_loss=True)
n_epoch = 5
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 64
num_workers = 6

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
model = model.to(device)
physics = physics.to(device)

for epoch in range(n_epoch):
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(tqdm(train_dataloader)):
        data = data.to(device)
        optim.zero_grad()

        y = physics.forward(data)
        pred = model(y, sigma)

        loss_value = loss(pred, data).mean()
        loss_value.backward()
        optim.step()

        running_loss += loss_value.item()

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{n_epoch}, Loss: {avg_loss:.6f}")

# %%
CKPT_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), CKPT_DIR / "denoising_model.pth")

# %%
# Langevin sampling
n_steps = 5000
step_size = 0.01 * 0.05**2

# Start from random noise
true_x = train_dataloader.dataset[0][0].unsqueeze(0).to(device)
batch_size = 16
x_sample = torch.randn(batch_size, *true_x.shape[1:]).to(device)

# Add noise to initial samples
x_sample = true_x.repeat(batch_size, 1, 1, 1) + 0.5 * torch.randn(
    batch_size, *true_x.shape[1:]
).to(device)

plot(x_sample[:8])  # Plot first 8 samples to avoid overcrowding

with torch.no_grad():
    for t in tqdm(range(n_steps)):
        grad = (model(x_sample, sigma) - x_sample) / sigma**2
        noise = torch.randn_like(x_sample) * np.sqrt(2 * step_size)
        x_sample = x_sample + step_size * grad + np.sqrt(step_size) * noise

plot(x_sample)

# %%
# minimal training example
