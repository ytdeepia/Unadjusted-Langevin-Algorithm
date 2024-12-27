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
import matplotlib.pyplot as plt

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
y = physics.forward(x).to(device)


model = DRUNet(
    in_channels=1, out_channels=1, pretrained=Path("./ckpts/denoising_model.pth")
).to(device)

x_denoised = model(y, sigma)
plot([x, y, x_denoised])

# %%

RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

for i in range(5):
    x = train_dataset[i][0].unsqueeze(0)
    y = physics.forward(x)
    x_denoised = model(y, sigma)

    # Using matplotlib to save
    plt.imsave(f"./img/noisy_{i}.png", y.squeeze().detach().cpu().numpy(), cmap="gray")
    plt.imsave(
        f"./img/denoised_{i}.png",
        x_denoised.squeeze().detach().cpu().numpy(),
        cmap="gray",
    )

# %%
# Langevin sampling
n_steps = 5000
step_size = 0.01 * 0.05**2

# Start from random noise
true_x = train_dataset[0][0].unsqueeze(0)
batch_size = 16
x_sample = torch.randn(batch_size, *true_x.shape[1:]).to(device)

plot(x_sample[:8])  # Plot first 8 samples to avoid overcrowding

with torch.no_grad():
    for t in tqdm(range(n_steps)):
        grad = (model(x_sample, sigma) - x_sample) / sigma**2
        noise = torch.randn_like(x_sample) * np.sqrt(2 * step_size)
        x_sample = x_sample + step_size * grad + np.sqrt(step_size) * noise

        # Save images every 100 iterations
        if t % 50 == 0:
            for b in range(batch_size):
                plt.imsave(
                    f"./img/noisy_sample_{b}_iter_{t}.png",
                    x_sample[b].squeeze().cpu().numpy(),
                    cmap="gray",
                )

plot(x_sample)
