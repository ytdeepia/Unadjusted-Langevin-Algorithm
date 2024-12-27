import torch
from torchvision import datasets, transforms
import deepinv

device = "cuda"

n_epoch = 10
batch_size = 64
num_workers = 6

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(train=True, transform=transform, download=True)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
)

sigma = 0.1
physics = deepinv.physics.Denoising(
    noise_model=deepinv.physics.GaussianNoise(sigma)
).to(device)

model = deepinv.models.DRUNet(in_channels=1, out_channels=1, device=device)
loss = deepinv.loss.metric.MSE(train_loss=True)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(n_epoch):
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_dataloader):
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
