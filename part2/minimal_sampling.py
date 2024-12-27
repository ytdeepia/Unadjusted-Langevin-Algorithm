n_steps = 5000
step_size = 0.01 * sigma**2

true_x = train_dataloader.dataset[0][0].unsqueeze(0).to(device)
x_sample = true_x + 0.5 * torch.randn_like(x_sample)

with torch.no_grad():
    for t in range(n_steps):
        grad = (model(x_sample, sigma) - x_sample) / sigma**2
        x_sample = (
            x_sample
            + step_size * grad
            + torch.randn_like(x_sample) * np.sqrt(2 * step_size)
        )
