import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from torchinfo import summary

# generate a single circle dataset
def generate_circle_data(n_samples=1000, radius=1, noise=0.1):
    angles = 2 * np.pi * np.random.rand(n_samples)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    data = np.vstack([x, y]).T
    data += noise * np.random.randn(n_samples, 2)  # Add some noise
    return torch.tensor(data, dtype=torch.float32)

# generate a 2D single circle dataset
n_samples = 1000
X = generate_circle_data(n_samples=n_samples, radius=1, noise=0.05)

# Plot the original dataset
plt.scatter(X[:, 0], X[:, 1], cmap='viridis')
plt.title("Single Circle Dataset")
plt.axis('equal')
plt.show()

# RealNVP model
class RealNVPNode(nn.Module):
    def __init__(self, mask, hidden_size):
        super(RealNVPNode, self).__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.s_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim)
        )

        self.t_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim)
        )

    def forward(self, x):
        x_mask = x * self.mask
        s = self.s_func(x_mask)
        t = self.t_func(x_mask)

        y = x_mask + (1 - self.mask) * (x * torch.exp(s) + t)

        log_det_jac = ((1 - self.mask) * s).sum(dim=-1)
        return y, log_det_jac

    def inverse(self, y):
        y_mask = y * self.mask
        s = self.s_func(y_mask)
        t = self.t_func(y_mask)

        x = y_mask + (1 - self.mask) * (y - t) * torch.exp(-s)

        inv_log_det_jac = ((1 - self.mask) * -s).sum(dim=-1)

        return x, inv_log_det_jac

class RealNVP(nn.Module):
    def __init__(self, masks, hidden_size):
        super(RealNVP, self).__init__()

        self.dim = len(masks[0])
        self.hidden_size = hidden_size

        self.masks = nn.ParameterList([nn.Parameter(mask.float(), requires_grad=False) for mask in masks])
        self.layers = nn.ModuleList([RealNVPNode(mask, self.hidden_size) for mask in self.masks])

        self.distribution = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim) * 0.5)

    def log_probability(self, x):
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for layer in reversed(self.layers):
            x, inv_log_det_jac = layer.inverse(x)
            log_prob += inv_log_det_jac
            
        log_prob += self.distribution.log_prob(x)

        return log_prob

    def rsample(self, num_samples):
        x = self.distribution.sample((num_samples,))
        log_prob = self.distribution.log_prob(x)

        for layer in self.layers:
            x, log_det_jac = layer.forward(x)
            log_prob += log_det_jac

        return x, log_prob

# Function to train the RealNVP model
def train_real_nvp(model, data, epochs=400, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        log_prob = model.log_probability(data)
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# masks for the RealNVP model
masks = [torch.tensor([1, 0]), torch.tensor([0, 1])] * 5  # 10 layers total
hidden_size = 64  

model = RealNVP(masks=masks, hidden_size=hidden_size)

train_real_nvp(model, X)

num_samples = 1000
samples, _ = model.rsample(num_samples)

plt.scatter(samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy(), cmap='coolwarm')
plt.title("Generated Samples from RealNVP")
plt.axis('equal')
plt.show()
print(summary(model))