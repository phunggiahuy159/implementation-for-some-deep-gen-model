import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

batch_size=32


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
dataset=datasets.MNIST(root="dataset/", transform=transform, download=False)
loader=DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Discriminator(nn.Module):
    def __init__(self,in_features) :
        super().__init__()
        self.disc=nn.Sequential(nn.Linear(in_features,128),
                                nn.LeakyReLU(0.01),
                                nn.Linear(128,1),
                                nn.Sigmoid()
                                )
    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self,z_dim,img_dim):
        super().__init__()
        self.gen=nn.Sequential(nn.Linear(z_dim,256),
                               nn.LeakyReLU(0.01),
                               nn.Linear(256,img_dim),
                               nn.Tanh())
    def forward(self,x):
        return self.gen(x)
    
z_dim=100
img_dim= 28*28
disc_in_features= 28*28 
gen=Generator(z_dim,img_dim)
disc=Discriminator(disc_in_features)

lr=3e-4
gen_optim=torch.optim.Adam(gen.parameters(),lr=lr)
disc_optim=torch.optim.Adam(disc.parameters(),lr=lr)
criterion=nn.BCELoss()
epochs=40


for epoch in range(epochs):
    for idx,(batch,y) in enumerate(loader):
        real_flat=batch.view(-1,28*28)
        batch_size=real_flat.shape[0]
        noise = torch.randn(batch_size, z_dim)
        fake=gen(noise)
        output_fake=disc(fake).view(-1)
        output_real=disc(real_flat).view(-1)
        loss_fake=criterion(output_fake,torch.zeros_like(output_fake))
        loss_real=criterion(output_real,torch.ones_like(output_real))
        total_loss_disc=(loss_real+loss_fake)/2
        disc.zero_grad()
        total_loss_disc.backward(retain_graph=True)
        disc_optim.step()

        output=disc(fake).view(-1)
        loss_gen=criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        gen_optim.step()
        if idx == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}]  \
                      Loss D: {total_loss_disc:.4f}, loss G: {loss_gen:.4f}")
def show_generated_images(images):
    img_grid = torchvision.utils.make_grid(images, normalize=True)
    np_img = img_grid.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
noise=torch.rand(batch_size,z_dim)
with torch.no_grad():
    final_fake = gen(noise).reshape(-1, 1, 28, 28)
    show_generated_images(final_fake)
























