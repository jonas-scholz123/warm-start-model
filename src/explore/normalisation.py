# %%
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from mlbnb.types import Split

from cdnp.data.cifar10 import Cifar10Dataset


# %%
@dataclass
class Paths:
    data: str


generator = torch.Generator()

paths = Paths(data="/home/jonas/Documents/code/denoising-np/_data")

dataset = Cifar10Dataset(
    norm_means=[0.5, 0.5, 0.5],  # ty: ignore
    norm_stds=[0.5, 0.5, 0.5],  # ty: ignore
    val_fraction=0.1,
    paths=paths,  # ty: ignore
    split=Split.TRAIN,
    generator=generator,
)

# %%

mean = torch.zeros_like(dataset[0][0])
second_moment = torch.zeros_like(dataset[0][0])

count = 0
for x, _ in dataset:
    mean += x
    second_moment += x**2
    count += 1

mean /= count
second_moment /= count

std = torch.sqrt(second_moment - mean**2)

# %%

# plt.imshow(dataset[0][0].permute(1, 2, 0))

mean_un = mean * 0.5 + 0.5
plt.imshow(mean_un.permute(1, 2, 0))
plt.show()
# %%

std_single = std[0, :, :].unsqueeze(0) * 0.5 + 0.5
plt.imshow(std_single.permute(1, 2, 0).squeeze())
plt.show()
# %%
# Save
mean = mean.numpy()
std = std.numpy()

path = "/home/jonas/Documents/code/denoising-np/_normalisation"

np.savez(f"{path}/cifar10_normalisation.npz", mean=mean, std=std)
# %%

# double check:
data = np.load(f"{path}/cifar10_normalisation.npz")
mean_loaded = data["mean"]
std_loaded = data["std"]

assert np.allclose(mean_loaded, mean)
assert np.allclose(std_loaded, std)

# %%

mean.shape
std.shape