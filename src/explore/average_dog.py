# %%
from matplotlib import colorbar
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors  # <--- 1. Import colors explicitly
from torchvision import datasets, transforms

data_path = "/home/jonas/Documents/code/denoising-np/_data"
dataset = datasets.CIFAR10(root=data_path, train=True, transform=transforms.ToTensor())

# %%
# 0 = airplane, 1 = automobile, 2 = bird, 3 = cat, 4 = deer
# 5 = dog, 6 = frog, 7 = horse, 8 = ship, 9 = truck
name_to_class = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}

class_to_name = {v: k for k, v in name_to_class.items()}

first_moments = {}
second_moments = {}
first_moments_color = {}

class_counts = {name: 0 for name in name_to_class.keys()}

for i in range(len(dataset)):
    img, im_class = dataset[i]
    class_name = class_to_name[im_class]
    img_color = img.clone()
    img = img.mean(dim=0, keepdim=True)  # Convert to grayscale
    if class_name not in first_moments:
        first_moments[class_name] = torch.zeros_like(img)
        second_moments[class_name] = torch.zeros_like(img)
        first_moments_color[class_name] = torch.zeros_like(img_color)
    first_moments[class_name] += img
    second_moments[class_name] += img**2
    first_moments_color[class_name] += img_color
    class_counts[class_name] += 1

#%%
means = {}
means_color = {}
stds = {}

for class_name in first_moments:
    count = class_counts[class_name]
    fm = first_moments[class_name] / count
    sm = second_moments[class_name] / count
    means[class_name] = fm
    means_color[class_name] = first_moments_color[class_name] / count
    stds[class_name] = (sm - fm**2) ** 0.5


show_classes = ["dog", "cat", "automobile", "airplane", "truck"]

fig, axs = plt.subplots(
    figsize=(8, 4), nrows=3, ncols=len(show_classes), constrained_layout=True, gridspec_kw={"height_ratios": [1, 1, 0.05]}
)

for i, show_class in enumerate(show_classes):
    # Plot Mean (RGB)
    axs[0, i].imshow(means_color[show_class].permute(1, 2, 0).numpy())
    axs[0, i].set_title(f"$C$ = {show_class}")
    axs[0, i].axis("off")  # <--- Removes grid, ticks, and box
    
    # Plot Std (Heatmap)
    std_data = stds[show_class].permute(1, 2, 0).numpy()
    
    # 2. Define scale manually so 0 is always black/purple (Standard Dev starts at 0)
    norm = mcolors.Normalize(vmin=std_data.min(), vmax=std_data.max())
    
    # 3. Pass 'norm' to imshow so the image obeys it
    im = axs[1, i].imshow(std_data, cmap="viridis", norm=norm)
    axs[1, i].axis("off")  # <--- Removes grid, ticks, and box
    
    # 4. Pass the image object 'im' to colorbar (easiest way to link them)
    fig.colorbar(im, cax=axs[2, i], orientation="horizontal")

axs[0, 0].set_ylabel("Mean Image")
axs[1, 0].set_ylabel("Std Dev Image")
fig.text(-0.03, 0.75, '$\hat{\mu}_C$', va='center', rotation='vertical', fontsize=12)
fig.text(-0.03, 0.33, '$\hat{\sigma}_C$', va='center', rotation='vertical', fontsize=12)

fig.savefig("class_conditional_average.pdf", bbox_inches="tight")