# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

csv_path = "../../fid_results.csv"
baseline_exp = "2025-12-04_16-21_xenial_kangaroo"
#wsd_exp = "2025-12-06_10-11_brave_xenon"
#baseline_exp = "2025-07-21_22-38_playful_xenon"
#wsd_exp = "new_warmth_scaling"
wsd_exp = "new_warmth_scaling_e2e_0_100"
nfes = [2, 4, 6, 8, 12, 20, 50]
context_fractions = [
    0.0,
    0.01,
    0.02,
    0.05,
    0.1,
    0.3,
    0.5,
    0.8,
    0.9,
    0.95,
    0.98,
    0.99,
]

df = pd.read_csv(csv_path)

wsd_heatmap = np.zeros((len(nfes), len(context_fractions)))
baseline_heatmap = np.zeros((len(nfes), len(context_fractions)))

for i, nfe in enumerate(nfes):
    for j, ctx_frac in enumerate(context_fractions):
        wsd_df = df[
            (df["nfe"] == nfe)
            & (df["ctx_frac"] == ctx_frac)
            & (df["experiment"] == wsd_exp)
        ]
        baseline_df = df[
            (df["nfe"] == nfe)
            & (df["ctx_frac"] == ctx_frac)
            & (df["experiment"] == baseline_exp)
        ]
        if wsd_df.empty:
            print(f"No WSD data for NFE={nfe}, ctx_frac={ctx_frac}")
            wsd_value = -1
        else:
            wsd_value = wsd_df["fid"].min()

        if baseline_df.empty:
            baseline_value = -1
        else:
            baseline_value = baseline_df["fid"].min()

        baseline_heatmap[i, j] = baseline_value
        wsd_heatmap[i, j] = wsd_value

plt.figure(figsize=(len(context_fractions) + 1, len(nfes)))

ratio = baseline_heatmap / wsd_heatmap

sns.heatmap(
    ratio,
    xticklabels=context_fractions,
    yticklabels=nfes,
    cmap="bwr",
    cbar_kws={"label": "Baseline FID / WSD FID"},
    vmin=0.6,
    vmax=1.4,
)

plt.xlabel("Context Fraction")
plt.show()
# %%

# %%
print(baseline_heatmap[0, 0], wsd_heatmap[0, 0])
#%%
difference = baseline_heatmap - wsd_heatmap

# Heatmap with printing the values in each cell:
sns.heatmap(
    difference,
    xticklabels=context_fractions,
    yticklabels=nfes,
    cmap="bwr",
    cbar_kws={"label": "Baseline FID - WSD FID"},
    vmin=-10,
    vmax=10,
    annot=True,
    fmt=".1f",
)

plt.xlabel("Context Fraction")
plt.show()