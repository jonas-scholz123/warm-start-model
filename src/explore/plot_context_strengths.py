# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


csv_path = "../../fid_results_context_strenghts.csv"
#csv_path = "../../fid_results_mid_run.csv"
#csv_path2 = "../../fid_results.old.csv"

num_samples = 50000

df = pd.read_csv(csv_path)
df = df[df["num_samples"] == num_samples]
#df2 = pd.read_csv(csv_path2)
#df = pd.concat([df, df2], ignore_index=True)

#baseline_exp = "cifar10_cold_fm_0_100"
#wsd_exp = "new_warmth_scaling_e2e_0_100"

# Compare all to all
#baseline_experiments = ["2025-12-04_16-21_xenial_kangaroo", "cifar10_cold_fm_0_100"]
#wsd_experiments = ["2025-12-06_10-11_brave_xenon", "new_warmth_scaling_e2e_0_100"]

# Compare the two standard FM approaches
#baseline_experiments = ["cifar10_cold_fm_0_100"]
#wsd_experiments = ["2025-12-04_16-21_xenial_kangaroo"]

# Compare the two warm start approaches
#baseline_experiments = ["new_warmth_scaling_e2e_0_100"]
#wsd_experiments = ["cifar10_cold_fm_0_100"]

# Compare Finetuned to finetuned
baseline_experiments = ["cifar10_cold_fm_0_100"]
wsd_experiments = ["new_warmth_scaling_e2e_0_100"]

nfes = [2, 4, 6, 8, 12, 20, 50,]
context_fractions = [
    #0.0,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    #0.7,
    #0.8,
    0.9,
    #0.95,
    #0.98,
    0.99,
]

mode = "eq"
#mode = "leq"


wsd_heatmap = np.zeros((len(nfes), len(context_fractions)))
baseline_heatmap = np.zeros((len(nfes), len(context_fractions)))

for i, nfe in enumerate(nfes):
    for j, ctx_frac in enumerate(context_fractions):
        if mode == "eq":
            nfe_df = df[df["nfe"] == nfe]
        elif mode == "leq":
            nfe_df = df[df["nfe"] <= nfe]
        wsd_df = nfe_df[
            (df["ctx_frac"] == ctx_frac)
            & (df["experiment"].isin(wsd_experiments))
        ]
        baseline_df = nfe_df[
            (df["ctx_frac"] == ctx_frac)
            & (df["experiment"].isin(baseline_experiments))
        ]
        if wsd_df.empty:
            print(f"No WSD data for NFE={nfe}, ctx_frac={ctx_frac}")
            wsd_value = np.nan
        else:
            wsd_value = wsd_df["fid"].min()

        if baseline_df.empty:
            baseline_value = np.nan
        else:
            baseline_value = baseline_df["fid"].min()

        baseline_heatmap[i, j] = baseline_value
        wsd_heatmap[i, j] = wsd_value

# %%

#%%
plt.figure(figsize=(len(context_fractions) * 0.7, len(nfes) *0.7))
difference = baseline_heatmap - wsd_heatmap

# Heatmap with printing the values in each cell:
sns.heatmap(
    difference,
    xticklabels=context_fractions,
    yticklabels=nfes,
    cmap="PuOr_r",
    cbar_kws={"label": "FID Improvement"},
    #vmin=-8,
    #vmax=8,
    center=0.0,
    annot=True,
    annot_kws={"fontsize":9},
    fmt=".1f",
)

plt.xlabel("Fraction of Visible Context Pixels, $\\rho$")
plt.ylabel("NFE")
#plt.title("CIFAR-10 FID Improvement (Baseline FID - WSD FID)")
plt.savefig("../../_results/context_strength_heatmap.pdf", bbox_inches="tight")
plt.show()

# %%

#%%

plt.figure(figsize=(len(context_fractions) + 1, len(nfes)))
ratio = (baseline_heatmap - wsd_heatmap) / baseline_heatmap
sns.heatmap(
    ratio,
    xticklabels=context_fractions,
    yticklabels=nfes,
    cmap="bwr",
    cbar_kws={"label": "FID Reduction"},
    vmin=-0.5,
    vmax=0.5,
    annot=True,
    fmt=".2f",
)

plt.xlabel("Context Fraction")
plt.show()
#%%
plt.figure(figsize=(len(context_fractions) + 1, len(nfes)))
sns.heatmap(
    wsd_heatmap,
    xticklabels=context_fractions,
    yticklabels=nfes,
    cbar_kws={"label": "WSD FID"},
    cmap="viridis",
    annot=True,
    fmt=".1f",
)
#%%
plt.figure(figsize=(len(context_fractions) + 1, len(nfes)))
sns.heatmap(
    ratio,
    xticklabels=context_fractions,
    yticklabels=nfes,
    cbar_kws={"label": "Baseline FID"},
    cmap="viridis",
    annot=True,
    fmt=".2f",
)
# %%
