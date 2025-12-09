# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

csv_path = "../../fid_results.csv"
baseline_exp = "2025-12-04_16-21_xenial_kangaroo"
wsd_exp = "2025-07-23_15-24_sassy_unicorn_better_cnp"
nfes = [2, 4, 6, 8, 12, 20, 50, 100]
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
    baseline_heatmap,
    xticklabels=context_fractions,
    yticklabels=nfes,
    cmap="bwr",
    cbar_kws={"label": "WSD FID / Baseline FID"},
)

plt.xlabel("Context Fraction")
