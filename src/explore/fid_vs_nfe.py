#%%
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class PlotConfig:
    title: str
    experiment_map: dict[str, str]
    min_samples: int = 50_000
    linewidth: float | None = None
    linestyle: str = "--"

plot_configs = [
    PlotConfig(
        title="Celeba-64x64 Inpainting",
        experiment_map={
            "celeba_cold_fm": "Flow Matching",
            "2025-07-29_22-57_quirky_jaguar": "Warm FM",
            #"2025-09-05_19-38_vibrant_fish_e2e": "Warm FM E2E",
        },
    ),
    PlotConfig(
        title="CIFAR10-32x32 Inpainting",
        experiment_map={
            "2025-07-21_22-38_playful_xenon": "Flow Matching",
            #"new_warmth_scaling": "Warm FM",
            # TODO?
            "new_warmth_scaling_end_to_end4": "Warm FM",
        },
    ),
    PlotConfig(
        title="CIFAR10-32x32 Inp. (All Ablations)",
        experiment_map={
            "2025-07-21_22-38_playful_xenon": "Flow Matching",
            #"new_warmth_scaling": "Warm FM",
            "new_warmth_scaling_end_to_end4": "Warm FM",
            "mean_only_continuation": "Mean Only",
            "2025-07-30_18-12_optimistic_narwhal": "No Warmth Blending",
            "feature_only_ablation": "Feature Only",
        },
    ),
    PlotConfig(
        title="AFHQ Superresolution",
        experiment_map={
            "2025-12-21_18-30_delightful_lion": "Flow Matching",
            "2025-12-28_22-39_witty_bear": "Warm FM",
            "2025-12-28_22-39_witty_bear_e2e": "Warm FM E2E",
        },
    ),
]

paths = [
    "/home/jonas/Documents/code/denoising-np/fid_results.csv",
    "/home/jonas/Documents/code/denoising-np/_results/fid_results.csv",
    "/home/jonas/Documents/code/denoising-np/_results/fid_results_celeba.csv",
    "/home/jonas/Documents/code/denoising-np/fid_results_2025-12-21_18-30_delightful_lion.csv",
    "/home/jonas/Documents/code/denoising-np/fid_results_2025-12-28_22-39_witty_bear.csv",
    "/home/jonas/Documents/code/denoising-np/fid_results_2025-12-28_22-39_witty_bear_e2e.csv",
    "/home/jonas/Documents/code/denoising-np/fid_results_2025-09-05_19-38_vibrant_fish_e2e.csv"
]

alphabet = "abcdefghijklmnopqrstuvwxyz"

dfs = []
for path in paths:
    df = pd.read_csv(path)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True).sort_values(["experiment", "nfe"])

#df = df[df["solver"] == "dpm_solver_3"]
#df = df[df["solver"] == "dpm_solver_2"]
#df = df[df["solver"] == "euler"]
# df = df[df["nfe"] >= 2]
#df = df[df["solver"] == "midpoint"]
df = df[df["nfe"] % 2 == 0]
df = df[df["nfe"] <= 50]


num_cols = len(plot_configs)
num_rows = 1
width = 4 * num_cols
height = 3.2 * num_rows

fig, axs = plt.subplots(num_rows, num_cols, figsize=(width, height), sharey=False)

for i, pc in enumerate(plot_configs):
    ax = axs[i]
    sub_df = df[df["experiment"].isin(pc.experiment_map.keys())]
    nfes = sub_df["nfe"].unique()
    for experiment, label in pc.experiment_map.items():
        exp_df = sub_df[sub_df["experiment"] == experiment]

        fids = []
        for nfe in nfes:
            fid = exp_df[exp_df["nfe"] == nfe]["fid"].min()
            fids.append(fid)

        ax.plot(nfes, fids, marker="x", label=label, linestyle=pc.linestyle, linewidth=pc.linewidth)
    ax.set_xscale("log")
    # Too crowded
    nfes = [nfe for nfe in nfes if nfe != 10]
    ax.set_xticks(nfes)
    ax.set_xticklabels(nfes)
    ax.set_xlabel("NFE")
    #ax.set_ylim(0.7, 6)
    ax.set_ylim(None, 6)
    # minor ticks off
    ax.minorticks_off()
    ax.legend()

    title = f"{alphabet[i]}) {pc.title}"
    ax.set_title(title)

axs[0].set_ylabel("Fréchet Inception Distance (FID)")
fig.subplots_adjust(wspace=0.1)

fig.savefig("fid_vs_nfe_all_experiments.pdf", bbox_inches="tight")


plt.show()




#%%

experiments = df["experiment"].unique().tolist()

for experiment in experiments:
    sub_df = df[df["experiment"] == experiment]
    nfes = sub_df["nfe"].unique()

    fids = []
    for nfe in nfes:
        fid = sub_df[sub_df["nfe"] == nfe]["fid"].min()
        fids.append(fid)

    plt.plot(nfes, fids, marker="x", label=experiment, linestyle="--")

plt.xscale("log")
plt.xticks(ticks=nfes, labels=nfes)
plt.xlabel("Number of Function Evaluations (NFE)")
plt.ylabel("Fréchet Inception Distance (FID)")
plt.minorticks_off()
plt.title("FID vs NFE")
plt.legend()
plt.show()
# %%

df = pd.read_csv(path)
df = df[df["nfe"] == 4]
df = df[df["experiment"] == "2025-07-23_15-24_sassy_unicorn_better_cnp"].sort_values("fid")
df.sort_values("fid")
#%%
# df[df["experiment"] == "2025-07-21_22-38_playful_xenon"]
#df[df["experiment"] == "2025-07-23_15-24_sassy_unicorn_better_cnp"].sort_values("fid")
best_solver = df[df["fid"] == df["fid"].min()]["solver"]
best_solver
# %%
df = pd.read_csv(path)
df = df[df["nfe"] == 10].sort_values("fid")
df
#%%

df = pd.read_csv(path)
df = df[df["experiment"] == "2025-07-21_22-38_playful_xenon"]
df = df.sort_values(["nfe", "fid"])
df.head(50)
