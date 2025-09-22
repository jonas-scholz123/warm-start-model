# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

results_dir = Path("[redacted]")
path = results_dir / "fid_results.csv"

name_map = {
    "2025-07-21_22-38_playful_xenon": "Flow Matching",
    "new_warmth_scaling": "Warm Flow Matching",
    "mean_only_continuation": "Mean Only",
    "2025-07-30_18-12_optimistic_narwhal": "No Warmth Blending",
}

ablations = [
    "mean_only_continuation",
    "2025-07-30_18-12_optimistic_narwhal",
]


def plot_fid_vs_nfe(
    df: pd.DataFrame,
    ax: plt.Axes,
    name_map: dict[str, str],
    ylim_top: float | None = 8.0,
    xscale: str = "log",
    title: str = "",
    legend: bool = False,
    y_label = "FID",
    **plot_kwargs,
):
    """
    Plots Fréchet Inception Distance (FID) vs. Number of Function Evaluations (NFE)
    from a pre-filtered DataFrame.

    Args:
        df (pd.DataFrame): A pre-filtered DataFrame containing the data to plot. It must
                           include 'experiment', 'nfe', and 'fid' columns.
        ax (plt.Axes): The matplotlib axes object to plot on.
        name_map (Dict[str, str]): A dictionary mapping experiment IDs from the DataFrame
                                   to human-readable names for the legend.
        ylim_top (Optional[float]): The upper limit for the y-axis.
        xscale (str): The scale for the x-axis (e.g., 'log', 'linear').
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        **plot_kwargs: Additional keyword arguments passed directly to ax.plot().
                       For example, marker='x', linestyle='--'.
    """
    df = df.copy()
    # The function now assumes `df` is already prepared.
    # It will plot data for any experiment ID found in the keys of `name_map`.
    for experiment_id, experiment_name in name_map.items():
        sub_df = df[df["experiment"] == experiment_id]

        if sub_df.empty:
            print(
                f"Warning: No data found for experiment '{experiment_id}' in the provided DataFrame."
            )
            continue

        # Find the minimum FID for each NFE using groupby
        min_fids_by_nfe = sub_df.groupby("nfe")["fid"].min()

        ax.plot(
            min_fids_by_nfe.index,
            min_fids_by_nfe.values,
            label=experiment_name,
            **plot_kwargs,
        )

    # --- Configure the plot aesthetics ---
    ax.set_xscale(xscale)
    ax.set_xlabel("NFE")
    ax.set_title(title)

    if ylim_top:
        ax.set_ylim(top=ylim_top)

    # Set ticks to be the actual NFE values present in the final data
    all_nfes = sorted(df["nfe"].unique())

    nfes_excl_10 = [nfe for nfe in all_nfes if nfe != 10]
    ax.set_xticks(ticks=nfes_excl_10, labels=nfes_excl_10)
    ax.xaxis.minorticks_off()

    if legend:
        ax.legend()


df = pd.read_csv(path).sort_values(["experiment", "nfe"])
df = df[df["num_samples"] == 50000]
df = df[df["experiment"].isin(name_map.keys())]
df = df[~df["experiment"].isin(ablations)]
# df = df[df["solver"] == "dpm_solver_3"]
# df = df[df["solver"] == "dpm_solver_2"]
# df = df[df["solver"] == "euler"]
# df = df[df["nfe"] >= 2]
# df = df[df["solver"] == "midpoint"]
df = df[df["nfe"] % 2 == 0]

fig, axs = plt.subplots(1, 3, figsize=(8, 3.5), sharey=True, tight_layout=True)
plot_fid_vs_nfe(
    df[df["solver"] == "midpoint"],
    axs[0],
    name_map,
    title="Midpoint",
    marker="x",
    linestyle="--",
)
plot_fid_vs_nfe(
    df[df["solver"] == "dpm_solver_3"],
    axs[1],
    name_map,
    title="DPM Solver 3",
    marker="x",
    linestyle="--",
)
plot_fid_vs_nfe(
    df,
    axs[2],
    name_map,
    title="Best",
    marker="x",
    linestyle="--",
    legend=True,
)

axs[0].set_ylabel("FID")
fig.savefig(results_dir / "fid_vs_nfe_cifar_all_samplers.pdf", bbox_inches="tight")
# %%
df = pd.read_csv(path).sort_values(["experiment", "nfe"])
df = df[df["num_samples"] == 50000]
df = df[df["experiment"].isin(name_map.keys())]
df = df[df["nfe"] % 2 == 0]

celeba_path = "[redacted]"
df_celeba = pd.read_csv(celeba_path).sort_values(["experiment", "nfe"])
df_celeba = df_celeba[df_celeba["num_samples"] == 50000]
df_celeba = df_celeba[df_celeba["nfe"] % 2 == 0]
celeba_name_map = {
    "celeba_cold_fm": "Flow Matching",
    #"2025-09-05_19-38_vibrant_fish": "Warm Flow Matching (Ours)",
    "2025-07-29_22-57_quirky_jaguar": "Warm Flow Matching"
}

fig, axs = plt.subplots(1, 3, figsize=(10, 3.8), tight_layout=True, sharey=True)
plot_fid_vs_nfe(
    df_celeba,
    axs[0],
    celeba_name_map,
    title="CelebA",
    marker="x",
    linestyle="--",
    legend=False,
)

plot_fid_vs_nfe(
    df[~df["experiment"].isin(ablations)],
    axs[1],
    name_map,
    title="CIFAR-10",
    marker="x",
    linestyle="--",
    legend=False,
)
plot_fid_vs_nfe(
    df,
    axs[2],
    name_map,
    title="CIFAR-10 (All Ablations)",
    marker="x",
    linestyle="--",
    legend=True,
)
axs[0].set_ylabel("FID")
fig.savefig(results_dir / "fid_vs_nfe_best_all.pdf", bbox_inches="tight")


# %%
# %%

df = pd.read_csv(path)
df = df[df["nfe"] == 4]
df = df[df["experiment"] == "2025-07-23_15-24_sassy_unicorn_better_cnp"].sort_values(
    "fid"
)
df.sort_values("fid")
# %%
# df[df["experiment"] == "2025-07-21_22-38_playful_xenon"]
nfes = df["nfe"].unique()
best_solvers = {}
for nfe in nfes:
    sub_df = df[df["experiment"] == "new_warmth_scaling"].sort_values("fid")
    sub_df = sub_df[sub_df["nfe"] == nfe]
    best_solver = sub_df[sub_df["fid"] == sub_df["fid"].min()]["solver"]
    best_solvers[int(nfe)] = best_solver.values[0]
best_solvers
# %%
df = pd.read_csv(path)
df = df[df["nfe"] == 10].sort_values("fid")
df
# %%

df = pd.read_csv(path)
df = df[df["experiment"] == "2025-07-21_22-38_playful_xenon"]
df = df.sort_values(["nfe", "fid"])
df.head(50)
# %%
df
