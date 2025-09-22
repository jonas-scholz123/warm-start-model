# %%

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

exp_names = {
    "2025-09-01_16-29_jolly_whale": "Flow Matching + DPM Solver",
    "2025-09-05_13-18_radiant_hippo": "Warm Flow Matching (Ours)",
}

exp_nfes = {
    "2025-09-01_16-29_jolly_whale": [2, 8, 10, 12, 14, 16, 20, 30],
    "2025-09-05_13-18_radiant_hippo": [2, 4, 8, 10, 20],
}

all_nfes = sorted(list(set(nfe for nfes in exp_nfes.values() for nfe in nfes)))
colors = {nfe: f"C{idx}" for idx, nfe in enumerate(all_nfes)}

var = "10m_u_component_of_wind"
var_name = "10m U Wind"
# var = "10m_v_component_of_wind"
# var_name = "10m V Wind"

normed = False

fig, axs = plt.subplots(
    1, len(exp_names), figsize=(5 * len(exp_names), 2.8), sharey=True
)

for i, (exp_name, title) in enumerate(exp_names.items()):
    csv_path = f"../../wind_results_{exp_name}.csv"
    df = pd.read_csv(csv_path)

    metric = "crps"
    # metric = "ensemble_rmse"

    metric_names = {
        "crps": "CRPS",
        "ensemble_rmse": "Ensemble RMSE",
    }

    metrics = list(metric_names.keys())

    df = df[df["n_members"] == 50]
    df = df[df["n_0_times"] == 40]
    df = df[df["metric"] == metric]
    nfes = exp_nfes[exp_name]
    nfes = [nfe for nfe in nfes if nfe in df["nfe"].unique()]
    # df = df[df["nfe"] != 6]
    # df = df[df["solver"] == "dpm_solver_3"]

    var_df = df[df["variable"] == var]

    nfe_df = var_df[var_df["nfe"] == max(nfes)]
    best_values = nfe_df["value"].values

    for j, nfe in enumerate(nfes):
        nfe_df = var_df[var_df["nfe"] == nfe]
        nfe_df = nfe_df.sort_values("time_delta_hrs")
        time_delta_hrs = nfe_df["time_delta_hrs"].values
        values = nfe_df["value"].values
        axs[i].plot(
            time_delta_hrs,
            values / best_values if normed else values,
            marker="x",
            label=f"NFE={nfe}",
            color=colors[nfe],
        )
        time_delta_days = time_delta_hrs // 24
        ticks = time_delta_hrs[3::4]
        labels = time_delta_days[3::4].astype(int)
        axs[i].set_xticks(ticks, labels=labels)
        axs[i].set_xlabel("Lead Time (days)")
        axs[i].set_title(title)
        axs[i].set_ylim(1.6, 2.25)
        axs[i].set_xlim(2 * 24, 5 * 24 + 3)
    axs[0].legend(ncols=1)
    axs[1].legend(ncols=1)
    axs[0].grid(True, linestyle="--", alpha=0.3)
    axs[1].grid(True, linestyle="--", alpha=0.3)
    axs[0].set_ylabel(f"{metric_names[metric]} for {var_name}")
# Reduce horizontal space between subplots:
fig.subplots_adjust(wspace=0.05)
fig.savefig(f"crps_{var}.pdf", bbox_inches="tight")
# %%
experiments = [
    "2025-09-01_16-29_jolly_whale",
    "2025-09-05_13-18_radiant_hippo",
]

experiment_to_title = {
    "2025-09-01_16-29_jolly_whale": "Flow Matching",
    "2025-09-05_13-18_radiant_hippo": "Warm FM (Ours)",
}

results = {}
for experiment in experiments:
    with open(f"../../frequency_results_{experiment}.pkl", "rb") as f:
        results[experiment] = pickle.load(f)
        wl: np.ndarray = results[experiment]["wl"]

samplers = ["dpm_solver_3", "midpoint", "best"]
cbar_width = 0.05
cbar_space = 0.3
fig, axs = plt.subplots(
    len(experiments),
    len(samplers) + 2,
    figsize=(4 * len(samplers) + cbar_width + cbar_space, 3.2 * len(experiments)),
    sharex=False,
    sharey=False,
    gridspec_kw={"width_ratios": [1, 1, cbar_width, cbar_space, 1.2]},
)
fig.subplots_adjust(wspace=0.1, hspace=0.2)


wl = wl.round(-1).astype(int)

yticklabels = [w if idx % 8 == 0 else "" for idx, w in enumerate(wl)]
nfes = [2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 50, 100]
plot_nfes = [2, 4, 6, 8, 10, 14, 20, 30, 50, 100]
vmin = 0.5
vmax = 1.5

for col, experiment in enumerate(experiments):
    sampler_results = results[experiment]
    plottable = sampler_results["midpoint"]
    sns.heatmap(
        plottable.T,
        xticklabels=nfes,
        yticklabels=yticklabels,
        cmap="vlag",
        cbar_ax=axs[0, 2],
        vmin=0.6,
        vmax=1.1,
        center=1.0,
        ax=axs[0, col],
    )

    plottable = sampler_results["dpm_solver_3"]
    sns.heatmap(
        plottable.T,
        xticklabels=nfes,
        yticklabels=yticklabels,
        cmap="vlag",
        cbar_ax=axs[1, 2],
        vmin=vmin,
        vmax=vmax,
        center=1.0,
        ax=axs[1, col],
    )

    axs[-1, 0].set_xlabel("NFE")
    axs[-1, 1].set_xlabel("NFE")
    axs[0, 0].set_title("$\eta(\lambda)$, Flow Matching")
    axs[0, 1].set_title("$\eta(\lambda)$, Warm FM (Ours)")
    axs[0, 0].set_ylabel("Midpoint Solver\n\n $\lambda$ (km)")
    axs[1, 0].set_ylabel("DPM Solver\n\n $\lambda$ (km)")
    axs[0, 1].set_yticks([], labels=[])
    axs[1, 1].set_yticks([], labels=[])

    plottable = sampler_results["midpoint"]
    abs_diff = np.abs(plottable - 1.0).sum(axis=1)
    axs[-1, 4].set_xlabel("NFE")
    axs[0, 4].set_title("Spectrum Deviation $\sum_\lambda |1 - \eta(\lambda)|$")
    axs[0, 4].plot(nfes, abs_diff, marker="x", label=experiment_to_title[experiment])
    axs[0, 4].set_xscale("log")
    axs[0, 4].legend()
    axs[0, 4].set_xticks(plot_nfes, labels=plot_nfes)

    plottable = sampler_results["dpm_solver_3"]
    abs_diff = np.abs(plottable - 1.0).sum(axis=1)
    axs[1, 4].plot(nfes, abs_diff, marker="x", label=experiment_to_title[experiment])
    axs[1, 4].set_xscale("log")
    axs[1, 4].legend()
    axs[1, 4].set_xticks(plot_nfes, labels=plot_nfes)

    axs[0, 3].xaxis.minorticks_off()
    axs[1, 3].xaxis.minorticks_off()

    axs[0, 0].tick_params(axis="y", labelsize=9)
    axs[1, 0].tick_params(axis="y", labelsize=9)

    axs[0, 3].set_visible(False)
    axs[1, 3].set_visible(False)


fig.savefig("weather_forecasting_eval.pdf", bbox_inches="tight")

plt.show()
# %%
# yticklabels = [f"{w:.0e}" if idx % 5 == 0 else "" for idx, w in enumerate(wl.numpy())]
experiment
