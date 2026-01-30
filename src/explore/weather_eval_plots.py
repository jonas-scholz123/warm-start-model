#%%
import matplotlib.pyplot as plt
import pandas as pd

exp_name = "2025-09-01_16-29_jolly_whale"
csv_path = f"/home/jonas/Documents/code/denoising-np/_results/wind_results_{exp_name}.csv"
#csv_path = f"../../wind_results_{exp_name}.csv"
df = pd.read_csv(csv_path)

#metric = "crps"
metric = "ensemble_rmse"

metric_names = {
    "crps": "CRPS",
    "ensemble_rmse": "Ensemble RMSE",
}

df = df[df["n_0_times"] == 80]
df = df[df["metric"] == metric]
# df = df[df["nfe"] != 2]
df = df[df["solver"] == "dpm_solver_3"]

nfes = sorted(df["nfe"].unique())

trg_vars = {
    "10m_u_component_of_wind": "10m U Wind",
    "10m_v_component_of_wind": "10m V Wind",
}


fig, axs = plt.subplots(1, len(trg_vars), figsize=(4 * len(trg_vars), 3))

for i, (var, var_name) in enumerate(trg_vars.items()):
    var_df = df[df["variable"] == var]
    for nfe in nfes:
        nfe_df = var_df[var_df["nfe"] == nfe]
        nfe_df = nfe_df.sort_values("time_delta_hrs")
        time_delta_hrs = nfe_df["time_delta_hrs"].values
        values = nfe_df["value"].values
        axs[i].plot(
            time_delta_hrs,
            values,
            marker="x",
            label=f"NFE={nfe}",
        )
        time_delta_days = time_delta_hrs // 24
        ticks = time_delta_hrs[3::4]
        labels = time_delta_days[3::4].astype(int)
        axs[i].set_xticks(ticks, labels=labels)
        axs[i].set_xlabel("Lead Time (days)")
        axs[i].set_title(var_name)
axs[1].legend()
axs[0].set_ylabel(metric_names[metric])
