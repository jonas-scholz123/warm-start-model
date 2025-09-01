# %%
import matplotlib.pyplot as plt
import pandas as pd

path = "/home/jonas/Documents/code/denoising-np/fid_results.csv"
df = pd.read_csv(path).sort_values(["experiment", "nfe"])
#df = df[df["solver"] == "dpm_solver_3"]
#df = df[df["solver"] == "dpm_solver_2"]
#df = df[df["solver"] == "euler"]
# df = df[df["nfe"] >= 2]
#df = df[df["solver"] == "midpoint"]
df = df[df["nfe"] % 2 == 0]

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