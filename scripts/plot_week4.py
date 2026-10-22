
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv("results/week4.csv")

assert (df["reason"] == "CONVERGED_RTOL").all(), "some runs did not converge!"

df["t_total"] = df["t_setup"] + df["t_solve"]

best = df.groupby(["ksp", "pc", "np", "M"], as_index=False).agg(
    its=("its", "first"),
    t_total=("t_total", "min"),
)
print(best)

matplotlib.use("Agg")
COLORS  = {"none": "#2a78d6", "jacobi": "#eb6834", "ilu": "#1baf7a", "gamg": "#eda100"}
MARKERS = {"none": "o", "jacobi": "s", "ilu": "^", "gamg": "D"}

exp1 = best[(best["np"] == 1) & (best["pc"].isin(COLORS.keys()))]
fig, ax = plt.subplots(figsize=(6, 4.5))
for pc in ["none", "jacobi", "ilu", "gamg"]:
    d = exp1[exp1["pc"] == pc]
    ax.plot(d["M"], d["its"], color=COLORS[pc], marker=MARKERS[pc],
            linewidth=2, markersize=8, label=pc)

# reference slope: iterations proportional to M
Ms = exp1["M"].unique()
ax.plot(Ms, 0.5 * 233 * Ms / 65, "--", color="gray", linewidth=1, label="∝ M (slope 1)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("grid points per direction M")
ax.set_ylabel("CG iterations")
ax.set_title("CG iterations vs. grid size (random rhs, rtol 1e-8)")
ax.grid(True, which="major", color="#e0e0e0", linewidth=0.8)
ax.legend()
fig.tight_layout()
fig.savefig("results/iterations.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 4.5))
for pc in ["none", "jacobi", "ilu", "gamg"]:
    d = exp1[exp1["pc"] == pc]
    ax.plot(d["M"]**2, d["t_total"], color=COLORS[pc], marker=MARKERS[pc],
            linewidth=2, markersize=8, label=pc)

# reference slope: iterations proportional to M
Ms = exp1["M"].unique()
N = Ms**2
t_gamg0 = exp1[(exp1["pc"] == "gamg") & (exp1["M"] == Ms[0])]["t_total"].iloc[0]
t_none0 = exp1[(exp1["pc"] == "none") & (exp1["M"] == Ms[0])]["t_total"].iloc[0]
ax.plot(N, 0.5 * t_gamg0 * (N / N[0]),        "--", color="gray", linewidth=1, label="∝ N")
ax.plot(N, 0.5 * t_none0 * (N / N[0])**1.5,   ":",  color="gray", linewidth=1, label="∝ N^1.5")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("unknowns N = M^2")
ax.set_ylabel("solver time (s), best of 3")
ax.set_title("CG solver time vs. problem size (1 process)")
ax.grid(True, which="major", color="#e0e0e0", linewidth=0.8)
ax.legend()
fig.tight_layout()
fig.savefig("results/time.png", dpi=150)
