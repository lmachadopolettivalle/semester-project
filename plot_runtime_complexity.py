from matplotlib import pyplot as plt
import numpy as np

BLUE = "#004488"

# Runtimes in seconds for each zmax
runtimes = {
    0.32: 300,
    0.93: 500,
    1.75: 650,
    3.5: 800,
}

BOXSIZE = 900
RUNINDEX = 0
DATAPATH = "/cluster/work/refregier/alexree/isw_on_lightcone/cosmo_grid_sims/benchmarks"
RUN_PATHS = {
    900: "fiducial_bench",
    2250: "box_size",
}
FILEPATH = f"{DATAPATH}/{RUN_PATHS[BOXSIZE]}/run_{RUNINDEX}"

info = np.load(f"{FILEPATH}/shell_info.npy")
z_bins = np.array([
    (row[2], row[3])
    for row in info
])

redshifts = [0.32, 0.93, 1.75, 3.5]
number_of_z_bins = [
    len([i for i in z_bins if i[0] < z])
    for z in redshifts
]

fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.4))

ax.errorbar(
    number_of_z_bins,
    runtimes.values(),
    yerr=20,
    c=BLUE,
    marker="o",
)

for bins, (zmax, runtime) in zip(number_of_z_bins, runtimes.items()):
    ax.annotate(
        f"zmax = {zmax:.2f}",
        xy=(bins-5, runtime+40),
        xycoords="data",
        bbox={"boxstyle":"square","fc":"white"}
    )

ax.set_xlabel("Number of Shells", fontsize=12)
ax.set_ylabel("Runtime [seconds]", fontsize=12)

ax.set_xlim([10, 80])
ax.set_ylim([250, 950])

ax.set_title("Runtime Complexity vs. Number of Shells", fontsize=12)

ax.grid()

plt.savefig("images/runtime_complexity.pdf")

plt.show()
