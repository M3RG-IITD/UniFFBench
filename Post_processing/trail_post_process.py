# %%

import csv
import os
import uuid
from glob import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import colors as mcolors
from tqdm import tqdm

from utils import (
    FileHandler,
    PropertyCalculator,
    find_missing_csv_files_v8,
    plot_scatter,
    process_file,
    save_bond_errors_to_txt,
    save_to_csv,
)

# %% [markdown]
# ## Set up analysis folders and constants

# %%
models = ["MACE", "M3GNet", "CHGNet", "MatterSim", "Orb"]  # , "SevenNet"]
model_map = {model.lower(): model for model in models}
colors = ["#698B66", "#D04F81", "#9069A1", "#9DC183", "#F4C2C2", "#D7BDE2"]


# root_folder = "/share/datasets-05/aimat_uip/uip_results_0/orb/"
root_folder = "/Users/sajid/Documents/github/UIP_EVAL/Post_processing/mace"


model_name = Path(root_folder).name
results_folder = "/Users/sajid/Documents/github/UIP_EVAL/Post_processing/results"

os.makedirs(f"{results_folder}/{model_name}/figs", exist_ok=True)
os.makedirs(f"{results_folder}/figs", exist_ok=True)

# %% [markdown]
# ## Overall Model's completion comparison

# %% [markdown]
# #### Change the root_folder and model_name to run for different models

# %%
combined_df, missing_csv_dirs, unreadable_csv_dirs = find_missing_csv_files_v8(
    root_folder, model_name, results_folder
)

# print(model_name)
completion_dict = yaml.safe_load(open(f"{results_folder}/completion_dict.yaml", "r"))
# print(completion_dict)
completion_dict[model_map[model_name]] = {
    "completed_simulations": len(combined_df),
    "total_folders": len(combined_df) + len(missing_csv_dirs),
}
with open(f"{results_folder}/completion_dict.yaml", "w") as f:
    yaml.safe_dump(completion_dict, f)

# %% [markdown]
# ## This is just for plotting the values obtained from the model completion

# %%
models = list(completion_dict.keys())
total_folders = [info["total_folders"] for info in completion_dict.values()]
completed_simulations = [
    info["completed_simulations"] for info in completion_dict.values()
]

# Calculate fractions
completed_fractions = [
    completed / total for completed, total in zip(completed_simulations, total_folders)
]

# Calculate the fraction of incomplete simulations
incomplete_fractions = [
    1 - completed_fraction for completed_fraction in completed_fractions
]

# Create lighter colors for incomplete simulations by reducing alpha values
incomplete_colors = [mcolors.to_rgba(color, alpha=0.4) for color in colors]

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Stack bars for completed and incomplete simulations
completed_bars = ax.bar(
    models, completed_fractions, color=colors, label="Completed Simulations"
)
incomplete_bars = ax.bar(
    models,
    incomplete_fractions,
    bottom=completed_fractions,
    color=incomplete_colors,
    label="Incomplete Simulations",
)

# Add percentage text labels on the bars
for i, model in enumerate(models):
    # Add text for completed simulations (green color)
    ax.text(
        model,
        completed_fractions[i] / 2,
        f"{completed_fractions[i]*100:.1f}%",
        ha="center",
        va="center",
        color="blue",
        fontsize=12,
    )

    # Add text for incomplete simulations (red color)
    ax.text(
        model,
        completed_fractions[i] + incomplete_fractions[i] / 2,
        f"{incomplete_fractions[i]*100:.1f}%",
        ha="center",
        va="center",
        color="red",
        fontsize=12,
    )

# Labels and title
ax.set_ylabel("Fraction of Simulations")
ax.set_ylim(0, 1.1)  # Set limit for y-axis
# ax.set_title('Fraction of Completed & Incomplete Simulations')

# Show plot
plt.show()
plt.savefig(f"{results_folder}/figs/fraction_of_simulations.png")

# %% [markdown]
# ## Parity plots

# %% [markdown]
# #### Read the csv data generated for the different model

# %%
combined_data_mace = pd.read_csv(f"{results_folder}/{model_name}/{model_name}.csv")

# %% [markdown]
# #### Just change model name for different model and set unfiltered_parity = True to plot unfiltered data
# 

# %%
# Create figures
fig_density, ax_density = plt.subplots(figsize=(6, 6))
fig_lattice, ax_lattice = plt.subplots(figsize=(6, 6))
# Density data
act_density = combined_data_mace["Exp_Density (g/cm³)"].values
pred_density = combined_data_mace["Sim_Density (g/cm³)"].values
act_a = combined_data_mace["Exp_a (Å)"].values
pred_a = combined_data_mace["Sim_a (Å)"].values
act_b = combined_data_mace["Exp_b (Å)"].values
pred_b = combined_data_mace["Sim_b (Å)"].values

act_c = combined_data_mace["Exp_c (Å)"].values
pred_c = combined_data_mace["Sim_c (Å)"].values


unfiltered_parity = False  # Set to True to plot unfiltered data

# Initialize dictionary to store all R2 scores
r2_scores_dict = {}

# Define marker styles
marker_density = "D"  # Diamond
markers = ["o", "s", "^"]  # Circle, Square, Triangle

# Apply masks
mask_density = (pred_density <= 1.5 * act_density) & (pred_density >= 0.5 * act_density)
mask_a = (pred_a <= 1.5 * act_a) & (pred_a >= 0.5 * act_a)
mask_b = (pred_b <= 1.5 * act_b) & (pred_b >= 0.5 * act_b)
mask_c = (pred_c <= 1.5 * act_c) & (pred_c >= 0.5 * act_c)
mask_final = mask_density & mask_a & mask_b & mask_c

if unfiltered_parity:
    mask_final = np.ones_like(mask_final, dtype=bool)  # All True

# Plot density data
r2_density, removed_sys = plot_scatter(
    ax_density,
    mask_final,
    act_density,
    pred_density,
    "Density (g/cm³)",
    "m",
    marker_density,
    model_name,
    r2_scores_dict,
)

# Plot lattice parameters
r2_scores = []
for param, act, pred, color, marker in zip(
    ["Cell Parameter a (Å)", "Cell Parameter b (Å)", "Cell Parameter c (Å)"],
    [act_a, act_b, act_c],
    [pred_a, pred_b, pred_c],
    ["r", "g", "b"],
    markers,
):
    r2, removed = plot_scatter(
        ax_lattice,
        mask_final,
        act,
        pred,
        param,
        color,
        marker,
        model_name,
        r2_scores_dict,
    )
    r2_scores.append(r2)

# Set titles and labels
ax_density.set_title(f"Density\n$R^2$ Score: {r2_density:.2f}", fontsize=16)
ax_density.set_xlabel("Experimental Density (g/cm³)", fontsize=16)
ax_density.set_ylabel("Simulated Density (g/cm³)", fontsize=16)
ax_density.legend()
fig_density.savefig(f"{results_folder}/{model_name}/figs/density_r2_scores.png")

overall_r2 = (
    f"a: {r2_scores[0]:.2f}, b: {r2_scores[1]:.2f}, c: {r2_scores[2]:.2f}"
    if all(not np.isnan(r2) for r2 in r2_scores)
    else "N/A"
)
ax_lattice.set_title(f"Lattice Parameters\n$R^2$ Scores: {overall_r2}", fontsize=16)
ax_lattice.set_xlabel("Experimental Lattice Parameters (Å)", fontsize=16)
ax_lattice.set_ylabel("Simulated Lattice Parameters (Å)", fontsize=16)
ax_lattice.legend(loc="upper left")
fig_lattice.savefig(f"{results_folder}/{model_name}/figs/lattice_r2_scores.png")

r2_scores = yaml.safe_load(open(f"{results_folder}/r2_scores.yaml", "r"))
r2_scores[model_map[model_name]] = r2_scores_dict
# with open(f"{results_folder}/r2_scores.yaml", "w") as f:
#     yaml.safe_dump(r2_scores, f)

# %% [markdown]
# ### This is just for plotting the R2 score saved in txt file from the above run

# %%
# Example Data
metrics = [
    "Density (g/cm³)",
    "Cell Parameter a (Å)",
    "Cell Parameter b (Å)",
    "Cell Parameter c (Å)",
]  # Bars in each group

# Bar settings
x = np.arange(len(models))  # Group positions
width = 0.2  # Width of each bar

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars for each metric with custom colors
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(
        x + i * width,
        [r2_scores[model][metric] for model in models],
        width,
        label=metric,
        color=color,
    )

# Customize plot
# ax.set_xlabel('Models', fontsize=14)
ax.set_ylabel("$R^2$ Score", fontsize=16)
ax.set_xticks(x + width * 1.5)  # Adjust group position
ax.set_xticklabels(models, fontsize=16)

# Position legend over bars
ax.legend(
    fontsize=16,
    title_fontsize=12,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.2),
    ncol=2,
)

# Add grid and display
# ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.savefig(f"{results_folder}/figs/r2_scores.png")
plt.close()
# %% [markdown]
# ## Trajectory based analysis - Set up to run with multiprocessing.

# %% [markdown]
# #### Just change the root folder name and model_name for different model. 
# 
# Splits processing up into `num_cpus()-2` processes and saves out a csv for each split.

# %%
def process_slice(args):
    root_folder, model_name, slice_number, xyz_files_slice, log_files_slice = args
    unique_uuid = uuid.uuid1().__str__()
    file_handler = FileHandler(root_folder, incoming_uuid=unique_uuid)
    calculator = PropertyCalculator()

    master_densities = []
    master_lattice_params = []
    master_temperature = []
    master_rdf_values = []
    master_time_temp_data = []
    os.makedirs(f"{model_name}/results/slice_{slice_number}", exist_ok=True)
    for (system_name, xyz_file_path), (_, log_file_path) in tqdm(
        zip(xyz_files_slice, log_files_slice),
        total=len(xyz_files_slice),
        desc=f"Processing Slice {slice_number}",
    ):
        (
            densities,
            lattice_params,
            temperature,
            rdf_error,
            time_temp_data,
            bond_error,
        ) = process_file(
            file_handler, calculator, system_name, xyz_file_path, log_file_path
        )

        bond_error_file_name = (
            f"{model_name}/results/slice_{slice_number}/bond_errors_{model_name}.txt"
        )
        save_bond_errors_to_txt(bond_error_file_name, bond_error)

        master_densities.append(densities)
        master_lattice_params.append(lattice_params)
        master_temperature.append(temperature)
        master_rdf_values.append(rdf_error)
        master_time_temp_data.append(time_temp_data)

    os.makedirs(f"{model_name}/results/slice_{slice_number}", exist_ok=True)
    save_to_csv(
        f"{model_name}/results/slice_{slice_number}/master_densities_{model_name}.csv",
        master_densities,
    )
    save_to_csv(
        f"{model_name}/results/slice_{slice_number}/master_lattice_params_{model_name}.csv",
        master_lattice_params,
    )
    save_to_csv(
        f"{model_name}/results/slice_{slice_number}/master_temperature_{model_name}.csv",
        [[temp] for temp in master_temperature],
    )
    save_to_csv(
        f"{model_name}/results/slice_{slice_number}/master_rdf_values_{model_name}.csv",
        master_rdf_values,
    )
    save_to_csv(
        f"{model_name}/results/slice_{slice_number}/master_time_temp_data_{model_name}.csv",
        master_time_temp_data,
    )
    print(f"Data saved for slice {slice_number} with model name '{model_name}'.")


def process_traj_to_csv(root_folder, model_name):
    file_handler = FileHandler(root_folder, incoming_uuid=uuid.uuid1().__str__())
    xyz_files, log_files = file_handler.find_xyz_files()

    total_files = len(xyz_files)
    num_slices = cpu_count() - 2
    slice_size = total_files // num_slices

    args_list = [
        (
            root_folder,
            model_name,
            slice_number,
            xyz_files[slice_number * slice_size : (slice_number + 1) * slice_size],
            log_files[slice_number * slice_size : (slice_number + 1) * slice_size],
        )
        for slice_number in range(num_slices)
    ]

    with Pool(num_slices) as pool:
        pool.map(process_slice, args_list)


process_traj_to_csv(root_folder, model_name)

# %% [markdown]
# #### Combines all splits into `./results_folder/model_name/all`.

# %%
def combine_csv_files(input_pattern, output_file):
    combined_data = []
    headers = set()

    # First pass to collect all unique headers
    for csv_file in glob(input_pattern, recursive=True):
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            file_header = tuple(next(reader))
            headers.update(file_header)

    headers = sorted(headers)
    header_index_map = {header: index for index, header in enumerate(headers)}

    # Second pass to read data and align columns
    for csv_file in glob(input_pattern, recursive=True):
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            file_header = tuple(next(reader))
            file_header_index_map = {
                header: index for index, header in enumerate(file_header)
            }

            for row in reader:
                aligned_row = [np.nan] * len(headers)
                for header, value in zip(file_header, row):
                    aligned_row[header_index_map[header]] = value
                combined_data.append(aligned_row)

    # Write combined data to output file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(combined_data)


def combine_bond_error_files(input_pattern, output_file):
    with open(output_file, "w") as outfile:
        for txt_file in glob(input_pattern, recursive=True):
            with open(txt_file, "r") as infile:
                outfile.write(infile.read())


def combine_all_csvs(model_name):
    all_folder = os.path.join(f"{results_folder}/{model_name}", "all")
    os.makedirs(all_folder, exist_ok=True)

    combine_csv_files(
        os.path.join(
            f"{results_folder}/{model_name}",
            "slice_*",
            f"master_densities_{model_name}.csv",
        ),
        os.path.join(all_folder, f"master_densities_{model_name}.csv"),
    )
    combine_csv_files(
        os.path.join(
            f"{results_folder}/{model_name}",
            "slice_*",
            f"master_lattice_params_{model_name}.csv",
        ),
        os.path.join(all_folder, f"master_lattice_params_{model_name}.csv"),
    )
    combine_csv_files(
        os.path.join(
            f"{results_folder}/{model_name}",
            "slice_*",
            f"master_temperature_{model_name}.csv",
        ),
        os.path.join(all_folder, f"master_temperature_{model_name}.csv"),
    )
    combine_csv_files(
        os.path.join(
            f"{results_folder}/{model_name}",
            "slice_*",
            f"master_rdf_values_{model_name}.csv",
        ),
        os.path.join(all_folder, f"master_rdf_values_{model_name}.csv"),
    )
    combine_csv_files(
        os.path.join(
            f"{results_folder}/{model_name}",
            "slice_*",
            f"master_time_temp_data_{model_name}.csv",
        ),
        os.path.join(all_folder, f"master_time_temp_data_{model_name}.csv"),
    )

    # Combine bond error files
    combine_bond_error_files(
        os.path.join(f"{results_folder}/{model_name}", "slice_*", "bond_errors_*.txt"),
        os.path.join(all_folder, f"bond_errors_{model_name}.txt"),
    )

    print(f"All data combined and saved in {all_folder}")


combine_all_csvs(model_name)

# %% [markdown]
# #### Read the saved master csv file for density, rdf and plot the time progress of error

# %%
master_densities = pd.read_csv(
    f"results/{model_name}/all/master_densities_mattersim.csv"
)

# %%
master_densities = np.array(master_densities)

# Calculate percentage error from initial value for each trajectory
initial_densities = master_densities[:, 0:1]
percentage_errors = (
    -1 * ((master_densities - initial_densities) / initial_densities) * 100
)

# Define error bins
error_bins = [0, 2, 5, 10, np.inf]
bin_labels = ["[0, 2)%", "[2, 5)%", "[5, 10)%", "[10, -∞)%"]

# Count trajectories in each bin at each timestep
timesteps = master_densities.shape[1]
binned_counts = np.zeros((len(bin_labels), timesteps))

for t in range(timesteps):
    bins = np.digitize(percentage_errors[:, t], error_bins[:-1])
    for i in range(len(bin_labels)):
        binned_counts[i, t] = np.sum(bins == i + 1)

# Calculate mean and standard deviation of percentage errors
mean_errors = np.mean(percentage_errors, axis=0)
std_errors = np.std(percentage_errors, axis=0)

# Define the x-axis values (timesteps)
x_values = np.arange(timesteps)

# Create the combined plot
fig, ax1 = plt.subplots(figsize=(15, 6))

# Plot the area plot (stackplot) on the first y-axis
stack = ax1.stackplot(
    np.log(x_values + 1), binned_counts, labels=bin_labels, colors=colors
)
ax1.set_xlabel("Timesteps (log scale)")
ax1.set_ylabel("Number of Simulations")

# Create a second y-axis
ax2 = ax1.twinx()

# Plot the mean trajectory with error fill on the second y-axis
(mean_line,) = ax2.plot(
    np.log(x_values + 1), mean_errors, label="Mean Error Trajectory", color="blue"
)
ax2.set_ylabel("Percentage Density Error (%)", color="blue")
# ax2.set_ylim(0, 19)  # Adjust based on the data
ax2.tick_params(axis="y", labelcolor="blue")
ax2.spines["right"].set_color("blue")

# Combine legends into one
(
    handles,
    labels,
) = ax1.get_legend_handles_labels()  # Get handles and labels from stackplot
handles.append(mean_line)  # Add the mean trajectory line
labels.append("Mean Error Trajectory")  # Add the corresponding label

# Add a single legend with appropriate size and placement
ax1.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),  # Adjusted position for combined legend
    fontsize="small",  # Set smaller font size for better fit
    frameon=False,  # Remove legend box outline for a cleaner look
    ncol=len(bin_labels) + 1,  # Adjust number of columns
)

# Add "Error Range" text
plt.text(
    -0.05, 1.02, "Error Ranges:", transform=plt.gca().transAxes, ha="left", fontsize=16
)
plt.tight_layout()
plt.show()
plt.savefig(f"{results_folder}/figs/error_ranges.png")
plt.close()
