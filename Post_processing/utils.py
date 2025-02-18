import csv
import os
import re
import sys
import uuid
from datetime import datetime
from typing import List, Tuple
from warnings import filterwarnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase import io
from ase.atoms import Atoms
from ase.geometry.analysis import Analysis
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from tqdm import tqdm

# Suppress warnings
filterwarnings('ignore')


# Function to find missing CSV files
def find_missing_csv_files_v8(root_folder, model_name, results_folder):
    """
    Combines all Data.csv files from .cif subdirectories under the given root folder.
    Generates a combined CSV and reports missing or unreadable files.

    Args:
        root_folder (str): The root directory containing subdirectories to scan.
        model_name (str): The name to use for the output files.

    Returns:
        Tuple: Combined DataFrame, missing CSV directories, and unreadable CSV directories.
    """
    df_list = []
    missing_csv_dirs = []  # Directories where Data.csv is missing
    unreadable_csv_dirs = []  # Directories with unreadable Data.csv
    successfully_read_csvs = []  # Paths of CSVs successfully read
    no_cif_folders = []  # Folders without .cif subdirectories
    hidden_folders = []  # Hidden folders
    all_cif_paths = []  # Track all .cif folders found

    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        subdir_path = os.path.abspath(subdir_path)

        if subdir.startswith('.'):
            hidden_folders.append(subdir_path)
            continue

        if os.path.isdir(subdir_path):
            version_0_path = os.path.join(subdir_path, 'version_0')

            if os.path.isdir(version_0_path):
                cif_folders = [
                    f for f in os.listdir(version_0_path)
                    if f.endswith('.cif') and os.path.isdir(os.path.join(version_0_path, f))
                ]

                if cif_folders:
                    for cif_folder in cif_folders:
                        cif_folder_path = os.path.join(version_0_path, cif_folder)
                        all_cif_paths.append(cif_folder_path)

                        csv_path = os.path.join(cif_folder_path, 'Data.csv')

                        if os.path.isfile(csv_path):
                            try:
                                df = pd.read_csv(csv_path)
                                successfully_read_csvs.append(csv_path)
                                df_list.append(df)
                            except Exception as e:
                                unreadable_csv_dirs.append(cif_folder_path)
                        else:
                            missing_csv_dirs.append(cif_folder_path)
                else:
                    no_cif_folders.append(version_0_path)
            else:
                no_cif_folders.append(subdir_path)

    print("\n=== Detailed Path Analysis ===")
    print(f"Total .cif folders found: {len(all_cif_paths)}")
    print(f"Successfully read Data.csv: {len(successfully_read_csvs)}")
    print(f"Missing Data.csv: {len(missing_csv_dirs)}")
    print(f"Unreadable Data.csv: {len(unreadable_csv_dirs)}")

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_csv(f"{results_folder}/{model_name}/{model_name}.csv", index=False)

        with open(f"{results_folder}/{model_name}/fraction_complete_{model_name}.txt", 'w') as file:
            file.write(f"{results_folder}/{model_name}\t Total .cif folders found: {len(all_cif_paths)}" +
                       f"\t\tSuccessfully read Data.csv: {len(successfully_read_csvs)}")

        return combined_df, missing_csv_dirs, unreadable_csv_dirs

    print("\nNo Data.csv files found in the specified folders.")
    return None, missing_csv_dirs, unreadable_csv_dirs

# Function for plotting parity plots
def plot_scatter(ax, mask, actual, predicted, parameter_name, color, marker, model_name, r2_dict):
    """
    Creates a scatter plot comparing actual vs predicted values with an optional mask.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        mask (np.ndarray): Mask to filter data.
        actual (np.ndarray): Actual values.
        predicted (np.ndarray): Predicted values.
        parameter_name (str): Name of the parameter being plotted.
        color (str): Color of the points.
        marker (str): Marker style.
        model_name (str): Name of the model.
        r2_dict (dict): Dictionary to store R2 values.

    Returns:
        Tuple: R2 score and indices of removed points.
    """
    filtered_actual = actual[mask]
    filtered_predicted = predicted[mask]

    r2 = r2_score(filtered_predicted, filtered_actual) if len(filtered_actual) > 0 else np.nan
    r2_dict[parameter_name] = r2

    ax.scatter(filtered_actual, filtered_predicted, label=f'{parameter_name}', color=color, marker=marker)
    max_val = max(max(filtered_actual), max(filtered_predicted)) if len(filtered_actual) > 0 else 1
    ax.plot([0, max_val], [0, max_val], color='black', linestyle='--')

    return r2, np.where(mask == False)

# Class for handling file operations
class FileHandler:
    """
    Handles file searching and reading operations.
    """

    def __init__(self, root_folder: str, incoming_uuid: str):
        self.root_folder = root_folder
        self.uuid = incoming_uuid

    def find_xyz_files(self) -> List[Tuple[str, str]]:
        xyz_files = []
        log_files = []

        for subdir in os.listdir(self.root_folder):
            subdir_path = os.path.join(self.root_folder, subdir)
            if not os.path.isdir(subdir_path) or subdir.startswith("."):
                continue

            version_0_path = os.path.join(subdir_path, "version_0")
            if not os.path.isdir(version_0_path):
                continue

            for folder in os.listdir(version_0_path):
                folder_path = os.path.join(version_0_path, folder)
                if not os.path.isdir(folder_path) or not folder.endswith(".cif"):
                    continue

                for file in os.listdir(folder_path):
                    if file.endswith(".xyz") and not file.startswith("._"):
                        system_name = folder.replace(".cif", "").replace(" ", "_")
                        xyz_files.append((system_name, os.path.join(folder_path, file)))
                    if file.endswith(".log") and not file.startswith("._"):
                        system_name = folder.replace(".cif", "").replace(" ", "_")
                        log_files.append((system_name, os.path.join(folder_path, file)))

        return xyz_files, log_files

    def safe_read_xyz(self, file_path: str) -> List[Atoms]:
        """
        Safely reads an XYZ file.

        Args:
            file_path (str): Path to the XYZ file.

        Returns:
            List[Atoms]: List of atomic structures.
        """
        try:
            # with open(file_path, "rb") as f:
            #     content = f.read()
            # decoded_content = content.decode("utf-8", errors="ignore")
            # temp_file_name = f"./temp_file_dir/temporary_{self.uuid}.xyz"
            # with open(temp_file_name, "w") as temp_file:
            #     temp_file.write(decoded_content)
            atoms = io.read(file_path, index=":")
            # os.remove(temp_file_name) 
            return atoms
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    @staticmethod
    def read_log_for_temperature(log_path: str) -> List[Tuple[float, float, float, float, float]]:
        """
        Reads a log file to extract temperature and energy data.

        Args:
            log_path (str): Path to the log file.

        Returns:
            List[Tuple]: Extracted data.
        """
        data = []
        try:
            with open(log_path, 'r') as log_file:
                for line in log_file:
                    match = re.match(r"(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)", line)
                    if match:
                        data.append(tuple(map(float, match.groups())))
        except Exception as e:
            print(f"Error reading log file {log_path}: {e}")
        return data

class PropertyCalculator:
    """Class for performing property calculations on atomic structures."""

    @staticmethod
    def calculate_density(atoms: Atoms) -> float:
        """
        Calculate the density of an atomic structure in g/cm³.
        
        Args:
            atoms (Atoms): The atomic structure to calculate the density for.
        
        Returns:
            float: The density in g/cm³.
        """
        # Conversion constants
        amu_to_grams = 1.66053906660e-24  # Atomic mass unit to grams
        angstrom_to_cm = 1e-8  # Angstrom to centimeters

        # Calculate mass in grams
        mass_amu = atoms.get_masses().sum()
        mass_g = mass_amu * amu_to_grams

        # Calculate volume in cm³
        volume_A3 = atoms.get_volume()
        volume_cm3 = volume_A3 * (angstrom_to_cm**3)

        # Return density
        return mass_g / volume_cm3
    
class XYZAnalyzer:
    """
    Class for analyzing and processing XYZ and log files related to atomic structures.
    Generates plots for various properties like density, temperature, and energy.

    Attributes:
        root_folder (str): The root folder containing the input files.
        output_dir (str): The directory to save output plots and analysis results.
        file_handler (FileHandler): An instance of the FileHandler class to read files.
        calculator (PropertyCalculator): An instance of the PropertyCalculator class to calculate properties.
    """

    def __init__(self, root_folder: str, output_dir: str):
        """
        Initializes the XYZAnalyzer with the root folder and output directory.
        
        Args:
            root_folder (str): Path to the root folder containing XYZ and log files.
            output_dir (str): Path to the directory where results and plots will be saved.
        """
        self.root_folder = root_folder
        self.output_dir = output_dir

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize FileHandler and PropertyCalculator
        self.file_handler = FileHandler(root_folder)
        self.calculator = PropertyCalculator()

    def process_file(self, system_name: str, xyz_file_path: str, log_file_path: str) -> Tuple[List[float], List[np.ndarray], List[Tuple[float, float, float, float, float]]]:
        """
        Process a single XYZ and log file to extract properties.
        
        Args:
            system_name (str): Name of the system being processed.
            xyz_file_path (str): Path to the XYZ file for the atomic structure.
            log_file_path (str): Path to the log file containing simulation data.
        
        Returns:
            Tuple:
                - List[float]: List of calculated densities.
                - List[np.ndarray]: List of lattice parameters for each structure.
                - List[Tuple[float, float, float, float, float]]: List of tuples containing time, temperature, total energy, potential energy, and kinetic energy.
        """
        densities = []
        lattice_params = []
        time_temp_data = []

        # Read the XYZ file and compute densities
        structures = self.file_handler.safe_read_xyz(xyz_file_path)
        # print(len(structures))

        for structure in structures:
            try:
                # Calculate density for each structure
                densities.append(self.calculator.calculate_density(structure))
                # Get lattice parameters
                lattice_params.append(structure.get_cell_lengths_and_angles())
            except Exception as e:
                print(f"Error processing structure in {system_name}: {e}")

        # Read the log file and extract time, temperature, and energy data
        time_temp_data = self.file_handler.read_log_for_temperature(log_file_path)

        return densities, lattice_params, time_temp_data

    def analyze_and_plot(self):
        """
        Analyze all XYZ and log files in the root folder and generate plots.
        
        - Generates plots for density evolution, temperature, and energies (Etot, Epot, Ekin) over time.
        - Skips systems where the temperature exceeds 3000K.
        """
        # Find XYZ and log files
        xyz_files, log_files = self.file_handler.find_xyz_files()
        print(f"Found {len(xyz_files)} XYZ files and {len(log_files)} log files.")

        if not xyz_files or not log_files:
            print("No XYZ or Log files found!")
            return

        # Create subplots for density, temperature, and energies
        fig, (ax_density, ax_temp, ax_etot, ax_epot, ax_ekin) = plt.subplots(5, 1, figsize=(12, 30))

        # Process each XYZ and log file pair
        for (system_name, xyz_file_path), (_, log_file_path) in zip(xyz_files, log_files):
            densities, _, time_temp_data = self.process_file(system_name, xyz_file_path, log_file_path)

            # Skip systems with temperature exceeding 3000K
            if any(temp_k > 3000 for _, temp_k, _, _, _ in time_temp_data):
                print(f"Skipping {system_name} due to temperature exceeding 3000K")
                continue

            # Extract time, temperature, and energy data
            time_steps, temperatures, etot_values, epot_values, ekin_values = zip(*time_temp_data)

            # Plot density evolution
            if densities:
                ax_density.plot(range(len(densities)), densities, label=system_name)

            # Plot temperature evolution
            ax_temp.plot(time_steps, temperatures, label=system_name)

            # Plot total energy (Etot) vs time
            ax_etot.plot(time_steps, etot_values, label=system_name)

            # Plot potential energy (Epot) vs time
            ax_epot.plot(time_steps, epot_values, label=system_name)

            # Plot kinetic energy (Ekin) vs time
            ax_ekin.plot(time_steps, ekin_values, label=system_name)

        # Set labels and titles for the subplots
        ax_density.set_xlabel("Timesteps")
        ax_density.set_ylabel("Density (g/cm³)")
        ax_density.set_title("Density Evolution")

        ax_temp.set_xlabel("Time (ps)")
        ax_temp.set_ylabel("Temperature (K)")
        ax_temp.set_title("Temperature Evolution")

        ax_etot.set_xlabel("Time (ps)")
        ax_etot.set_ylabel("Etot (eV)")
        ax_etot.set_title("Etot Evolution")

        ax_epot.set_xlabel("Time (ps)")
        ax_epot.set_ylabel("Epot (eV)")
        ax_epot.set_title("Epot Evolution")

        ax_ekin.set_xlabel("Time (ps)")
        ax_ekin.set_ylabel("Ekin (eV)")
        ax_ekin.set_title("Ekin Evolution")

        # Adjust layout and save the figure
        fig.tight_layout()
        fig.savefig(f"{self.output_dir}/energy_and_temperature.png", bbox_inches='tight')
        plt.show()
        
        
def symmetricize_replicate(curr_atoms, max_atoms, box_lengths):
    """
    Determine the replication factors needed to increase the number of atoms in the system
    to at least half of the target max_atoms while maintaining the symmetry of the cell.
    """
    replication = [1, 1, 1]  # Initial replication factors for each direction
    atom_count = curr_atoms
    lengths = box_lengths
    while atom_count < (max_atoms // 2):
        direction = np.argmin(box_lengths)  # Choose the smallest direction to replicate
        replication[direction] += 1  # Increase replication factor for that direction
        lengths[direction] = box_lengths[direction] * replication[direction]
        atom_count = curr_atoms * replication[0] * replication[1] * replication[2]
    return replication, atom_count

def replicate_system(atoms, replicate_factors):
    """
    Replicates the given ASE Atoms object according to the specified replication factors
    (nx, ny, nz).
    """
    nx, ny, nz = replicate_factors
    original_cell = atoms.get_cell()  # Original cell parameters
    original_positions = atoms.get_positions() #@ original_cell  # Convert positions to cartesian
    original_numbers = atoms.get_atomic_numbers()  # Atomic numbers of the atoms
    x_cell, y_cell, z_cell = original_cell[0], original_cell[1], original_cell[2]
    
    new_numbers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                new_numbers += [original_numbers]
                
    # Replicate positions along each direction
    pos_after_x = np.concatenate([original_positions + i * x_cell for i in range(nx)])
    pos_after_y = np.concatenate([pos_after_x + i * y_cell for i in range(ny)])
    pos_after_z = np.concatenate([pos_after_y + i * z_cell for i in range(nz)])
    
    # New cell dimensions after replication
    new_cell = [nx * original_cell[0], ny * original_cell[1], nz * original_cell[2]]
    
    # Create a new Atoms object with the replicated structure
    new_atoms = Atoms(
        numbers=np.concatenate(new_numbers),
        positions=pos_after_z,
        cell=new_cell,
        pbc=atoms.get_pbc(),
    )
    return new_atoms

def min_height(cell_matrix):
    """
    Calculate the perpendicular heights in three directions given a 3x3 cell matrix.
    The minimum height corresponds to the shortest height along the principal axes.
    """
    a, b, c = cell_matrix[:, 0], cell_matrix[:, 1], cell_matrix[:, 2]
    volume = abs(np.dot(a, np.cross(b, c)))
    
    # Calculate the cross products for each pair of axes
    a_cross_b, b_cross_c, c_cross_a = (
        np.linalg.norm(np.cross(a, b)),
        np.linalg.norm(np.cross(b, c)),
        np.linalg.norm(np.cross(c, a)),
    )
    
    # Calculate the perpendicular heights
    height_a, height_b, height_c = (
        abs(volume / a_cross_b),
        abs(volume / b_cross_c),
        abs(volume / c_cross_a),
    )
    return min(height_a, height_b, height_c)

def perturb_config(atoms, displacement_std=0.01):
    """
    Perturb the atomic positions of the input ASE Atoms object by a Gaussian random displacement
    to simulate thermal fluctuations.
    """
    positions = atoms.get_positions()
    displacements = np.random.normal(scale=displacement_std, size=positions.shape)  # Gaussian noise
    new_positions = positions + displacements  # Apply perturbations to positions
    
    # Create a copy of the atoms object with the perturbed positions
    new_perturbed_atoms = atoms.copy()
    new_perturbed_atoms.set_positions(new_positions)
    return new_perturbed_atoms

def get_pairs(atoms):
    """
    Generate all unique pairs of atomic types present in the system.
    """
    Atom_types = np.unique(atoms.get_chemical_symbols())  # List unique atomic species
    Pairs = []
    for i in range(len(Atom_types)):
        for j in range(i, len(Atom_types)):
            Pairs += [[Atom_types[i], Atom_types[j]]]
    return Pairs

def getfirstpeaklength(r, rdf, r_max=6.0):
    """
    Find the position of the first peak in the RDF (Radial Distribution Function) within
    a specified range (r_max).
    """
    bin_size = (r[-1] - r[0]) / len(r)
    cut_index = int(r_max / bin_size)
    cut_index = min(cut_index, len(r))
    Peak_index = np.argmax(rdf[:cut_index])  # Index of the first peak
    return Peak_index, r[Peak_index]  # Return peak index and corresponding bond length

def get_partial_rdfs(Traj, r_max=6.0, dr=0.01):
    """
    Compute the partial radial distribution functions (RDFs) for all pairs of elements in the trajectory.
    """
    rmax = min(r_max, min_height(Traj[0].get_cell()) / 2.7)
    analysis = Analysis(Traj)  # Initialize RDF analysis
    dr = dr
    nbins = int(rmax / dr)  # Number of bins in the RDF calculation
    pairs_list = get_pairs(Traj[0])  # List of element pairs to compute RDFs
    Pair_rdfs = dict()
    
    for pair in pairs_list:
        rdf = analysis.get_rdf(
            rmax=rmax, nbins=nbins, imageIdx=None, elements=pair, return_dists=True
        )
        x = rdf[0][1]
        y = np.array([rdf[k][0] for k in range(len(rdf))]).mean(axis=0)
        Pair_rdfs["-".join(pair)] = [x, y]
    return Pair_rdfs

def get_partial_rdfs_smoothened(
    inp_atoms, perturb=10, noise_std=0.01, max_atoms=300, r_max=6.0, dr=0.01
):
    """
    Compute smoothened partial RDFs by perturbing the input atoms and replicating the system to a target size.
    """
    atoms = inp_atoms.copy()
    replication_factors, _ = symmetricize_replicate(
        len(atoms),
        max_atoms=max_atoms,
        box_lengths=atoms.get_cell_lengths_and_angles()[:3],
    )
    atoms = replicate_system(atoms, replication_factors)
    
    # Perturb the system multiple times to smoothen the RDF
    Traj = [perturb_config(atoms, noise_std) for k in range(perturb)]
    return get_partial_rdfs(Traj, r_max=r_max, dr=dr)

def get_bond_lengths_noise(
    inp_atoms, perturb=10, noise_std=0.01, max_atoms=300, r_max=6.0, dr=0.01
):
    """
    Get the bond lengths by computing the first peak in the RDF after perturbing the system.
    """
    Pair_rdfs = get_partial_rdfs_smoothened(
        inp_atoms,
        perturb=perturb,
        noise_std=noise_std,
        max_atoms=max_atoms,
        r_max=r_max,
        dr=dr,
    )
    Bond_lengths = dict()
    for key in Pair_rdfs:
        r, rdf = Pair_rdfs[key]
        Bond_lengths[key] = getfirstpeaklength(r, rdf)[1]  # Get bond length at the first peak
    return Bond_lengths, Pair_rdfs

def get_bond_lengths_TrajAvg(Traj, r_max=6.0, dr=0.01):
    """
    Compute the bond lengths from the average RDF of a trajectory.
    """
    Pair_rdfs = get_partial_rdfs(Traj, r_max=r_max, dr=dr)
    Bond_lengths = dict()
    for key in Pair_rdfs:
        r, rdf = Pair_rdfs[key]
        Bond_lengths[key] = getfirstpeaklength(r, rdf)[1]
    return Bond_lengths, Pair_rdfs

def get_initial_rdf(
    inp_atoms,
    perturb=10,
    noise_std=0.01,
    max_atoms=300,
    replicate=False,
    Structid=0,
    r_max=6.0,
    dr=0.01,
):
    """
    Compute the initial RDF for a given input atoms object with optional perturbations and replication.
    """
    atoms = inp_atoms.copy()
    
    if replicate:
        replication_factors, size = symmetricize_replicate(
            len(atoms),
            max_atoms=max_atoms,
            box_lengths=atoms.get_cell_lengths_and_angles()[:3],
        )
        atoms = replicate_system(atoms, replication_factors)
    
    rmax = min(r_max, min_height(atoms.get_cell()) / 2.7)  # Ensure r_max doesn't exceed the cell height
    analysis = Analysis([perturb_config(atoms, noise_std) for k in range(perturb)])
    
    dr = dr
    nbins = int(rmax / dr)  # Number of bins for RDF calculation
    rdf = analysis.get_rdf(
        rmax=rmax, nbins=nbins, imageIdx=None, elements=None, return_dists=True
    )
    x = rdf[0][1]
    y = np.array([rdf[k][0] for k in range(len(rdf))]).mean(axis=0)
    return x, y

def get_rdf(Traj, r_max=6.0, dr=0.01):
    """
    Compute the radial distribution function (RDF) for a given trajectory.
    """
    rmax = min(r_max, min_height(Traj[0].get_cell()) / 2.7)
    analysis = Analysis(Traj)
    dr = dr
    nbins = int(rmax / dr)
    rdf = analysis.get_rdf(
        rmax=rmax, nbins=nbins, imageIdx=None, elements=None, return_dists=True
    )
    x = rdf[0][1]
    y = np.array([rdf[k][0] for k in range(len(rdf))]).mean(axis=0)
    return x, y


def process_file(file_handler, calculator, system_name: str, xyz_file_path: str, log_file_path: str):
    """Process a single XYZ and log file to extract properties."""
    densities = []
    lattice_params = []
    temperature = []
    rdf_error = []  # Will store RDF values for each window
    bond_error = dict()
    time_temp_data = []

    # Read the XYZ file and compute densities
    structures = file_handler.safe_read_xyz(xyz_file_path)
    if not structures:
        return [], [], [], [], {}, []

    # Calculate initial RDF
    _, initial_rdf = get_initial_rdf(
        structures[0], perturb=20, noise_std=0.05, max_atoms=200, replicate=True
    )

    # Calculate initial partial_rdf
    initial_bond_lengths, Initial_Pair_rdfs = get_bond_lengths_noise(
        structures[0], perturb=20, noise_std=0.05, max_atoms=200, r_max=6.0
    )

    counter = 0
    window_size = 100  # Size of the trajectory window for RDF calculation

    for structure in structures:
        try:
            densities.append(calculator.calculate_density(structure))
            lattice_params.append(structure.get_cell_lengths_and_angles())
            temperature.append(structure.get_temperature())

            if (counter + 1) % window_size == 0:
                window_start = max(0, counter - window_size + 1)
                window_structures = structures[window_start:counter + 1]

                r, current_rdf = get_rdf(window_structures, r_max=6.0)
                min_len = min(len(current_rdf), len(initial_rdf))
                current_rdf = current_rdf[:min_len]
                initial_rdf_ = initial_rdf[:min_len]
                error_rdf = (
                    100
                    * (((current_rdf - initial_rdf_) ** 2).sum())
                    / (((initial_rdf_) ** 2).sum())
                )
                rdf_error.append(error_rdf)

                curr_bond_lengths, Pair_rdfs = get_bond_lengths_TrajAvg(
                    window_structures, r_max=6.0
                )

                for key in curr_bond_lengths.keys():
                    if key in Initial_Pair_rdfs:
                        r, initial_rdf = Initial_Pair_rdfs[key]
                        r, rdf = Pair_rdfs[key]
                        RDF_len = min(len(rdf), len(initial_rdf))
                        r = r[:RDF_len]
                        rdf = rdf[:RDF_len]
                        initial_rdf_ = initial_rdf[:RDF_len]

                        pair_error = (
                            100
                            * (((rdf - initial_rdf_) ** 2).sum())
                            / (((initial_rdf_) ** 2).sum())
                        )

                        if key not in bond_error:
                            bond_error[key] = []
                        bond_error[key].append(pair_error)

        except Exception as e:
            print(f"Error processing structure {counter} in {system_name}: {e}")

        counter += 1

    time_temp_data = file_handler.read_log_for_temperature(log_file_path)

    return densities, lattice_params, temperature, rdf_error, time_temp_data, bond_error



def save_bond_errors_to_txt(file_name, bond_error):
    """Save bond errors to a text file."""
    with open(file_name, mode='w') as file:
        for bond, errors in bond_error.items():
            file.write(f"Bond: {bond}\n")
            file.write("Errors:\n")
            file.write(", ".join(map(str, errors)) + "\n")
            file.write("\n")


def save_to_csv(file_name, data):
    """Save data to a CSV file."""
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        if isinstance(data[0], (list, tuple)):
            writer.writerow([f"Column {i+1}" for i in range(len(data[0]))])
        writer.writerows(data)