import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.geometry.analysis import Analysis
from ase.optimize import FIRE
from ase.units import GPa      ## 1 GPa = 1 / 160.21766208 eV/Å³.
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pandas as pd


def minimize_structure(atoms, fmax=0.05, steps=50):
    """
    Perform energy minimization on the given ASE Atoms object using the FIRE optimizer.

    Parameters:
    atoms (ase.Atoms): The Atoms object to be minimized.
    fmax (float): The maximum force tolerance for the optimization (default: 0.01 eV/Å).
    steps (int): The maximum number of optimization steps (default: 1000).

    Returns:
    ase.Atoms: The minimized Atoms object.
    """
    dyn = FIRE(atoms, trajectory=None)
    dyn.run(fmax=fmax, steps=steps)
    return atoms




def min_height(cell_matrix):
    """
    Calculate the perpendicular heights in three directions given a 3x3 cell matrix.
    """
    a, b, c = cell_matrix[:, 0], cell_matrix[:, 1], cell_matrix[:, 2]
    volume = abs(np.dot(a, np.cross(b, c)))
    # Calculate the cross products
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
    # Create a new Atoms object with the perturbed positions
    positions = atoms.get_positions()
    displacements = np.random.normal(scale=displacement_std, size=positions.shape)
    new_positions = positions + displacements
    new_perturbed_atoms = atoms.copy()
    new_perturbed_atoms.set_positions(new_positions)
    return new_perturbed_atoms


def plot_pair_rdfs(Pair_rdfs, shift=0):
    counter = 0
    plt.figure()
    for key in Pair_rdfs.keys():
        plt.plot(Pair_rdfs[key][0], Pair_rdfs[key][1] + shift * counter, label=key)
        counter += 1
    plt.legend(loc=(1.2, 0))
    plt.xlabel("r (Angstrom)")
    plt.ylabel("g(r)")
    plt.show()


def replicate_system(atoms: Atoms, replicate_factors: np.ndarray) -> Atoms:
    """
    Replicates the given ASE Atoms object according to the specified replication factors.
    """
    nx, ny, nz = replicate_factors
    original_cell = atoms.get_cell()
    original_positions = (
        atoms.get_scaled_positions() @ original_cell
    )  # Scaled or Unscaled ?
    original_numbers = atoms.get_atomic_numbers()
    x_cell, y_cell, z_cell = original_cell[0], original_cell[1], original_cell[2]
    new_numbers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                new_numbers += [original_numbers]
    pos_after_x = np.concatenate([original_positions + i * x_cell for i in range(nx)])
    pos_after_y = np.concatenate([pos_after_x + i * y_cell for i in range(ny)])
    pos_after_z = np.concatenate([pos_after_y + i * z_cell for i in range(nz)])
    new_cell = [nx * original_cell[0], ny * original_cell[1], nz * original_cell[2]]
    new_atoms = Atoms(
        numbers=np.concatenate(new_numbers),
        positions=pos_after_z,
        cell=new_cell,
        pbc=atoms.get_pbc(),
    )
    new_atoms.calc = atoms.calc
    return new_atoms


def write_xyz(Filepath, atoms):
    """Writes ovito xyz file"""
    R = atoms.get_position()
    species = atoms.get_atomic_numbers()
    cell = atoms.get_cell()
    f = open(Filepath, "w")
    f.write(str(R.shape[0]) + "\n")
    flat_cell = cell.flatten()
    f.write(
        f'Lattice="{flat_cell[0]} {flat_cell[1]} {flat_cell[2]} {flat_cell[3]} {flat_cell[4]} {flat_cell[5]} {flat_cell[6]} {flat_cell[7]} {flat_cell[8]}" Properties=species:S:1:pos:R:3 Time=0.0'
    )
    for i in range(R.shape[0]):
        f.write(
            "\n"
            + str(species[i])
            + "\t"
            + str(R[i, 0])
            + "\t"
            + str(R[i, 1])
            + "\t"
            + str(R[i, 2])
        )


def symmetricize_replicate(curr_atoms: int, max_atoms: int, box_lengths: np.ndarray):
    replication = [1, 1, 1]
    atom_count = curr_atoms
    lengths = box_lengths
    while atom_count < (max_atoms // 2):
        direction = np.argmin(box_lengths)
        replication[direction] += 1
        lengths[direction] = box_lengths[direction] * replication[direction]
        atom_count = curr_atoms * replication[0] * replication[1] * replication[2]
    return replication, atom_count


def get_pairs(atoms):
    Atom_types = np.unique(atoms.get_chemical_symbols())
    Pairs = []
    for i in range(len(Atom_types)):
        for j in range(i, len(Atom_types)):
            Pairs += [[Atom_types[i], Atom_types[j]]]
    return Pairs


def getfirstpeaklength(r, rdf, r_max=6.0):
    bin_size = (r[-1] - r[0]) / len(r)
    cut_index = int(r_max / bin_size)
    cut_index = min(cut_index, len(r))
    Peak_index = np.argmax(rdf[:cut_index])
    # Returns : Peak index and Bond length
    return Peak_index, r[Peak_index]


def get_partial_rdfs(Traj, r_max=6.0, dr=0.01):
    rmax = min(r_max, min_height(Traj[0].get_cell()) / 2.7)
    analysis = Analysis(Traj)
    dr = dr
    nbins = int(rmax / dr)
    pairs_list = get_pairs(Traj[0])
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
    atoms = inp_atoms.copy()
    replication_factors, _ = symmetricize_replicate(
        len(atoms),
        max_atoms=max_atoms,
        box_lengths=atoms.get_cell_lengths_and_angles()[:3],
    )
    atoms = replicate_system(atoms, replication_factors)
    Traj = [perturb_config(atoms, noise_std) for k in range(perturb)]
    return get_partial_rdfs(Traj, r_max=r_max, dr=dr)


def get_bond_lengths_noise(
    inp_atoms, perturb=10, noise_std=0.01, max_atoms=300, r_max=6.0, dr=0.01
):
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
        Bond_lengths[key] = getfirstpeaklength(r, rdf)[1]
    return Bond_lengths, Pair_rdfs


def get_bond_lengths_TrajAvg(Traj, r_max=6.0, dr=0.01):
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
    atoms = inp_atoms.copy()
    # write_xyz(f"StabilityXYZData2/{Structid}.xyz",atoms.get_positions(),atoms.get_chemical_symbols(),atoms.get_cell())
    if replicate:
        replication_factors, size = symmetricize_replicate(
            len(atoms),
            max_atoms=max_atoms,
            box_lengths=atoms.get_cell_lengths_and_angles()[:3],
        )
        atoms = replicate_system(atoms, replication_factors)
    rmax = min(r_max, min_height(atoms.get_cell()) / 2.7)
    # atoms.rattle(0.01)
    analysis = Analysis([perturb_config(atoms, noise_std) for k in range(perturb)])
    # write_xyz(f"StabilityXYZDataReplicated2/{Structid}.xyz",atoms.get_positions(),atoms.get_chemical_symbols(),atoms.get_cell())
    dr = dr
    nbins = int(rmax / dr)
    rdf = analysis.get_rdf(
        rmax=rmax, nbins=nbins, imageIdx=None, elements=None, return_dists=True
    )
    x = rdf[0][1]
    y = np.array([rdf[k][0] for k in range(len(rdf))]).mean(axis=0)
    return x, y


def get_rdf(Traj, r_max=6.0, dr=0.01):
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


def get_density(atoms: Atoms) -> float:
    amu_to_grams = 1.66053906660e-24  # 1 amu = 1.66053906660e-24 grams
    angstrom_to_cm = 1e-8  # 1 Å = 1e-8 cm
    mass_amu = atoms.get_masses().sum()
    mass_g = (
        mass_amu * amu_to_grams
    )  # Get the volume of the atoms object in cubic angstroms (Å³)
    volume_A3 = atoms.get_volume()
    volume_cm3 = volume_A3 * (angstrom_to_cm**3)  # 1 Å³ = 1e-24 cm³
    density = mass_g / volume_cm3

    return density


def elastic_tensor_calculation(atoms, calculator,filename):
  atoms.calc = calculator

  # Minimize the structure before stress calculations
  dyn = FIRE(atoms)
  dyn.run(fmax=0.01, steps=1000)  # Converge forces below 0.01 eV/Å

  # Define small strain range
  eps = 1e-4  # Maximum strain
  n_points = 20  # Number of points from -eps to +eps
  strain_values = np.linspace(-eps, eps, n_points)

  Cij = np.zeros((6, 6))  # Elastic tensor storage

  # Define strain matrices for Voigt notation (6 independent strains)
  strain_matrices = [
      [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # e_xx
      [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # e_yy
      [[0, 0, 0], [0, 0, 0], [0, 0, 1]],  # e_zz
      [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]],  # e_yz
      [[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]],  # e_xz
      [[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]],  # e_xy
  ]

  # Labels for the strain components in Voigt notation
  voigt_labels = ['11', '22', '33', '23', '13', '12']

  # Compute reference stress
  ref_stress = atoms.get_stress(voigt=True)
  elastic_data=[]

  for i, strain_matrix in enumerate(strain_matrices):
      stresses = np.zeros((n_points, 6))
      intercepts=np.zeros((1,6))
      r_values=np.zeros((1,6))
      std_errs=np.zeros((1,6))

      for j, strain in enumerate(strain_values):
          strained_atoms = atoms.copy()
          deformation_matrix = np.eye(3) + strain * np.array(strain_matrix)
          strained_atoms.set_cell(atoms.cell @ deformation_matrix, scale_atoms=True)

          strained_atoms.calc = calculator
          dyn = FIRE(strained_atoms)
          dyn.run(fmax=0.05, steps=1000)  # Minimize structure
          stresses[j, :] = strained_atoms.get_stress(voigt=True) - ref_stress

          elastic_data.append([strain] + list(stresses[j, :]))

      # Perform linear regression to find the best slope
      for k in range(6):
          slope, intercept, r_value, p_value, std_err = linregress(strain_values, stresses[:, k])

          Cij[i, k] = slope
          intercepts[:,k]=intercept
          r_values[:,k]=r_value
          std_errs[:,k]=std_err


  # Convert the elastic tensor to GPa
  Cij_GPa = Cij / GPa # Convert the elastic tensor to GPa (Cij_GPa)

#   print("Elastic Stiffness Tensor (Cij) in GPa:")
#   print(np.array2string(Cij_GPa, precision=2, suppress_small=True,
#                          formatter={'float_kind': lambda x: f"{x:6.2f}"}))

# Save data as CSV using pandas
  columns = ["Strain"] + [f"Stress_{label}" for label in voigt_labels]
  df = pd.DataFrame(elastic_data, columns=columns)
  df.to_csv(filename, index=False)

  return Cij_GPa