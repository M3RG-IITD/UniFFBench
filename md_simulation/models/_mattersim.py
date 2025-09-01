import warnings
from types import MethodType

import torch
from ase.units import GPa
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.potential import Potential

warnings.filterwarnings("ignore")

# https://github.com/microsoft/mattersim/tree/81e6b01c09945f1c707eb2c29921932a12f920d1/pretrained_models


def load_pretrained_mattersim():
    mattersim_model = Potential.from_checkpoint(
        load_path="mattersim-v1.0.0-1m", device="cpu"
    )

    mattersim_model.original_forward = mattersim_model.forward

    def forward(self, atoms):
        dataloader = build_dataloader([atoms], only_inference=True)
        mattersim_model.forward = mattersim_model.original_forward
        predictions = mattersim_model.mattersim_forward(
            dataloader, include_forces=True, include_stresses=True
        )
        mattersim_model.forward = MethodType(forward, mattersim_model)
        s = predictions[2][0] * GPa  # eV/A^3
        stress = torch.tensor([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        results = {
            "energy": torch.tensor(predictions[0][0]),
            "forces": torch.tensor(predictions[1][0]),
            "stress": stress,
        }
        return results

    mattersim_model.mattersim_forward = mattersim_model.predict_properties
    mattersim_model.forward = MethodType(forward, mattersim_model)
    return mattersim_model


"""
from ase.io import read 

atoms = read("/store/nosnap/mlip-eval/uip-data/amcsd_processed_final/all/Abellaite/0_Abellaite_298.00_1.01_.cif")

structures = [atoms]
device = "cpu"
potential = Potential.from_checkpoint(device=device)
dataloader = build_dataloader(structures, only_inference=True)
for _ in range(5):
    predictions = potential.predict_properties(dataloader, include_forces=True, include_stresses=True)
"""


# # set up the structure
# si = bulk("Si", "diamond", a=5.43)

# # replicate the structures to form a list
# structures = [si] * 10

# # load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Running MatterSim on {device}")
# potential = Potential.from_checkpoint(device=device)

# # build the dataloader that is compatible with MatterSim
# dataloader = build_dataloader(structures, only_inference=True)

# # make predictions
# predictions = potential.predict_properties(dataloader, include_forces=True, include_stresses=True)

# # print the predictions
# print(f"Total energy in eV: {predictions[0]}")
# print(f"Forces in eV/Angstrom: {predictions[1]}")
# print(f"Stresses in GPa: {predictions[2]}")
# print(f"Stresses in eV/A^3: {np.array(predictions[2])*GPa}")
