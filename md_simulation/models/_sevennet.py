from types import MethodType

import numpy as np
import pytest
import sevenn._keys as KEY
import torch
from ase.build import bulk, molecule
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import model_from_checkpoint, pretrained_name_to_path

# https://github.com/MDIL-SNU/SevenNet?tab=readme-ov-file#sevennet-l3i5-12dec2024


def load_pretrained_sevennet():
    cp_path = pretrained_name_to_path("7net-0_11July2024")
    sevennet_model, config = model_from_checkpoint(cp_path)
    cutoff = config["cutoff"]

    sevennet_model.original_forward = sevennet_model.forward
    sevennet_model.set_is_batch_data(False)

    def forward(self, atoms):
        data = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms, cutoff))
        sevennet_model.forward = sevennet_model.original_forward
        output = sevennet_model(data)
        sevennet_model.forward = MethodType(forward, sevennet_model)
        energy = output[KEY.PRED_TOTAL_ENERGY]
        results = {
            "free_energy": energy,
            "energy": energy,
            "energies": (output[KEY.ATOMIC_ENERGY].reshape(len(atoms))),
            "forces": output[KEY.PRED_FORCE],
            "stress": (-output[KEY.PRED_STRESS])[
                [0, 1, 2, 4, 5, 3]
            ],  # as voigt notation),
        }
        return results

    sevennet_model.forward = MethodType(forward, sevennet_model)

    return sevennet_model
