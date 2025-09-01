# import torch
# from e3nn.o3 import Irreps
# from mace.modules import ScaleShiftMACE
# from mace.modules.blocks import RealAgnosticResidualInteractionBlock
# from matsciml.common.types import BatchDict
# from matsciml.models.base import ScalarRegressionTask, ForceRegressionTask
# from matsciml.models.pyg.mace import MACEWrapper
# from torch import nn


# class MACEForwardModule(ForceRegressionTask):
#     def forward(self, batch):
#         outputs = self.encoder(batch)
#         breakpoint()
#         return outputs


# class OGMACE(MACEWrapper):
#     def _forward(
#         self,
#         graph,
#         node_feats: torch.Tensor,
#         pos: torch.Tensor,
#         **kwargs,
#     ):
#         mace_data = {
#             "positions": pos.to(torch.float64),
#             "node_attrs": node_feats.to(torch.float64),
#             "ptr": graph.ptr,
#             "cell": kwargs["cell"].to(torch.float64),
#             "shifts": kwargs["shifts"],
#             "batch": graph.batch,
#             "edge_index": graph.edge_index,
#             "unit_shifts": kwargs["unit_offsets"].to(torch.float64),
#         }
#         outputs = self.encoder(
#             mace_data,
#             training=self.training,
#             compute_force=True,
#             compute_virials=False,
#             compute_stress=True,
#             compute_displacement=False,
#         )
#         stress = outputs["stress"].squeeze(0)
#         six_stress = torch.tensor(
#             [
#                 stress[0, 0],
#                 stress[1, 1],
#                 stress[2, 2],
#                 stress[1, 2],
#                 stress[0, 2],
#                 stress[0, 1],
#             ]
#         )
#         outputs["stress"] = six_stress
#         return outputs

#     def forward(self, batch: BatchDict):
#         input_data = self.read_batch(batch)
#         input_data["unit_offsets"] = batch["unit_offsets"].to(torch.float64)
#         outputs = self._forward(**input_data)
#         return outputs


# def load_mace(checkpoint):
#     available_models = {
#         "mace": {
#             "encoder_class": OGMACE,
#             "encoder_kwargs": {
#                 "mace_module": ScaleShiftMACE,
#                 "num_atom_embedding": 89,
#                 "r_max": 6.0,
#                 "num_bessel": 10,
#                 "num_polynomial_cutoff": 5.0,
#                 "max_ell": 3,
#                 "interaction_cls": RealAgnosticResidualInteractionBlock,
#                 "interaction_cls_first": RealAgnosticResidualInteractionBlock,
#                 "num_interactions": 2,
#                 "atom_embedding_dim": 128,
#                 "MLP_irreps": Irreps("16x0e"),
#                 "avg_num_neighbors": 61.964672446250916,
#                 "correlation": 3,
#                 "radial_type": "bessel",
#                 "gate": nn.SiLU(),
#                 "atomic_inter_scale": 0.804153875447809696197509765625,
#                 "atomic_inter_shift": 0.164096963591873645782470703125,
#                 "distance_transform": None,
#                 ###
#                 # fmt: off
#                 "atomic_energies": torch.Tensor(
#                     [-3.6672, -1.3321, -3.4821, -4.7367, -7.7249, -8.4056, -7.3601, -7.2846, -4.8965, 0.0, -2.7594, -2.814, -4.8469, -7.6948, -6.9633, -4.6726, -2.8117, -0.0626, -2.6176, -5.3905, -7.8858, -10.2684, -8.6651, -9.2331, -8.305, -7.049, -5.5774, -5.1727, -3.2521, -1.2902, -3.5271, -4.7085, -3.9765, -3.8862, -2.5185, 6.7669, -2.5635, -4.938, -10.1498, -11.8469, -12.1389, -8.7917, -8.7869, -7.7809, -6.85, -4.891, -2.0634, -0.6396, -2.7887, -3.8186, -3.5871, -2.8804, -1.6356, 9.8467, -2.7653, -4.991, -8.9337, -8.7356, -8.019, -8.2515, -7.5917, -8.1697, -13.5927, -18.5175, -7.6474, -8.123, -7.6078, -6.8503, -7.8269, -3.5848, -7.4554, -12.7963, -14.1081, -9.3549, -11.3875, -9.6219, -7.3244, -5.3047, -2.3801, 0.2495, -2.324, -3.73, -3.4388, -5.0629, -11.0246, -12.2656, -13.8556, -14.9331, -15.2828]
#                 ).to(torch.double),
#                 "atomic_numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 94],
#                 # fmt: on
#             },
#             "output_kwargs": {"lazy": False, "input_dim": 256, "hidden_dim": 256},
#         }
#     }

#     if "2024-01-07-mace-128-L2_epoch-199.model" in checkpoint:
#         available_models["mace"]["encoder_kwargs"]["hidden_irreps"] = Irreps(
#             "128x0e+128x1o+128x2e"
#         )

#     model = MACEForwardModule(**available_models["mace"])
#     model.encoder.encoder.load_state_dict(
#         torch.load(checkpoint, map_location=torch.device("cpu")).state_dict(),
#         strict=True,
#     )
#     model = model.to(torch.double)
#     return model


from types import MethodType
from warnings import warn

import torch
from mace.calculators import mace_mp
#"small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",

"""
For reference :
https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py

urls = {
    "small":"https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "large": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model",
}"""


def load_pretrained_mace(model_name):
    warn(
        "Defaulting to small model for now. See https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py for options."
    )
    mace_calc = mace_mp(model="medium-omat-0", default_dtype="float64", device="cpu")
    mace_model = mace_calc.models[0]

    mace_kwargs = {"compute_force": True, "compute_stress": True}

    def forward(self, atoms):
        batch = mace_calc._atoms_to_batch(atoms)
        output = self.mace_forward(batch, **mace_kwargs)
        s = output["stress"].squeeze(0)
        output["stress"] = torch.tensor(
            [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
        )
        return output

    mace_model.mace_forward = mace_model.forward
    mace_model.forward = MethodType(forward, mace_model)
    return mace_model
