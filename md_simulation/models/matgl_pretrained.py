from types import MethodType

import matgl
import torch
from matgl.ext.ase import Atoms2Graph


def load_pretrained_matgl(model_name):
    model_dict = {
        "chgnet_dgl": "CHGNet-MPtrj-2024.2.13-PES-11M",
        "m3gnet_dgl": "M3GNet-MP-2021.2.8-PES",
    }
    # matgl.float_th = torch.float64
    matgl_model = matgl.load_model(model_dict[model_name])

    def forward(self, atoms):
        graph_converter = Atoms2Graph(
            element_types=matgl_model.model.element_types,
            cutoff=matgl_model.model.cutoff,
        )
        graph, lattice, state_feats_default = graph_converter.get_graph(atoms)
        graph.edata["pbc_offshift"] = torch.matmul(
            graph.edata["pbc_offset"], lattice[0]
        )
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lattice[0]
        state_feats = torch.tensor(state_feats_default)
        total_energies, forces, stresses, *others = self.matgl_forward(
            graph,
            lattice,
            state_feats,
        )
        output = {}
        output["energy"] = total_energies
        output["forces"] = forces
        output["stress"] = stresses
        return output

    matgl_model.matgl_forward = matgl_model.forward
    matgl_model.forward = MethodType(forward, matgl_model)
    # matgl_model = matgl_model.to(torch.double)
    return matgl_model
