from types import MethodType

import ase
from ase.build import bulk
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs

# https://github.com/orbital-materials/orb-models/blob/main/orb_models/forcefield/pretrained.py


def load_pretrained_orb():
    orb_model = pretrained.orb_v2(device="cpu")

    orb_model.original_forward = orb_model.forward

    def forward(self, atoms):
        graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device="cpu")
        orb_model.forward = orb_model.original_forward
        output = orb_model.predict(graph)
        orb_model.forward = MethodType(forward, orb_model)
        results = {
            "energy": output["graph_pred"],
            "forces": output["node_pred"],
            "stress": output["stress_pred"].squeeze(),
        }
        return results

    orb_model.forward = MethodType(forward, orb_model)
    return orb_model


# def load_pretrained_orb():
#     orb_model = pretrained.orb_v3_direct-20-omat(device="cpu")

#     orb_model.original_forward = orb_model.forward

#     def forward(self, atoms):
#         graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device="cpu")
#         orb_model.forward = orb_model.original_forward
#         output = orb_model.predict(graph)
#         orb_model.forward = MethodType(forward, orb_model)
#         results = {
#             "energy": output["graph_pred"],
#             "forces": output["node_pred"],
#             "stress": output["stress_pred"].squeeze(),
#         }
#         return results

#     orb_model.forward = MethodType(forward, orb_model)
#     return orb_model








# device = "cpu"  # or device="cuda"
# orbff = pretrained.orb_v2(device=device)
# atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
# graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device=device)

# # Optionally, batch graphs for faster inference
# # graph = batch_graphs([graph, graph, ...])

# result = orbff.predict(graph)

# # Convert to ASE atoms (unbatches the results and transfers to cpu if necessary)
# atoms = atomic_system.atom_graphs_to_ase_atoms(
#     graph,
#     energy=result["graph_pred"],
#     forces=result["node_pred"],
#     stress=result["stress_pred"]
# )
