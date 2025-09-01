from types import MethodType

from ase.build import add_adsorbate, fcc100, molecule
from ase.optimize import LBFGS
from fairchem.core import OCPCalculator
from fairchem.core.datasets import data_list_collater


def load_pretrained_equiformerv2():
    equiformer_calc = OCPCalculator(
        model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
        local_cache="pretrained_models",
        cpu=True,
    )

    equiformerv2_model = equiformer_calc.trainer
    # equiformerv2_model.original_forward = equiformerv2_model._forward

    def forward(self, atoms):
        data_object = equiformer_calc.a2g.convert(atoms)
        batch = data_list_collater([data_object], otf_graph=True)
        # equiformerv2_model.forward = equiformerv2_model.original_forward
        output = equiformerv2_model.predict(batch, per_image=False, disable_tqdm=True)
        # equiformerv2_model.forward = MethodType(forward, equiformerv2_model)
        results = {
            "energy": output["energy"],
            "forces": output["forces"],
            "stress": output["stress"],
        }
        return results

    equiformerv2_model.forward = MethodType(forward, equiformerv2_model)
    return equiformerv2_model
