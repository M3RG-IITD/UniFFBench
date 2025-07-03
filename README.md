<div align="center">
<p align="center">
  <img src="uniff.png" alt="My Project Logo" width="150">
</p>



[![PyPI version](https://badge.fury.io/py/mlipx.svg)](https://badge.fury.io/py/mlipx)
[![ZnTrack](https://img.shields.io/badge/Powered%20by-ZnTrack-%23007CB0)](https://zntrack.readthedocs.io/en/latest/)
[![ZnDraw](https://img.shields.io/badge/works_with-ZnDraw-orange)](https://github.com/zincware/zndraw)
[![open issues](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/basf/mlipx/issues)



[🛠️Installation](https://mlipx.readthedocs.io/en/latest/installation.html) |
[📜Recipes](https://mlipx.readthedocs.io/en/latest/recipes.html) |
[🚀Quickstart](https://mlipx.readthedocs.io/en/latest/quickstart.html)

</div>

<div style="text-align: center;">
    <h1>UniFF-MD: Universal Machine Learning Force Fields Molecular Dynamics</h1>
</div>

We present a comprehensive evaluation of six state-of-the-art UMLFFs (CHGNet, M3GNet,
MACE, MatterSim, SevenNet, Orb) on a carefully curated dataset, namely
`MinX-1.5K`, comprising ∼1,500 minerals with experimentally obtained crystal
structures and elastic properties. Our analysis is divided into three parts: 
-  A systematic comparison of model prediction across the minerals dataset.
-  A quantitative assessment of temporal evolution during MD simulations.
-  Evaluation of elastic constants prediction to study their efficacy of modelling mechanical properties

## Installation

- `Docker`: We provide a Dockerfile inside the `docker` that can be run to install a container using standard docker commands.
- `mamba`: We have included a `mamba` specification that provides a complete out-of-the-box installation. Run `mamba env create -n matsciml --file conda.yml`, and will install all dependencies and `matsciml` as an editable install.
- `pip`: In this case, we assume you are bringing your own virtual environment. Depending on what hardware platform you have, you can copy-paste the following commands; because the absolute mess that is modern Python packaging, these commands include the URLs for binary distributions of PyG and DGL graph backends.

For CPU only (good for local laptop development):

```console
pip install -f https://data.pyg.org/whl/torch-2.4.0+cpu.html -f https://data.dgl.ai/wheels/torch-2.4/repo.html -e './[all]'
```

For XPU usage, you will need to install PyTorch separately first, followed by `matsciml`; note that the PyTorch version is lower
as 2.3.1 is the latest XPU binary distributed.

```console
pip install torch==2.3.1+cxx11.abi torchvision==0.18.1+cxx11.abi torchaudio==2.3.1+cxx11.abi intel-extension-for-pytorch==2.3.110+xpu oneccl_bind_pt==2.3.100+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install -f https://data.pyg.org/whl/torch-2.3.0+cpu.html -f https://data.dgl.ai/wheels/torch-2.3/repo.html -e './[all]'
```

For CUDA usage, substitute the index links with your particular toolkit version (e.g. 12.1 below):

```console
pip install -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html -f https://data.pyg.org/whl/torch-2.4.0+cu121.html -e './[all]'
```

Additionally, for a development install, one can specify the extra packages like `black` and `pytest` with `pip install './[dev]'`. These can be
added to the commit workflow by running `pre-commit install` to generate `git` hooks.

## Quickstart

## Training folder structure

`configs` contains the YAML configuration files used by the `matsciml` experiment parser.
At a high level, a single experiment YAML defines the scope of the experiment (e.g. task,
model, and dataset) imperatively. The experiment is then composed by passing definitions
for each component, i.e. a path to the LiPS dataset YAML file. An example call looks
like this:

```console
python matsciml/experiments/training_script.py \
	-e configs/experiments/faenet_lips_force.yaml \
	-m configs/models/faenet_pyg.yaml \
	-d configs/datasets/lips.yaml \
	-t configs/trainer.yaml
```

### Adding an experiment

1. Copy one of the experiment YAML configs; no hard and fast rule for naming scheme,
but to start off we have `<model>_<dataset>_<task>.yml` just for the ease of access.
2. Modify the keys in the experiment YAML config - the keys must match what are
defined in the other configs (e.g. `lips` refers to the name of the YAML file)
3. Update `trainer.yaml` as needed: _in particular, set the `wandb` entity to yours_!
4. Update the dataset YAML file as needed: pay attention to batch size, and paths.

### Common tweaking parameters

- Batch size (per DDP worker) is modified in the dataset YAML.
- Number of workers, epochs, callbacks are configured in `trainer.yaml`
- Learning rate is configured in the model YAML.

### Energy-Volume Curve

Compute an energy-volume curve using the `mp-1143` structure from the Materials Project and MLIPs such as `mace-mpa-0`, `sevennet`, and `orb-v2`:

```bash
mlipx recipes ev --models mace-mpa-0,sevennet,orb-v2 --material-ids=mp-1143 --repro
mlipx compare --glob "*EnergyVolumeCurve"
```

> [!NOTE]
> `mlipx` utilizes [ASE](https://wiki.fysik.dtu.dk/ase/index.html),
> meaning any ASE-compatible calculator for your MLIP can be used.
> If we do not provide a preset for your model, you can either adapt the `models.py` file, raise an [issue](https://github.com/basf/mlipx/issues/new) to request support, or submit a pull request to add your model directly.

Below is an example of the resulting comparison:

![ZnDraw UI](https://github.com/user-attachments/assets/2036e6d9-3342-4542-9ddb-bbc777d2b093#gh-dark-mode-only "ZnDraw UI")
![ZnDraw UI](https://github.com/user-attachments/assets/c2479d17-c443-4550-a641-c513ede3be02#gh-light-mode-only "ZnDraw UI")

> [!NOTE]
> Set your default visualizer path using: `export ZNDRAW_URL=http://localhost:1234`.

### Structure Optimization

Compare the performance of different models in optimizing multiple molecular structures from `SMILES` representations:

```bash
mlipx recipes relax --models mace-mpa-0,sevennet,orb-v2 --smiles "CCO,C1=CC2=C(C=C1O)C(=CN2)CCN" --repro
mlipx compare --glob "*0_StructureOptimization"
mlipx compare --glob "*1_StructureOptimization"
```

![ZnDraw UI](https://github.com/user-attachments/assets/7e26a502-3c59-4498-9b98-af8e17a227ce#gh-dark-mode-only "ZnDraw UI")
![ZnDraw UI](https://github.com/user-attachments/assets/a68ac9f5-e3fe-438d-ad4e-88b60499b79e#gh-light-mode-only "ZnDraw UI")

### Nudged Elastic Band (NEB)

Run and compare nudged elastic band (NEB) calculations for a given start and end structure:

```bash
mlipx recipes neb --models mace-mpa-0,sevennet,orb-v2 --datapath ../data/neb_end_p.xyz --repro
mlipx compare --glob "*NEBs"
```

![ZnDraw UI](https://github.com/user-attachments/assets/a2e80caf-dd86-4f14-9101-6d52610b9c34#gh-dark-mode-only "ZnDraw UI")
![ZnDraw UI](https://github.com/user-attachments/assets/0c1eb681-a32c-41c2-a15e-2348104239dc#gh-light-mode-only "ZnDraw UI")

## Citations

If you use Open MatSci ML Toolkit in your technical work or publication, we would appreciate it if you cite the Open MatSci ML Toolkit paper in TMLR:

<details>

<summary>
Miret, S.; Lee, K. L. K.; Gonzales, C.; Nassar, M.; Spellings, M. The Open MatSci ML Toolkit: A Flexible Framework for Machine Learning in Materials Science. Transactions on Machine Learning Research, 2023.
</summary>

```bibtex
@article{openmatscimltoolkit,
  title = {The Open {{MatSci ML}} Toolkit: {{A}} Flexible Framework for Machine Learning in Materials Science},
  author = {Miret, Santiago and Lee, Kin Long Kelvin and Gonzales, Carmelo and Nassar, Marcel and Spellings, Matthew},
  year = {2023},
  journal = {Transactions on Machine Learning Research},
  issn = {2835-8856}
}
```

</details>



<details>

<summary>
Lee, K. L. K., Gonzales, C., Nassar, M., Spellings, M., Galkin, M., & Miret, S. (2023). MatSciML: A Broad, Multi-Task Benchmark for Solid-State Materials Modeling. arXiv preprint arXiv:2309.05934.
</summary>

```bibtex
@article{lee2023matsciml,
  title={MatSciML: A Broad, Multi-Task Benchmark for Solid-State Materials Modeling},
  author={Lee, Kin Long Kelvin and Gonzales, Carmelo and Nassar, Marcel and Spellings, Matthew and Galkin, Mikhail and Miret, Santiago},
  journal={arXiv preprint arXiv:2309.05934},
  year={2023}
}
```

</details>
