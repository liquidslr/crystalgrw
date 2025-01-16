import time
import argparse
import torch
import numpy as np
import os

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from types import SimpleNamespace
from torch_geometric.data import Batch

from ..common.eval_utils import load_model, load_control, load_classifier
from ..common.model_utils import get_model

from ..common.data_utils import (
    lattice_params_to_matrix_torch,
    lattice_params_from_matrix,
)

from ..gnn.embeddings import MAX_ATOMIC_NUM
from ..common.stats import MP20_NATOM_DIST
from ..common.data_utils import get_ase_atoms, get_ase_traj_atoms


def sample_properties(model, natoms, batch=None, corrupt_coords=True,
                      corrupt_lattices=True, corrupt_types=True,
                      force_atom_types=False):
    if corrupt_coords:
        frac_coords = torch.rand((natoms.sum(0), 3)).to(model.device)
    else:
        frac_coords = batch.frac_coords.to(model.device)

    if corrupt_lattices:
        b_m = model.sde_fn.b_m["lattices"]
        sigma = np.sqrt((b_m[1] + b_m[0]) / 2)
        lattices = sigma * torch.randn((natoms.size(0), 3, 3)).to(model.device)
    else:
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles).to(model.device)

    if corrupt_types and (not force_atom_types):
        atom_types = torch.randint(1, MAX_ATOMIC_NUM + 1, (natoms.sum(0),)).to(model.device)
    else:
        atom_types = batch.atom_types.to(model.device)

    return frac_coords, lattices, atom_types


def sample_natoms(model, sample_num_atoms, batch_size):
    if isinstance(sample_num_atoms, dict):
        natoms = torch.randint(**sample_num_atoms,
                               size=(batch_size,)).to(model.device)
    elif isinstance(sample_num_atoms, int):
        natoms = torch.tensor([sample_num_atoms] * batch_size).to(model.device)
    elif sample_num_atoms == "mp20_stat":
        p = torch.tensor(MP20_NATOM_DIST).float()
        p = p / p.sum(-1)
        natoms = torch.multinomial(p, num_samples=batch_size,
                                   replacement=True).to(model.device)
    else:
        raise NotImplementedError(f"{sample_num_atoms} is not implemented.")

    return natoms


def store_samples(samples, stored_data, save_traj=False, down_sample_traj_step=1,
                  scatck="store"):
    if len(stored_data) == 0:
        stored_data = {k: [] for k in ["frac_coords", "num_atoms", "atom_types",
                                       "lengths", "angles", "traj_frac_coords",
                                       "traj_atom_types", "traj_lattices"]}

    for k in ["frac_coords", "num_atoms", "atom_types", "lengths", "angles"]:
        if scatck == "store":
            stored_data[k].append(samples[k].detach().cpu())
        elif scatck == "stack":
            stored_data[k].append(torch.stack(samples[k], dim=0))
        elif scatck == "cat":
            stored_data[k] = torch.cat(samples[k], dim=1)

    if save_traj:
        for k in ["traj_frac_coords", "traj_atom_types", "traj_lattices"]:
            if scatck == "store":
                stored_data[k].append(samples[k][:, ::down_sample_traj_step].detach().cpu())
            elif scatck == "stack":
                stored_data[k].append(torch.stack(samples[k], dim=0))
            elif scatck == "cat":
                stored_data[k] = torch.cat(samples[k], dim=1)

    return stored_data


def save_output(samples, model_path, out_name, args, load_data, start_time):
    output = {
        "eval_setting": args,
        "frac_coords": samples["frac_coords"],
        "num_atoms": samples["num_atoms"],
        "atom_types": samples["atom_types"],
        "lengths": samples["lengths"],
        "angles": samples["angles"],
        "time": time.time() - start_time
    }

    if args.save_traj:
        output.update({
            "traj_frac_coords": samples["traj_frac_coords"],
            "traj_atom_types": samples["traj_atom_types"],
            "traj_lattices": samples["traj_lattices"],
        })

    if load_data:
        output.update({"input_data_batch": samples["input_data_batch"]})
    torch.save(output, model_path / out_name)

    if args.save_xyz:
        from ase.io import write

        if args.save_traj:
            structures = get_ase_traj_atoms(samples)
        else:
            structures = get_ase_atoms(samples)

        save_path = os.path.join(args.model_path, out_name.split(".")[0])
        os.makedirs(save_path, exist_ok=True)
        write(save_path+".xyz", structures)


def reconstruction(loader, model, ld_kwargs, num_evals, sample_num_atoms,
                   force_num_atoms=False, force_atom_types=False, batch_size=None,
                   down_sample_traj_step=1, labels=None, guidance_strength=1):
    """
    Reconstruct crystal structures given some crystal information.
    """
    input_data_list = []
    stacked_data = {}

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f"batch {idx} in {len(loader)}")

        stored_data = {}

        for eval_idx in range(num_evals):
            if force_num_atoms or force_atom_types:
                natoms = batch.num_atoms
            else:
                natoms = sample_natoms(model, sample_num_atoms, batch_size)

            frac_coords, lattices, atom_types = sample_properties(
                model, natoms, batch, ld_kwargs.cfg.model.corrupt_coords,
                ld_kwargs.cfg.model.corrupt_lattices, ld_kwargs.cfg.model.corrupt_types,
                force_atom_types
            )

            samples = model.sample(
                frac_coords=frac_coords,
                lattices=lattices,
                atom_types=atom_types,
                natoms=natoms,
                ld_kwargs=ld_kwargs,
                z=None,
                labels=labels,
                guidance_strength=guidance_strength,
                input_encoder=batch,
            )

            # collect sampled crystals in this batch.
            stored_data = store_samples(samples, stored_data,
                                        ld_kwargs.save_traj,
                                        down_sample_traj_step)

        stacked_data = store_samples(stored_data, stacked_data,
                                     ld_kwargs.save_traj, scatck="stack")
        input_data_list = input_data_list + batch.to_data_list()

    input_data_batch = Batch.from_data_list(input_data_list).to_dict()
    stacked_data = store_samples(stacked_data, {}, ld_kwargs.save_traj, scatck="cat")
    stacked_data.update({"input_data_batch": input_data_batch})
    return stacked_data


def partial_generation(loader, model, ld_kwargs, num_evals, sample_num_atoms,
                       force_num_atoms=False, force_atom_types=False, batch_size=None,
                       down_sample_traj_step=1, labels=None, guidance_strength=1):
    """
    Generate some crystal properties either
    frac_coords, lattices, or atom_types given some crystal information.
    """
    input_data_list = []
    stacked_data = {}

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f"batch {idx} in {len(loader)}")

        stored_data = {}

        for eval_idx in range(num_evals):
            if force_num_atoms or force_atom_types:
                natoms = batch.num_atoms
            else:
                natoms = sample_natoms(model, sample_num_atoms, batch_size)

            frac_coords, lattices, atom_types = sample_properties(
                model, natoms, batch, ld_kwargs.cfg.model.corrupt_coords,
                ld_kwargs.cfg.model.corrupt_lattices, ld_kwargs.cfg.model.corrupt_types,
                force_atom_types
            )

            samples = model.sample(
                frac_coords=frac_coords,
                lattices=lattices,
                atom_types=atom_types,
                natoms=natoms,
                ld_kwargs=ld_kwargs,
                z=None,
                labels=labels,
                guidance_strength=guidance_strength,
            )

            # collect sampled crystals in this batch.
            stored_data = store_samples(samples, stored_data,
                                        ld_kwargs.save_traj,
                                        down_sample_traj_step)

        stacked_data = store_samples(stored_data, stacked_data,
                                     ld_kwargs.save_traj, scatck="stack")
        input_data_list = input_data_list + batch.to_data_list()

    input_data_batch = Batch.from_data_list(input_data_list).to_dict()
    stacked_data = store_samples(stacked_data, {}, ld_kwargs.save_traj, scatck="cat")
    stacked_data.update({"input_data_batch": input_data_batch})
    return stacked_data


def full_generation(model, ld_kwargs, num_batches_to_sample,
                    sample_num_atoms, batch_size=64, down_sample_traj_step=1,
                    labels=None, guidance_strength=1, num_samples_per_z=1):
    """
    Generate all crystal properties, frac_coords, lattices, and atom_types.
    """
    stacked_data = {}

    for z_idx in range(num_batches_to_sample):
        stored_data = {}

        for sample_idx in range(num_samples_per_z):
            natoms = sample_natoms(model, sample_num_atoms, batch_size)
            frac_coords, lattices, atom_types = sample_properties(model, natoms)

            samples = model.sample(
                frac_coords=frac_coords,
                lattices=lattices,
                atom_types=atom_types,
                natoms=natoms,
                ld_kwargs=ld_kwargs,
                z=None,
                labels=labels,
                guidance_strength=guidance_strength,
            )

            # collect sampled crystals in this batch.
            stored_data = store_samples(samples, stored_data,
                                        ld_kwargs.save_traj,
                                        down_sample_traj_step)

        # collect sampled crystals for this z.
        stacked_data = store_samples(stored_data, stacked_data,
                                     ld_kwargs.save_traj, scatck="stack")

    return store_samples(stacked_data, {}, ld_kwargs.save_traj, scatck="cat")


def run_eval(args):
    model_path = Path(args.model_path)
    cfg = OmegaConf.create(OmegaConf.to_container(
        OmegaConf.load(str(model_path / "hparams.yaml")), resolve=True))

    load_data = not (
            cfg.model.corrupt_coords and
            cfg.model.corrupt_lattices and
            cfg.model.corrupt_types
    ) or args.load_data or args.force_num_atoms or args.force_atom_types

    if load_data:
        if args.dataset_path is not None:
            try:
                OmegaConf.update(cfg, "data.datamodule.datasets.test.*.path", args.dataset_path)
            except Exception as e:
                # print("Exception:", e)
                cfg.data.datamodule.datasets.test[0]["path"] = os.path.abspath(args.dataset_path)
        OmegaConf.update(cfg, "data.datamodule.batch_size.test", args.batch_size)

    model, test_loader = load_model(get_model, model_path,
                                    cfg, load_data=load_data)
    ld_kwargs = SimpleNamespace(cfg=cfg,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar,
                                adaptive_timestep=args.adaptive_timestep)

    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    if args.sample_num_atoms == "random":
        if hasattr(cfg.model, "min_atoms"):
            min_num_atoms = cfg.model.min_atoms
        else:
            min_num_atoms = 1
        args.sample_num_atoms = {"low": min_num_atoms,
                                 "high": cfg.model.max_atoms}
    elif "stat" in args.sample_num_atoms:
        pass
    else:
        args.sample_num_atoms = int(args.sample_num_atoms)

    if "gen" in args.tasks:
        if load_data:
            print("Evaluate model on the partial generation task.")
            start_time = time.time()
            samples = partial_generation(
                loader=test_loader,
                model=model,
                ld_kwargs=ld_kwargs,
                num_evals=args.num_evals,
                sample_num_atoms=args.sample_num_atoms,
                batch_size=args.batch_size,
                force_num_atoms=args.force_num_atoms,
                force_atom_types=args.force_atom_types,
                down_sample_traj_step=args.down_sample_traj_step,
                labels=args.labels,
                guidance_strength=args.guidance_strength,
            )

        else:
            print("Evaluate model on the full generation task.")
            start_time = time.time()

            samples = full_generation(model=model, ld_kwargs=ld_kwargs,
                                      num_batches_to_sample=args.num_batches_to_samples,
                                      sample_num_atoms=args.sample_num_atoms, batch_size=args.batch_size,
                                      down_sample_traj_step=args.down_sample_traj_step, labels=args.labels,
                                      guidance_strength=args.guidance_strength)

        if args.suffix == "":
            out_name = "gen_samples.pt"
        else:
            out_name = f"gen_samples_{args.suffix}.pt"

        save_output(samples, model_path, out_name, args, load_data, start_time)

    if "recon" in args.tasks:
        print("Evaluate model on the reconstruction task.")
        start_time = time.time()
        samples = reconstruction(
            loader=test_loader,
            model=model,
            ld_kwargs=ld_kwargs,
            num_evals=args.num_evals,
            sample_num_atoms=args.sample_num_atoms,
            batch_size=args.batch_size,
            force_num_atoms=args.force_num_atoms,
            force_atom_types=args.force_atom_types,
            down_sample_traj_step=args.down_sample_traj_step,
            labels=args.labels,
            guidance_strength=args.guidance_strength,
        )

        if args.suffix == "":
            out_name = "recon_samples.pt"
        else:
            out_name = f"recon_samples_{args.suffix}.pt"

        save_output(samples, model_path, out_name, args, load_data, start_time)
