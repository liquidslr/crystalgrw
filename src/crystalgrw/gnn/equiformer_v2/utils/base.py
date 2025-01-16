"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from ..common.utils import (
    compute_neighbors,
    conditional_grad,
    #get_pbc_distances,
    #radius_graph_pbc,
)

from crystalgrw.common.data_utils import get_pbc_distances, radius_graph_pbc


class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None):
        super(BaseModel, self).__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

    def generate_graph(
        self,
        pos,
        natoms,
        lengths,
        angles,
        batch,
        cell_offsets=None,
        edge_index=None,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
    ):
        # cutoff = cutoff or self.cutoff
        # max_neighbors = max_neighbors or self.max_neighbors
        # use_pbc = use_pbc or self.use_pbc
        # otf_graph = otf_graph or self.otf_graph
        
        if use_pbc:
            if otf_graph:
                # edge_index, cell_offsets, neighbors = radius_graph_pbc(
                #     pos, natoms, cell, cutoff, max_neighbors
                # )
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    pos, (lengths, angles), natoms, cutoff, max_neighbors,
                    device=pos.device
                )


            # out = get_pbc_distances(
            #    pos,
            #    edge_index,
            #    cell,
            #    cell_offsets,
            #    neighbors,
            #    return_offsets=True,
            #    return_distance_vec=True,
            # )
            out = get_pbc_distances(
                pos,
                edge_index,
                lengths,
                angles,
                cell_offsets,
                natoms,
                neighbors,
                coord_is_cart=True,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
            
        else:
            if otf_graph:
                edge_index = radius_graph(
                    pos,
                    r=cutoff,
                    batch=batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = pos[j] - pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=pos.device
            )
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=pos.device
            )
            neighbors = compute_neighbors(pos, natoms, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
