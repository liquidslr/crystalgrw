import torch
from torch import nn
from omegaconf import ListConfig


class BaseController(nn.Module):
    def __init__(self):
        super().__init__()

    def _forward(self, y, num_atoms) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, y, num_atoms) -> torch.Tensor:
        y = self._forward(y, num_atoms)
        if y.size(0) == num_atoms.size(0):
            y = y.repeat_interleave(num_atoms, dim=0)
        return y


class ConditionEmbedding(BaseController):
    def __init__(self, cfg):
        super().__init__()
        self.task = cfg.task

        if cfg.task == "classification":
            self.num_class = cfg.num_class
            if isinstance(cfg.num_class, list) or isinstance(cfg.num_class, ListConfig):
                self.emb = []
                for nc in cfg.num_class:
                    self.emb.append(nn.Embedding(nc, cfg.hidden_dim))
                self.emb = nn.ModuleList(self.emb)
                self.lin = nn.Linear(cfg.hidden_dim * len(cfg.num_class), cfg.hidden_dim)
            else:
                self.emb = nn.Embedding(cfg.num_class, cfg.hidden_dim)

        elif cfg.task == "regression":
            self.emb = nn.Sequential(nn.Linear(cfg.num_prop, cfg.hidden_dim),
                                     nn.SiLU(),
                                     nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                                     )
            try:
                self.norm_factor = cfg.num_class - 1
                if cfg.num_class <= 1:
                    self.norm_factor = 1
            except:
                self.norm_factor = 1

    def _forward(self, y, num_atoms):
        if self.task == "classification":
            y = y.long()
            if isinstance(self.num_class, list) or isinstance(self.num_class, ListConfig):
                condition = []
                for i, emb in enumerate(self.emb):
                    condition.append(emb(y[:, i]).squeeze(dim=1))
                condition = torch.cat(condition, dim=-1)
                condition = self.lin(condition)
            else:
                condition = self.emb(y).squeeze(dim=1)

        elif self.task == "regression":
            condition = self.emb(y / self.norm_factor)

        else:
            raise NotImplementedError

        return condition  #.repeat_interleave(num_atoms, dim=0)
