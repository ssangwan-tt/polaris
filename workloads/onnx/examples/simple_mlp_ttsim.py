#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

class SimpleMLP(SimNN.Module):
    """
    Simple MLP in TTSIM:

        x: [bs, in_dim]
        fc1: Linear(in_dim -> hidden_dim)
        ReLU
        Softmax(dim=-1)
        LayerNorm(hidden_dim)
        fc2: Linear(hidden_dim -> out_dim)
    """

    def __init__(self, objname: str, cfg: dict):
        super().__init__()
        self.name = objname

        self.bs         = int(cfg.get("bs", 1))
        self.in_dim     = int(cfg.get("in_dim", 1024))
        self.hidden_dim = int(cfg.get("hidden_dim", 2048))
        self.out_dim    = int(cfg.get("out_dim", 512))

        # nn.Linear → F.Linear (torch2ttsim mapping)
        self.fc1     = F.Linear(f"{self.name}_fc1", self.in_dim, self.hidden_dim, bias=True)
        self.relu    = F.Relu  (f"{self.name}_relu")
        self.softmax = F.Softmax(f"{self.name}_softmax", axis=-1)
        self.ln      = F.LayerNorm(f"{self.name}_ln", self.hidden_dim)  # NEW
        self.fc2     = F.Linear(f"{self.name}_fc2", self.hidden_dim, self.out_dim, bias=True)

        super().link_op2module()

    def set_batch_size(self, new_bs: int) -> None:
        self.bs = int(new_bs)

    def create_input_tensors(self) -> None:
        self.input_tensors = {
            "x_in": F._from_shape(
                "x_in",
                [self.bs, self.in_dim],
                is_param=False,
            )
        }

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    def analytical_param_count(self) -> int:
        return 0

    def __call__(self):
        x = self.input_tensors["x_in"]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.softmax(x)
        x = self.ln(x)
        x = self.fc2(x)
        return x