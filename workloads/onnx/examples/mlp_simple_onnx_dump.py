#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # macOS OpenMP workaround

import torch
import torch.nn as nn
from pathlib import Path

class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 2048, out_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.ln = nn.LayerNorm(hidden_dim)       # NEW
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.softmax(x)
        x = self.ln(x)                           # NEW
        x = self.fc2(x)
        return x

def main() -> None:
    in_dim, hidden_dim, out_dim = 1024, 2048, 512
    bs = 1

    model = SimpleMLP(in_dim, hidden_dim, out_dim)
    model.eval()
    model.half()  # cast to fp16

    dummy = torch.randn(bs, in_dim).half()

    out_path = Path("workloads/onnx/mlp_simple.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    print(f"Saved {out_path} (fp16, with Softmax + LayerNorm)")

if __name__ == "__main__":
    main()