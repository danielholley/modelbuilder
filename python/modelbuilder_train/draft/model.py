"""Speculative-decoding drafters trained on a frozen target's features.

- **Medusa** (Cai et al., 2024, arXiv 2401.10774, §2.1): ``K`` heads on the
  target's last hidden state ``h_t``; head ``k`` (1-based) predicts token
  ``x[t+k+1]``. Each head is a residual SiLU block, ``h + SiLU(W h + b)`` with
  ``W`` initialized to zero, followed by its own LM head initialized from the
  target's (Medusa-1: the backbone stays frozen). ``share_lm_head`` reuses the
  frozen target head instead, a cheaper variant. Tensor names follow vLLM's
  Medusa layout (``blocks.{k}.layers.0``, ``lm_heads.{k}``).
- **EAGLE** (Li et al., 2024, arXiv 2401.15077, §3): an autoregressive draft
  over features. ``fc(concat(embed(x[t+1]), h_t))`` (2h → h), then decoder
  layers of the target's own shape, predicts the next feature ``ĥ_{t+1}``; the
  frozen LM head turns it into a draft of ``x[t+2]``. As in the EAGLE code,
  the first layer has no input norm and there is no final norm (the features
  are already post-norm). The loss is SmoothL1 on the feature plus 0.1 ×
  cross-entropy on the token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from modelbuilder_train.mtp.model import DecoderLayer, MtpConfig

#: EAGLE §3.2: weight of the token loss next to the feature regression.
EAGLE_CLS_WEIGHT = 0.1
#: Medusa-1: head k's loss is weighted λ^k (λ = 0.8).
MEDUSA_DECAY = 0.8


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden)])
        nn.init.zeros_(self.layers[0].weight)  # starts as the identity: head k ≈ the target's head
        nn.init.zeros_(self.layers[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.silu(self.layers[0](x))


class Medusa(nn.Module):
    def __init__(self, hidden: int, heads: int, lm_head: torch.Tensor | None) -> None:
        """``lm_head`` [vocab, hidden] initializes each head's own LM head; ``None`` shares the frozen one."""
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(heads)])
        self.lm_heads = None
        if lm_head is not None:
            self.lm_heads = nn.ModuleList()
            for _ in range(heads):
                lin = nn.Linear(hidden, lm_head.shape[0], bias=False)
                with torch.no_grad():
                    lin.weight.copy_(lm_head)
                self.lm_heads.append(lin)

    @property
    def heads(self) -> int:
        return len(self.blocks)

    def head_weight(self, k: int, frozen: torch.Tensor) -> torch.Tensor:
        return frozen if self.lm_heads is None else self.lm_heads[k].weight

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """Per head, the hidden state its LM head reads."""
        return [b(hidden) for b in self.blocks]


class Eagle(nn.Module):
    def __init__(self, c: MtpConfig, blocks: int) -> None:
        super().__init__()
        self.config = c
        self.fc = nn.Linear(2 * c.hidden_size, c.hidden_size, bias=False)
        self.layers = nn.ModuleList([DecoderLayer(c) for _ in range(blocks)])
        self.layers[0].input_layernorm = nn.Identity()

    def forward(self, trunk_hidden: torch.Tensor, next_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Predicted next features [B, T, H]."""
        x = self.fc(torch.cat([next_embeds, trunk_hidden], dim=-1))
        for layer in self.layers:
            x = layer(x, positions)
        return x
