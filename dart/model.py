"""DecisionModel: backbone + last-token pooling + masked decision head. One forward pass, no LM head."""
import torch
from torch import nn

from data_prep.schema import MAX_OPTIONS

MASK_VALUE = -1e4


def position_ids_from_mask(attention_mask):
    """Positions that ignore left padding, so batched and single inputs see the same positions."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


class DecisionModel(nn.Module):
    def __init__(self, backbone, hidden_size, max_options=MAX_OPTIONS):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, max_options)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def features(self, input_ids, attention_mask):
        """Last-token hidden state (float32): the only thing the decision head reads."""
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids_from_mask(attention_mask),
            use_cache=False,
        ).last_hidden_state[:, -1]
        return hidden.float()

    def forward(self, input_ids, attention_mask, option_mask):
        return self.head(self.features(input_ids, attention_mask)).masked_fill(~option_mask, MASK_VALUE)

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, option_mask):
        """Probabilities over option slots (zero on slots beyond the option count)."""
        return torch.softmax(self(input_ids, attention_mask, option_mask), dim=-1)


def ranked_options(probabilities, options):
    """[(option, probability)] sorted by probability, highest first."""
    pairs = [(option, float(probabilities[i])) for i, option in enumerate(options)]
    return sorted(pairs, key=lambda pair: pair[1], reverse=True)
