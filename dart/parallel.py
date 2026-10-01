"""Parallel decisions over one shared context.

The context (prefix) is prefilled once. Its attention cache is repeated along the batch, and the short
per-question suffixes run as one batch on top of it. Each question's answer is read from the hidden state of
its own last real token, so a question gets the same answer alone or batched with others.

Shapes: suffixes are right-padded (the cache holds the prefix on the left); position ids continue from the
prefix length; the attention mask covers prefix + suffix.
"""
import torch

from data_prep.schema import MAX_PROMPT_TOKENS, format_prefix, format_suffix
from dart.data import option_mask
from dart.model import MASK_VALUE, ranked_options

MODEL_NAME = "D.A.R.T."


@torch.no_grad()
def predict_shared(model, prefix_ids, suffix_ids, option_masks, pad_id=0):
    """Probabilities over option slots for each suffix, sharing one prefix. Returns a (K, slots) tensor.

    prefix_ids: 1-D LongTensor (at least one token). suffix_ids: list of 1-D LongTensors.
    option_masks: (K, slots) bool tensor.
    """
    device = option_masks.device
    prefix = prefix_ids.reshape(1, -1).to(device)
    prefix_len, count = prefix.shape[1], len(suffix_ids)
    lengths = torch.tensor([len(s) for s in suffix_ids], device=device)
    width = int(lengths.max())
    if prefix_len + width > MAX_PROMPT_TOKENS:
        raise ValueError(f"prefix ({prefix_len}) + longest suffix ({width}) exceeds {MAX_PROMPT_TOKENS} tokens")
    padded = torch.full((count, width), pad_id, dtype=torch.long, device=device)
    suffix_mask = torch.zeros(count, width, dtype=torch.long, device=device)
    for k, ids in enumerate(suffix_ids):
        padded[k, : len(ids)] = ids.to(device)
        suffix_mask[k, : len(ids)] = 1
    prefill = model.backbone(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                             position_ids=torch.arange(prefix_len, device=device).unsqueeze(0), use_cache=True)
    cache = prefill.past_key_values
    cache.batch_repeat_interleave(count)
    mask = torch.cat([torch.ones(count, prefix_len, dtype=torch.long, device=device), suffix_mask], dim=1)
    positions = (prefix_len + torch.arange(width, device=device)).unsqueeze(0).expand(count, width)
    hidden = model.backbone(input_ids=padded, attention_mask=mask, position_ids=positions,
                            past_key_values=cache, use_cache=True).last_hidden_state
    pooled = hidden[torch.arange(count, device=device), lengths - 1].float()
    logits = model.head(pooled).masked_fill(~option_masks, MASK_VALUE)
    return torch.softmax(logits, dim=-1)


class ParallelDecider:
    """Text interface: one context, several questions, one ranked list per question."""

    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def _encode(self, text):
        return torch.tensor(self.tokenizer(text, add_special_tokens=False)["input_ids"], device=self.device)

    def decide(self, context, questions):
        """questions: list of {"query": str, "options": [str, ...]}. Returns [[(option, probability), ...], ...]."""
        if not context:
            raise ValueError("a shared context is required; use the single-decision path otherwise")
        suffixes = [self._encode(format_suffix(q["query"], q["options"])) for q in questions]
        masks = torch.tensor([option_mask(len(q["options"])) for q in questions], device=self.device)
        pad_id = self.tokenizer.pad_token_id or 0
        probs = predict_shared(self.model, self._encode(format_prefix(context)), suffixes, masks, pad_id).cpu()
        return [ranked_options(probs[k], q["options"]) for k, q in enumerate(questions)]


def decisions_payload(question_ids, ranked_lists):
    """The PRD 5.1 output contract for a set of parallel decisions."""
    decisions = []
    for question_id, ranked in zip(question_ids, ranked_lists):
        decisions.append({
            "id": question_id,
            "ranked": [{"option": option, "probability": probability} for option, probability in ranked],
            "decision": ranked[0][0],
            "confidence": ranked[0][1],
        })
    return {"model": MODEL_NAME, "decisions": decisions}
