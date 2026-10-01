"""CUDA-graph inference for DecisionModel.

Eager PyTorch launches hundreds of small GPU operations from Python, so a single decision is
CPU-bound (about 70 ms for Qwen3-0.6B on a laptop GPU). A CUDA graph records the whole forward
pass once and replays it as one launch (about 14 ms). Graphs need fixed shapes, so prompts are
left-padded up to the nearest length bucket, exactly like normal left-padded batching.

Not thread-safe: one predictor serves one stream of calls.
"""
import torch

from data_prep.schema import MAX_OPTIONS

DEFAULT_BUCKETS = (64, 96, 128, 192, 256)
WARMUP_RUNS = 3


class GraphedPredictor:
    def __init__(self, model, batch_sizes=(1,), buckets=DEFAULT_BUCKETS):
        self.model = model.eval()
        self.buckets = tuple(sorted(buckets))
        self.device = next(model.parameters()).device
        self.pool = torch.cuda.graph_pool_handle()
        self.entries = {(batch, length): self._capture(batch, length)
                        for batch in batch_sizes for length in self.buckets}

    def _capture(self, batch, length):
        static = {
            "input_ids": torch.zeros(batch, length, dtype=torch.long, device=self.device),
            "attention_mask": torch.ones(batch, length, dtype=torch.long, device=self.device),
            "option_mask": torch.ones(batch, MAX_OPTIONS, dtype=torch.bool, device=self.device),
        }
        args = (static["input_ids"], static["attention_mask"], static["option_mask"])
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(WARMUP_RUNS):
                self.model.predict(*args)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            output = self.model.predict(*args)
        return graph, static, output

    def predict(self, input_ids, attention_mask, option_mask):
        """Probabilities over option slots, same result as `model.predict` (within float tolerance)."""
        batch, length = input_ids.shape
        bucket = next((b for b in self.buckets if b >= length), None)
        if bucket is None or (batch, bucket) not in self.entries:
            raise ValueError(f"no captured graph for batch {batch} and length {length}")
        graph, static, output = self.entries[(batch, bucket)]
        pad = bucket - length
        static["input_ids"].zero_()
        static["attention_mask"].zero_()
        static["input_ids"][:, pad:] = input_ids
        static["attention_mask"][:, pad:] = attention_mask
        static["option_mask"].copy_(option_mask)
        graph.replay()
        return output.clone()
