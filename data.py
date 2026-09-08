# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset, IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (PreTrainedTokenizer, DataCollatorForLanguageModeling)


class StatefulStreamingDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        context_length: int = 2048,
        rank: int = 0,
        world_size: int = 1,
        buffer_size: int = -1,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data = dataset
        self.context_length = context_length
        self.rank = rank
        self.world_size = world_size
        if buffer_size == -1:
            self.buffer_size = 1024 if context_length <= 2049 else 512
            self.buffer_size = 256 if context_length >= 8192 else self.buffer_size        
            self.buffer_size = 128 if context_length >= 16384 else self.buffer_size        
            self.buffer_size = 64 if context_length >= 32768 else self.buffer_size        
        else:
            self.buffer_size = buffer_size
        
        self.data = split_dataset_by_node(self.dataset, self.rank, self.world_size)
        if tokenizer.vocab_size < torch.iinfo(torch.int16).max:
            self.dtype = torch.int16
        elif tokenizer.vocab_size < torch.iinfo(torch.int32).max:
            self.dtype = torch.int32
        else:
            self.dtype = torch.int64
        self.states = None
        self.buffer = torch.tensor([], dtype=self.dtype)
        self.tokens = []
        self.rand_id = 0
        self.token_id = 0
        self.rng_state = None
        # ``datasets.IterableDataset.set_epoch`` writes ``_epoch`` and recent
        # datasets releases may expose ``epoch`` as a read-only property.
        self._epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(int(self._epoch) + self.rank)
        if self.rng_state is not None:
            g.set_state(self.rng_state)
        rand_it = self.randint(0, self.buffer_size, g=g)
        if self.states is not None:
            self.data.load_state_dict(self.states)
        for sample in self.tokenize(self.data):
            self.tokens += sample
            if len(self.buffer) < self.buffer_size:
                # max number of tokens allowed in the chunk buffer
                n_tokens = self.buffer_size * self.context_length
                if len(self.tokens) >= n_tokens:
                    self.buffer = torch.tensor(self.tokens[:n_tokens], dtype=self.dtype).view(self.buffer_size, -1)
                    self.tokens = self.tokens[n_tokens:]
            if len(self.buffer) >= self.buffer_size:
                yield from self.sample(rand_it)

        n_chunks = len(self.tokens) // self.context_length
        if n_chunks > 0:
            n_tokens = n_chunks * self.context_length
            self.buffer = torch.tensor(self.tokens[:n_tokens], dtype=self.dtype).view(n_chunks, -1)
            self.tokens = self.tokens[n_tokens:]
        for i in self.buffer[torch.randperm(len(self.buffer), generator=g)].unbind(0):
            yield {'input_ids': i.to(torch.long)}

    def tokenize(self, data, batch_size: int = 32):
        buffer = []
        for sample in data:
            buffer.append(sample['text'])
            if len(buffer) == batch_size:
                yield from self.tokenizer(buffer)['input_ids']
                buffer = []
        if len(buffer) > 0:
            yield from self.tokenizer(buffer)['input_ids']

    def sample(self, indices):
        n_tokens = (len(self.tokens) // self.context_length) * self.context_length
        while self.token_id < n_tokens:
            i = next(indices)
            start, end = self.token_id, self.token_id + self.context_length
            self.token_id += self.context_length
            yield {'input_ids': self.buffer[i].to(torch.long)}
            self.buffer[i] = torch.tensor(self.tokens[start:end], dtype=self.dtype)
        self.token_id = 0
        self.tokens = self.tokens[n_tokens:]

    def randint(self, low: int, high: int, batch_size: int = 32, g: torch.Generator = torch.Generator()) -> Iterable[int]:
        while True:
            # record the generator states before sampling
            self.rng_state = g.get_state()
            indices = torch.randint(low, high, (batch_size,), generator=g).tolist()
            for i in indices[self.rand_id:]:
                self.rand_id += 1
                yield i
            self.rand_id = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def state_dict(self):
        return {
            'states': self.data.state_dict(),
            'buffer': self.buffer.clone(),
            'tokens': deepcopy(self.tokens),
            'rand_id': self.rand_id,
            'token_id': self.token_id,
            'rng_state': self.rng_state,
            'epoch': self._epoch
        }

    def load_state_dict(self, state_dict):
        self.states = state_dict['states']
        self.buffer = state_dict['buffer']
        self.tokens = state_dict['tokens']
        self.rand_id = state_dict['rand_id']
        self.token_id = state_dict['token_id']
        self.rng_state = state_dict['rng_state']
        self._epoch = state_dict['epoch']
    


def resolve_streaming_files(corpus_name: str, path: str | os.PathLike, split: str) -> list[Path]:
    """Resolve the exact, ordered set of source shards for a streaming run."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"streaming data directory does not exist: {root}")
    if corpus_name == "fineweb-edu-sample":
        pattern = "*.parquet"
    elif corpus_name == "fineweb-edu" and split == "train":
        pattern = "*/*.parquet"
    elif corpus_name == "fineweb-edu":
        pattern = "*.parquet"
    elif corpus_name == "slimpajama" and split in {"train", "val"}:
        pattern = "*/*.jsonl.zst"
    elif corpus_name == "slimpajama" and split == "val_sampled":
        pattern = "*.parquet"
    else:
        raise ValueError(f"unsupported streaming corpus/split: {corpus_name!r}/{split!r}")
    files = sorted(candidate for candidate in root.glob(pattern) if candidate.is_file())
    if not files:
        suffix = "parquet files" if "parquet" in pattern else "dataset shards"
        raise FileNotFoundError(f"no {suffix} found under {root} for {corpus_name!r}/{split!r}")
    return files


def streaming_data_manifest(files: Iterable[str | os.PathLike]) -> dict[str, object]:
    """Return a stable inventory digest without reading multi-terabyte shard bodies."""

    entries = []
    for raw_path in files:
        path = Path(raw_path).resolve()
        stat = path.stat()
        entries.append({"path": str(path), "size": stat.st_size})
    entries.sort(key=lambda item: item["path"])
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(entries),
        "total_bytes": sum(int(item["size"]) for item in entries),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": entries,
    }


def get_stateful_stream_tok_dataset(corpus_name='slimpajama', path=None, split='train', tokenizer=None, block_size=2048, rank=0, world_size=1, batch_size=32, num_workers=8):
    files = resolve_streaming_files(corpus_name, path, split)
    if corpus_name == 'slimpajama':
        if split in ['train', 'val']:
            dataset = load_dataset('json', data_files=[str(p) for p in files], split='train', streaming=True, keep_in_memory=False)
        elif split == 'val_sampled':
            dataset = load_dataset('parquet', data_files=[str(p) for p in files], split='train', streaming=True, keep_in_memory=False)
        else:
            raise NotImplementedError
    elif corpus_name == 'fineweb-edu-sample':
        dataset = load_dataset('parquet', data_files=[str(p) for p in files], split='train', streaming=True)
    elif corpus_name == 'fineweb-edu':
        dataset = load_dataset('parquet', data_files=[str(p) for p in files], split='train', streaming=True, keep_in_memory=False)
    else:
        raise NameError(f"Unknown corpus name: {corpus_name}")
    assert dataset.n_shards != 0, "You are loading empty dataset, please check the path"
    manifest = streaming_data_manifest(files)
    print(
        f"Loading dataset from {Path(path).resolve()} with {dataset.n_shards} shards "
        f"(inventory={manifest['sha256']})"
    )
    buffer_size= -1 if split == 'train' else 1
    # we do not want distributed sharding during validation because it just brings A LOT OF headaches.
    world_size = world_size if split == 'train' else 1
    rank = rank if split == 'train' else 0
    dataset = StatefulStreamingDataset(dataset, tokenizer, context_length=block_size, rank=rank, world_size=world_size, buffer_size=buffer_size)
    loader = StatefulDataLoader(dataset=dataset,
                                batch_size=batch_size,
                                collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
                                num_workers=num_workers,
                                persistent_workers=num_workers > 0,
                                pin_memory=False
                                )
    loader.kdn_data_manifest = manifest
    return loader
    
