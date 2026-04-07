import glob
import torch
import numpy as np
from pathlib import Path
import sentencepiece as spm
import time

def load_data_shard(file: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def debug():
    pattern = "data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    print(f"Found {len(files)} files.")
    if not files: return
    
    sp = spm.SentencePieceProcessor(model_file="data/tokenizers/fineweb_1024_bpe.model")
    eos_id = sp.eos_id()
    print(f"Tokenizer eos_id = {eos_id}")
    
    first_file = load_data_shard(files[0])
    print(f"First file loaded: {first_file.shape} tokens.")
    
    # Check for eos presence
    count_eos = (first_file == eos_id).sum().item()
    print(f"Number of eos_ids found in first file: {count_eos}")
    
    # Are there ANY special tokens? 0, 1, 2, 3?
    for i in range(5):
        print(f"Count of id {i}: {(first_file == i).sum().item()}")

if __name__ == "__main__":
    debug()
