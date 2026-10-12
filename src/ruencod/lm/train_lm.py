import math
import time

import numpy as np
import torch


def encode_corpus(tok, texts) -> np.ndarray:
    parts = []
    batch = []
    for t in texts:
        batch.append(t)
        if len(batch) == 256:
            parts += [[tok.bos_id] + ids for ids in tok.encode_batch(batch)]
            batch = []
    if batch:
        parts += [[tok.bos_id] + ids for ids in tok.encode_batch(batch)]
    return np.fromiter((i for p in parts for i in p), dtype=np.uint32)


def train_lm(model, tokens: np.ndarray, lm_cfg: dict, seed: int, log_every: int = 50) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(lm_cfg["threads"])
    ctx, bs = lm_cfg["context"], lm_cfg["batch_size"]
    n_chunks = (len(tokens) - 1) // ctx
    order = np.random.default_rng(seed).permutation(n_chunks)
    steps = math.ceil(n_chunks / bs)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": lm_cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lm_cfg["lr"], betas=(0.9, 0.95))
    warmup = max(1, int(steps * lm_cfg["warmup_frac"]))
    lr_max, lr_min = lm_cfg["lr"], lm_cfg["lr"] * 0.1
    t0, loss_val = time.time(), float("nan")
    model.train()
    for step in range(steps):
        idx = order[step * bs:(step + 1) * bs]
        win = np.stack([tokens[i * ctx:i * ctx + ctx + 1] for i in idx]).astype(np.int64)
        x, y = torch.from_numpy(win[:, :-1]), torch.from_numpy(win[:, 1:])
        if step < warmup:
            lr = lr_max * (step + 1) / warmup
        else:
            prog = (step - warmup) / max(1, steps - warmup)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = lr
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_val = loss.item()
        if step % log_every == 0:
            done = (step + 1) * bs * ctx
            print(f"step {step}/{steps} loss {loss_val:.3f} tok/s {done / (time.time() - t0):.0f}", flush=True)
    secs = time.time() - t0
    trained = n_chunks * ctx
    return {"steps": steps, "train_tokens": trained, "seconds": round(secs, 1),
            "tokens_per_sec": trained / secs if secs else 0.0, "final_loss": loss_val}
