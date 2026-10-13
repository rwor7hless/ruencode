import contextlib
import math
import os
import time
from pathlib import Path

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


def pick_device(pref: str = "auto") -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def amp_dtype(device: str, precision: str = "auto"):
    """bf16 на Ampere/Ada и новее; на старых картах (T4, P100, V100) — fp16 с масштабированием loss;
    на CPU или при precision=fp32 — без смешанной точности (None)."""
    if device != "cuda" or precision == "fp32":
        return None
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def autocast(device: str, dtype=torch.bfloat16):
    if device == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def save_checkpoint(path, model, opt, scaler, step: int, elapsed: float, loss: float, n_chunks: int) -> None:
    """Атомарно: при обрыве во время записи старое сохранение остаётся целым."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                "elapsed": elapsed, "loss": loss, "n_chunks": n_chunks}, tmp)
    os.replace(tmp, path)


def train_lm(model, tokens: np.ndarray, lm_cfg: dict, seed: int, log_every: int = 50,
             ckpt_path=None, resume: bool = False) -> dict:
    """ckpt_path: куда сохраняться раз в lm_cfg["ckpt_minutes"] минут, при Ctrl+C и в конце;
    resume: продолжить с сохранения (порядок батчей задан seed, так что итог тот же, что без перерыва)."""
    torch.manual_seed(seed)
    device = pick_device(lm_cfg.get("device", "auto"))
    if device == "cpu":
        torch.set_num_threads(lm_cfg["threads"])
    else:
        torch.cuda.reset_peak_memory_stats()  # пик — только этого кандидата
    model.to(device)
    ctx, bs = lm_cfg["context"], lm_cfg["batch_size"]
    n_chunks = (len(tokens) - 1) // ctx
    order = np.random.default_rng(seed).permutation(n_chunks)
    steps = math.ceil(n_chunks / bs)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": lm_cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lm_cfg["lr"], betas=(0.9, 0.95),
                             fused=device == "cuda")
    dtype = amp_dtype(device, lm_cfg.get("precision", "auto"))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)  # для bf16 и fp32 — пустая обёртка
    warmup = max(1, int(steps * lm_cfg["warmup_frac"]))
    lr_max, lr_min = lm_cfg["lr"], lm_cfg["lr"] * 0.1
    start, before, loss_val = 0, 0.0, float("nan")
    if resume and ckpt_path and Path(ckpt_path).exists():
        st = torch.load(ckpt_path, map_location=device)
        if st["n_chunks"] != n_chunks:
            raise ValueError(f"{ckpt_path}: сохранение от другого корпуса ({st['n_chunks']} != {n_chunks} окон)")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        start, before, loss_val = st["step"], st["elapsed"], st["loss"]
        print(f"resume from step {start}/{steps}", flush=True)
    t0 = last_save = time.time()
    every = lm_cfg.get("ckpt_minutes", 10) * 60

    def save(done_steps: int) -> None:
        save_checkpoint(ckpt_path, model, opt, scaler, done_steps, before + time.time() - t0, loss_val, n_chunks)

    model.train()
    next_step = start
    try:
        for step in range(start, steps):
            idx = order[step * bs:(step + 1) * bs]
            win = np.stack([tokens[i * ctx:i * ctx + ctx + 1] for i in idx]).astype(np.int64)
            x, y = torch.from_numpy(win[:, :-1]).to(device), torch.from_numpy(win[:, 1:]).to(device)
            if step < warmup:
                lr = lr_max * (step + 1) / warmup
            else:
                prog = (step - warmup) / max(1, steps - warmup)
                lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * prog))
            for g in opt.param_groups:
                g["lr"] = lr
            with autocast(device, dtype):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            next_step = step + 1
            loss_val = loss.item()
            if step % log_every == 0:
                done = (step + 1 - start) * bs * ctx
                print(f"step {step}/{steps} loss {loss_val:.3f} tok/s {done / (time.time() - t0):.0f}", flush=True)
            if ckpt_path and time.time() - last_save >= every:
                save(next_step)
                last_save = time.time()
    except KeyboardInterrupt:
        if ckpt_path:
            save(next_step)  # незавершённый шаг начнётся заново
            print(f"saved {ckpt_path} at step {next_step}/{steps}", flush=True)
        raise
    if ckpt_path and start < steps:
        save(steps)  # обрыв во время оценки не потребует обучать заново
    secs = before + time.time() - t0
    trained = n_chunks * ctx
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    return {"device": device, "steps": steps, "train_tokens": trained, "seconds": round(secs, 1),
            "tokens_per_sec": trained / secs if secs else 0.0, "final_loss": loss_val, "peak_gpu_gb": peak,
            "precision": {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype, "fp32")}
