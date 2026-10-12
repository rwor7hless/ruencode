import math

import torch


@torch.no_grad()
def nll_bits_and_bytes(model, tok, texts, context: int) -> tuple[float, int]:
    model.eval()
    nll, nbytes = 0.0, 0
    for t in texts:
        ids = [tok.bos_id] + tok.encode(t)
        nbytes += len(t.encode("utf-8"))
        for s in range(0, len(ids) - 1, context):
            chunk = ids[s:s + context + 1]
            x = torch.tensor(chunk[:-1]).unsqueeze(0)
            y = torch.tensor(chunk[1:]).unsqueeze(0)
            _, loss = model(x, y, reduction="sum")
            nll += loss.item()
    return nll / math.log(2), nbytes
