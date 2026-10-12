import argparse
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config
from ruencod.eval.adapters import resolve
from ruencod.eval.claude_count import ClaudeCounter
from ruencod.eval.metrics import aligned_pieces
from ruencod.viz.charts import _label

COLORS = ["#cde2fb", "#fbd9c9", "#c9efe0", "#fbe6b8"]  # светлые ступени палитры dataviz
_FONT = font_manager.findfont("DejaVu Sans")


def _shown(piece: str) -> str:
    return piece.replace(" ", "·").replace("\n", "↵")


def render(rows, title: str, out: Path, width: int = 1600) -> Path:
    big, small = ImageFont.truetype(_FONT, 34), ImageFont.truetype(_FONT, 26)
    label_w, pad, line_h = 400, 24, 64
    lines = []
    for label, content in rows:
        if isinstance(content, str):
            lines.append((label, content, 1))
            continue
        x, n_rows = label_w, 1
        for text, _ in content:
            w = big.getlength(_shown(text)) + 20
            if x + w > width - pad - 230:
                x, n_rows = label_w, n_rows + 1
            x += w + 6
        lines.append((label, content, n_rows))
    height = 120 + sum(n * line_h + 30 for *_, n in lines)
    img = Image.new("RGB", (width, height), "#fcfcfb")
    d = ImageDraw.Draw(img)
    d.text((pad, 30), title, font=big, fill="#0b0b0b")
    y = 110
    for label, content, n_rows in lines:
        d.text((pad, y + 14), label, font=small, fill="#52514e")
        if isinstance(content, str):
            d.text((label_w, y + 10), content, font=big, fill="#0b0b0b")
        else:
            total = sum(n for _, n in content)
            x, row_y = label_w, y
            for i, (text, n) in enumerate(content):
                s = _shown(text)
                w = big.getlength(s) + 20
                if x + w > width - pad - 230:
                    x, row_y = label_w, row_y + line_h
                d.rounded_rectangle([x, row_y, x + w, row_y + 52], 8, fill=COLORS[i % len(COLORS)])
                d.text((x + 10, row_y + 6), s, font=big, fill="#0b0b0b")
                if n > 1:
                    d.text((x + w - 30, row_y - 8), f"×{n}", font=small, fill="#d03b3b")
                x += w + 6
            d.text((width - pad - 190, y + 14), f"{total} ток.", font=small, fill="#52514e")
        y += n_rows * line_h + 30
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def make_all(cfg: dict, claude_client=None) -> list[Path]:
    figs = cfg_path(cfg, "results") / "figures"
    names = ["o200k", "llama-3", cfg["final"]["tokenizer"]]
    morph = "morph-" + cfg["final"]["tokenizer"].replace("morph-", "").replace("unigram", "bpe")
    if morph != cfg["final"]["tokenizer"]:
        names.append(morph)
    toks = []
    for n in names:
        try:
            toks.append(resolve(cfg, n))
        except KeyError:  # нет в конфиге/артефактах (например, в smoke) — строку пропускаем
            print(f"skip {n}: not available")
    counter = None
    if cfg["eval"]["claude"]["models"]:
        counter = ClaudeCounter(cfg["prices"]["claude"]["tokenizer"],
                                cfg_path(cfg, "results") / "claude_cache.json", client=claude_client)
    paths = []
    for i, text in enumerate(cfg["examples"]):
        rows = [(_label(cfg, t.name), aligned_pieces(t, text)) for t in toks]
        if counter:
            rows.insert(1, (cfg["prices"]["claude"]["label"], f"{counter.count(text)} токенов (разбиение закрыто)"))
        paths.append(render(rows, text, figs / f"split_{i}.png"))
    if counter:
        counter.save()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    for p in make_all(load_config(ap.parse_args().config)):
        print(p)


if __name__ == "__main__":
    main()
