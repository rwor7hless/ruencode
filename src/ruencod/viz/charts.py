import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ruencod.config import DEFAULT_CONFIG, cfg_path, load_config

# палитра из скилла dataviz (проверена validate_palette.js, light): цвет закреплён за сущностью
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = {"ours": "#2a78d6", "claude": "#eb6834", "baseline": "#a3a29c",
           "bpe": "#2a78d6", "unigram": "#1baf7a", "morph-bpe": "#eda100"}
plt.rcParams.update({
    "font.size": 13, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlecolor": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def _fert_metric(df) -> str:
    """Сравнение с Claude честно только на его выборке; без неё — fertility на всём heldout."""
    has = ((df.metric == "fertility_claude_sample") & (df.domain == "ru_all")).any()
    return "fertility_claude_sample" if has else "fertility"


def _get(df, metric, domain="ru_all") -> dict:
    sub = df[(df.metric == metric) & (df.domain == domain)]
    return dict(zip(sub.tokenizer, sub.value))


def _label(cfg, name):
    for b in cfg["eval"]["baselines"]:
        if b["name"] == name:
            return b.get("label", name)
    if name == cfg["prices"]["claude"]["tokenizer"]:
        return cfg["prices"]["claude"]["label"]
    return f"ruEncod {name}"


def _kind(df, name):
    return df[df.tokenizer == name].kind.iloc[0]


def _hbar(ax, labels, values, colors, fmt):
    ax.barh(labels, values, color=colors, height=0.6)
    for i, v in enumerate(values):
        ax.text(v, i, f"  {fmt(v)}", va="center", color=TEXT)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.18)


def fertility_bar(cfg, df, out: Path) -> Path:
    vals = _get(df, _fert_metric(df))
    final_v = cfg["final"]["tokenizer"].split("-")[-1]
    keep = {k: v for k, v in vals.items() if _kind(df, k) != "ours" or k.endswith(final_v)}
    items = sorted(keep.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(items) + 1.5))
    _hbar(ax, [_label(cfg, k) for k, _ in items], [v for _, v in items],
          [PALETTE[_kind(df, k)] for k, _ in items], lambda v: f"{v:.2f}")
    ax.set_xlabel("токенов на слово русского текста (меньше — лучше)")
    fig.savefig(out)
    plt.close(fig)
    return out


def vocab_curve(cfg, df, out: Path) -> Path:
    vals = _get(df, "fertility")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sizes = sorted(cfg["train"]["vocab_sizes"])
    for algo in cfg["train"]["algos"]:
        pts = [(v // 1000, vals[f"{algo}-{v // 1000}k"]) for v in sizes if f"{algo}-{v // 1000}k" in vals]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", lw=2, ms=8, color=PALETTE[algo], label=algo, mec=SURFACE, mew=2)
        ax.annotate(f" {algo}", (xs[-1], ys[-1]), color=TEXT, va="center")
    if "o200k" in vals:
        ax.axhline(vals["o200k"], ls="--", lw=1.5, color=PALETTE["baseline"], label=_label(cfg, "o200k"))
    ax.set_xlabel("размер словаря, тыс. токенов")
    ax.set_ylabel("токенов на слово")
    ax.legend(frameon=False)
    fig.savefig(out)
    plt.close(fig)
    return out


def domains_bar(cfg, df, out: Path) -> Path:
    metric = _fert_metric(df)
    names = [n for n in (cfg["final"]["tokenizer"], "o200k", cfg["prices"]["claude"]["tokenizer"])
             if n in set(df.tokenizer)]
    sub = df[(df.metric == metric) & (df.tokenizer.isin(names)) & (~df.domain.isin(["ru_all", "morph_test"]))]
    pv = sub.pivot_table(index="domain", columns="tokenizer", values="value").reindex(columns=names)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        xs = [x + (i - (len(names) - 1) / 2) * w for x in range(len(pv))]
        ax.bar(xs, pv[n], width=w * 0.92, label=_label(cfg, n), color=PALETTE[_kind(df, n)])
    ax.set_xticks(range(len(pv)), pv.index, rotation=20)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("токенов на слово")
    ax.legend(frameon=False)
    fig.savefig(out)
    plt.close(fig)
    return out


def cost_bar(cfg, df, out: Path) -> Path:
    f = _get(df, _fert_metric(df))
    final = cfg["final"]["tokenizer"]
    bars = []
    for key in ("gpt", "claude"):
        p = cfg["prices"][key]
        if p["tokenizer"] not in f:
            continue
        price = p["usd_per_mtok"]
        bars.append((p["label"], f[p["tokenizer"]] * price / 1000, _kind(df, p["tokenizer"])))
        bars.append((f"ruEncod по цене {p['label']}", f[final] * price / 1000, "ours"))
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(bars) + 1.5))
    _hbar(ax, [b[0] for b in bars], [b[1] for b in bars], [PALETTE[b[2]] for b in bars], lambda v: f"${v:.4f}")
    ax.set_xlabel("$ за 1000 слов русского текста (входные токены)")
    fig.savefig(out)
    plt.close(fig)
    return out


def lm_bar(cfg, lm: pd.DataFrame, out: Path) -> Path:
    sub = lm[lm.domain == "ru_all"].sort_values("bpb")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(sub) + 1.5))
    colors = [PALETTE["baseline"] if c == "o200k" else PALETTE[c.rsplit("-", 1)[0]] for c in sub.candidate]
    _hbar(ax, [_label(cfg, c) for c in sub.candidate], list(sub.bpb), colors, lambda v: f"{v:.3f}")
    ax.set_xlabel("бит на байт текста (меньше — лучше)")
    fig.savefig(out)
    plt.close(fig)
    return out


def make_all(cfg: dict) -> list[Path]:
    res = cfg_path(cfg, "results")
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res / "metrics.csv")
    paths = [fertility_bar(cfg, df, figs / "1_fertility.png"), vocab_curve(cfg, df, figs / "2_vocab_curve.png"),
             domains_bar(cfg, df, figs / "3_domains.png"), cost_bar(cfg, df, figs / "4_cost.png")]
    if (res / "lm_bpb.csv").exists():
        paths.append(lm_bar(cfg, pd.read_csv(res / "lm_bpb.csv"), figs / "5_lm_bpb.png"))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    for p in make_all(load_config(ap.parse_args().config)):
        print(p)


if __name__ == "__main__":
    main()
