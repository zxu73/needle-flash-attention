"""Model benchmark: a Transformer language model with and without flash attention.

Run from the project root:
    python apps/bench_flash_transformer.py --device cuda --ptb-dir data/ptb \
        --out results/transformer_cuda.json

Both experiments use apps/models.LanguageModel(seq_model='transformer'), built twice from the
same seed: once with the default attention ("naive") and once with flash=True ("flash").

    step      one training step (forward, loss, backward, Adam update) on random tokens for
              increasing sequence lengths: median wall time and peak memory
    training  --train-steps steps of next-word prediction on Penn Treebank, recording the loss
              after every step; the two models must follow the same curve

The training experiment needs train.txt and test.txt of Penn Treebank in --ptb-dir (the files
used in homework 4) and is skipped when they are not there.
"""
import argparse
import gc
import os
import sys
import time

sys.path.append("./python")

import numpy as np

import needle as ndl
import needle.nn as nn
from bench_common import (GB, PeakMemory, describe_hardware, format_cell, get_device, measure,
                          save_results)
from models import LanguageModel

IMPLS = ["flash", "naive"]
LABELS = {"flash": "nn.Transformer(flash=True)", "naive": "nn.Transformer, default attention"}
NUM_HEAD, DIM_HEAD = 8, 32  # nn.Transformer's defaults, which LanguageModel uses

# Peak memory of a training step with the default attention, relative to the (B, H, N, N, D)
# products of all layers together; measured with this script at small N.  Only used to decide
# which runs to skip (--budget-gb).
NAIVE_STEP_PEAK_FACTOR = 9.0


def build_model(vocab, seq_len, flash, args, device):
    np.random.seed(0)  # both variants start from the same parameters
    return LanguageModel(args.embedding_size, vocab, args.hidden_size, num_layers=args.layers,
                         seq_model="transformer", seq_len=seq_len, flash=flash, device=device)


def train_step(model, opt, loss_fn, tokens, targets):
    """One optimizer step; returns the loss before the update."""
    opt.reset_grad()
    logits, _ = model(tokens)
    loss = loss_fn(logits, targets)
    loss.backward()
    opt.step()
    return float(loss.numpy())  # the copy to the host also synchronizes cuda


def exceeds_budget(impl, seq_len, args):
    if impl != "naive":
        return False
    products = args.layers * 4 * args.batch * NUM_HEAD * seq_len * seq_len * DIM_HEAD
    return NAIVE_STEP_PEAK_FACTOR * products > args.budget_gb * GB


def warm_up(args, device, steps=10):
    """A few untimed training steps, so the first measurement does not include one-time costs
    (an idle GPU clocking up, first-use initialization of the cuda runtime)."""
    seq_len = min(args.lens)
    ids = np.zeros((seq_len, args.batch), dtype=np.float32)
    tokens = ndl.Tensor(ids, device=device, requires_grad=False)
    targets = ndl.Tensor(ids.reshape(-1), device=device, requires_grad=False)
    model = build_model(args.vocab, seq_len, True, args, device)
    opt = ndl.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(steps):
        train_step(model, opt, nn.SoftmaxLoss(), tokens, targets)


def bench_training_step(args, device):
    """Time and memory of one training step against the sequence length."""
    warm_up(args, device)
    records = []
    for seq_len in args.lens:
        rng = np.random.default_rng(seq_len)
        ids = rng.integers(0, args.vocab, size=(seq_len + 1, args.batch)).astype(np.float32)
        tokens = ndl.Tensor(ids[:-1], device=device, requires_grad=False)
        targets = ndl.Tensor(ids[1:].reshape(-1), device=device, requires_grad=False)
        for impl in IMPLS:
            record = dict(experiment="step", impl=impl, N=seq_len, time_ms=None, peak_mb=None)
            if exceeds_budget(impl, seq_len, args):
                record["status"] = "skipped"
            else:
                model = build_model(args.vocab, seq_len, impl == "flash", args, device)
                opt = ndl.optim.Adam(model.parameters(), lr=args.lr)
                loss_fn = nn.SoftmaxLoss()
                record.update(measure(
                    lambda: train_step(model, opt, loss_fn, tokens, targets), args.repeats))
                del model, opt
            records.append(record)
            print(f"  step      N={seq_len:<5} {impl:<5} "
                  f"{format_cell(record, 'time_ms', '.1f'):>10} ms "
                  f"{format_cell(record, 'peak_mb', '.1f'):>10} MB", flush=True)
    return records


def bench_training(args, device):
    """Loss curves of the two variants on Penn Treebank."""
    if not all(os.path.exists(os.path.join(args.ptb_dir, f)) for f in ("train.txt", "test.txt")):
        print(f"\nno Penn Treebank files in {args.ptb_dir}: skipping the training experiment")
        return []
    corpus = ndl.data.Corpus(args.ptb_dir, max_lines=args.ptb_lines)
    vocab = len(corpus.dictionary)
    data = ndl.data.batchify(corpus.train, args.batch, device=device, dtype="float32")
    seq_len = args.train_seq_len
    steps = min(args.train_steps, (data.shape[0] - 1) // seq_len)
    print(f"\ntraining: PTB, {len(corpus.train)} tokens, vocabulary {vocab}, "
          f"{steps} steps of {seq_len} x {args.batch} tokens")

    records = []
    for impl in IMPLS:
        record = dict(experiment="training", impl=impl, N=seq_len, vocab=vocab, losses=[],
                      time_ms=None, peak_mb=None)
        if exceeds_budget(impl, seq_len, args):
            record["status"] = "skipped"
            records.append(record)
            continue
        model = build_model(vocab, seq_len, impl == "flash", args, device)
        model.train()
        opt = ndl.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.SoftmaxLoss()
        times, peak = [], 0
        for step in range(steps):
            tokens, targets = ndl.data.get_batch(data, step * seq_len, seq_len, device=device)
            gc.collect()  # free the previous step's graph (see bench_common.measure)
            start = time.perf_counter()
            if step == 1:  # after the warm-up step, before timing settles
                with PeakMemory() as tracker:
                    loss = train_step(model, opt, loss_fn, tokens, targets)
                peak = tracker.peak
            else:
                loss = train_step(model, opt, loss_fn, tokens, targets)
                times.append(time.perf_counter() - start)
            record["losses"].append(loss)
            if step % max(1, steps // 10) == 0 or step == steps - 1:
                print(f"  training  {impl:<5} step {step:>4}  loss {loss:.4f}", flush=True)
        record.update(status="ok", time_ms=float(np.median(times[1:] or times)) * 1e3,
                      peak_mb=peak / 2**20)
        records.append(record)
        del model, opt
    return records


def print_step_table(records):
    find = lambda N, impl: next(r for r in records if (r["N"], r["impl"]) == (N, impl))
    header = (f"{'N':>6} | {'flash ms':>10} {'naive ms':>10} {'speedup':>8} | "
              f"{'flash MB':>10} {'naive MB':>10} {'ratio':>8}")
    print(f"\none training step\n{header}\n{'-' * len(header)}")
    for N in sorted({r["N"] for r in records}):
        flash, naive = find(N, "flash"), find(N, "naive")
        both = flash["status"] == naive["status"] == "ok"
        speedup = f"{naive['time_ms'] / flash['time_ms']:.1f}x" if both else "-"
        ratio = f"{naive['peak_mb'] / flash['peak_mb']:.1f}x" if both else "-"
        print(f"{N:>6} | {format_cell(flash, 'time_ms', '.1f'):>10} "
              f"{format_cell(naive, 'time_ms', '.1f'):>10} {speedup:>8} | "
              f"{format_cell(flash, 'peak_mb', '.1f'):>10} "
              f"{format_cell(naive, 'peak_mb', '.1f'):>10} {ratio:>8}")


def print_training_summary(records):
    by_impl = {r["impl"]: r for r in records}
    print("\ntraining on Penn Treebank")
    for impl, r in by_impl.items():
        if r["status"] != "ok":
            print(f"  {impl:<5} {r['status']}")
            continue
        print(f"  {impl:<5} loss {r['losses'][0]:.4f} -> {r['losses'][-1]:.4f}   "
              f"{r['time_ms']:.1f} ms / step   peak {r['peak_mb']:.1f} MB")
    if all(r["status"] == "ok" for r in by_impl.values()) and len(by_impl) == 2:
        gap = np.abs(np.array(by_impl["flash"]["losses"]) - np.array(by_impl["naive"]["losses"]))
        print(f"  largest difference between the two loss curves: {gap.max():.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "cpu_numpy"])
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256, help="width of each layer's MLP")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--budget-gb", type=float, default=4.0,
                        help="skip the default attention when its estimated peak memory "
                             "exceeds this")
    parser.add_argument("--out", default=None, help="write the records to this JSON file")
    step = parser.add_argument_group("training-step experiment")
    step.add_argument("--lens", type=int, nargs="*", default=[32, 64, 128, 256, 512, 1024],
                      help="sequence lengths (none: skip the experiment)")
    step.add_argument("--vocab", type=int, default=512)
    step.add_argument("--repeats", type=int, default=3)
    train = parser.add_argument_group("Penn Treebank training experiment")
    train.add_argument("--ptb-dir", default="data/ptb")
    train.add_argument("--ptb-lines", type=int, default=2000,
                       help="lines of each PTB file to read (sets the vocabulary size)")
    train.add_argument("--train-steps", type=int, default=100,
                       help="optimizer steps (0: skip the experiment)")
    train.add_argument("--train-seq-len", type=int, default=128)
    args = parser.parse_args()

    device = get_device(args.device)
    meta = dict(benchmark="transformer", device=args.device,
                hardware=describe_hardware(args.device), embedding_size=args.embedding_size,
                hidden_size=args.hidden_size, layers=args.layers, num_head=NUM_HEAD,
                dim_head=DIM_HEAD, batch=args.batch, lr=args.lr, vocab=args.vocab,
                budget_gb=args.budget_gb, labels=LABELS)
    print(f"device={args.device} ({meta['hardware']})  layers={args.layers} "
          f"heads={NUM_HEAD}x{DIM_HEAD} embedding={args.embedding_size} batch={args.batch}")

    records = bench_training_step(args, device) if args.lens else []
    if args.train_steps > 0:
        records += bench_training(args, device)

    if args.lens:
        print_step_table([r for r in records if r["experiment"] == "step"])
    if any(r["experiment"] == "training" for r in records):
        print_training_summary([r for r in records if r["experiment"] == "training"])
    if args.out:
        save_results(args.out, meta, records)


if __name__ == "__main__":
    main()
