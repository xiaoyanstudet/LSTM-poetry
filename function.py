import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_mpl_cache = Path.cwd() / "data" / "matplotlib_cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import classmodel


TANG_OFFSETS = list(range(0, 58000, 1000))
RAW_FOLDER = urllib.parse.quote("全唐诗")
RAW_URLS = [
    "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"
    f"{RAW_FOLDER}/poet.tang.{{offset}}.json",
    "https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/"
    f"{RAW_FOLDER}/poet.tang.{{offset}}.json",
    "https://fastly.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/"
    f"{RAW_FOLDER}/poet.tang.{{offset}}.json",
]


@dataclass
class TrainConfig:
    raw_dir: str = "tangshi"
    data_dir: str = "data"
    poem_type: int = 7
    batch_size: int = 64
    epochs: int = 60
    lr: float = 3e-4
    embedding_dim: int = 128
    hidden_dim: int = 600
    num_layers: int = 2
    dropout: float = 0.2
    val_ratio: float = 0.1
    eval_batches: int = 30
    sample_count: int = 8
    checkpoint_every_batches: int = 200
    log_interval: int = 60
    max_files: int = 0
    download_retries: int = 3
    strict_download: bool = False
    force_download: bool = False
    force_process: bool = False
    resume: bool = True
    device: str = "auto"
    seed: int = 42
    start_words: str = "湖光秋月两相和"
    temperature: float = 0.9
    top_k: int = 8


def choose_device(device_name="auto"):
    if device_name != "auto":
        return torch.device(device_name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _valid_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, list)
    except (OSError, json.JSONDecodeError):
        return False


def _download_file(urls, destination, retries=3):
    if isinstance(urls, str):
        urls = [urls]

    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    last_error = None

    for url in urls:
        for attempt in range(1, retries + 1):
            if temp_path.exists():
                temp_path.unlink()
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Apple Silicon) LSTM-poetry-downloader",
                        "Accept": "application/json,text/plain,*/*",
                    },
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    total = int(response.headers.get("Content-Length", "0") or 0)
                    downloaded = 0
                    with open(temp_path, "wb") as f:
                        while True:
                            chunk = response.read(1024 * 256)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded / total * 100
                                print(
                                    f"  {destination.name}: {downloaded / 1024 / 1024:.1f}MB "
                                    f"/ {total / 1024 / 1024:.1f}MB ({pct:.0f}%)",
                                    end="\r",
                                )
                temp_path.replace(destination)
                if not _valid_json(destination):
                    destination.unlink(missing_ok=True)
                    raise RuntimeError("downloaded file is not valid JSON")
                print(f"  {destination.name}: done{' ' * 30}")
                return
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
                last_error = exc
                if temp_path.exists():
                    temp_path.unlink()
                if attempt < retries:
                    wait_seconds = min(20, 2 ** (attempt - 1)) + random.random()
                    print(
                        f"  {destination.name}: retry {attempt}/{retries} failed "
                        f"({exc}); wait {wait_seconds:.1f}s"
                    )
                    time.sleep(wait_seconds)
                else:
                    print(f"  {destination.name}: mirror failed after {retries} attempts: {url}")

    raise RuntimeError(f"download failed: {destination.name}\nlast error: {last_error}")


def ensure_dataset(raw_dir, max_files=0, force_download=False, download_retries=3, strict_download=False):
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    custom_jsons = sorted(raw_path.glob("*.json"))
    expected_names = {f"poet.tang.{offset}.json" for offset in TANG_OFFSETS}
    has_custom_dataset = custom_jsons and not any(p.name in expected_names for p in custom_jsons)
    if has_custom_dataset and not force_download:
        print(f"Found {len(custom_jsons)} local JSON files in {raw_path}; skip auto-download.")
        return custom_jsons

    offsets = TANG_OFFSETS[:max_files] if max_files and max_files > 0 else TANG_OFFSETS
    print(f"Checking Tang poem JSON dataset in {raw_path} ...")
    failed = []
    for offset in offsets:
        destination = raw_path / f"poet.tang.{offset}.json"
        if destination.exists() and not force_download and _valid_json(destination):
            continue
        print(f"Downloading {destination.name}")
        urls = [template.format(offset=offset) for template in RAW_URLS]
        try:
            _download_file(urls, destination, retries=download_retries)
        except RuntimeError as exc:
            failed.append(destination.name)
            if strict_download:
                raise
            print(f"Warning: skip {destination.name} for now: {exc}")

    jsons = [path for path in sorted(raw_path.glob("*.json")) if _valid_json(path)]
    if not jsons:
        raise FileNotFoundError(f"No JSON files found in {raw_path}")
    if failed:
        print(
            f"Downloaded/available JSON files: {len(jsons)}. "
            f"Missing {len(failed)} files; training will use the available subset."
        )
    return jsons


def _normalize_poem(paragraphs):
    poem = "".join(paragraphs).strip()
    return (
        poem.replace(",", "，")
        .replace(".", "。")
        .replace("?", "？")
        .replace("!", "！")
        .replace(" ", "")
    )


def _poem_kind(poem):
    if len(poem) == 24 and poem[5] == "，" and poem[11] == "。" and poem[17] == "，" and poem[23] == "。":
        return 5
    if len(poem) == 32 and poem[7] == "，" and poem[15] == "。" and poem[23] == "，" and poem[31] == "。":
        return 7
    return 0


def process_1(path, path2, force=False, max_files=0):
    """Extract five-character and seven-character quatrains from JSON files."""

    output_dir = Path(path2)
    output_dir.mkdir(parents=True, exist_ok=True)
    poem_5_path = output_dir / "poem_5.txt"
    poem_7_path = output_dir / "poem_7.txt"

    if not force and poem_5_path.exists() and poem_7_path.exists():
        count_5 = _count_lines(poem_5_path)
        count_7 = _count_lines(poem_7_path)
        if count_5 > 0 or count_7 > 0:
            print(f"Using cached processed poems: five={count_5}, seven={count_7}")
            return {"poem_5": count_5, "poem_7": count_7}

    poem_5 = []
    poem_7 = []
    jsons = sorted(Path(path).glob("*.json"))
    if max_files and max_files > 0:
        jsons = jsons[:max_files]
    if not jsons:
        raise FileNotFoundError(f"No JSON files found in {path}")

    print(f"Processing {len(jsons)} JSON files ...")
    for js in jsons:
        with open(js, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skip invalid JSON: {js}")
                continue

        for item in data:
            paragraphs = item.get("paragraphs", [])
            if not paragraphs:
                continue
            poem = _normalize_poem(paragraphs)
            kind = _poem_kind(poem)
            tokenized = " ".join(list(poem))
            if kind == 5:
                poem_5.append(tokenized)
            elif kind == 7:
                poem_7.append(tokenized)

    _write_lines(poem_5_path, poem_5)
    _write_lines(poem_7_path, poem_7)
    print(f"Processed poems saved: five={len(poem_5)}, seven={len(poem_7)}")
    return {"poem_5": len(poem_5), "poem_7": len(poem_7)}


def _write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_poems(path):
    poems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            poems.append(line.split() if " " in line else list(line))
    return poems


def build_vocab(poems):
    counts = {}
    for poem in poems:
        for char in poem:
            counts[char] = counts.get(char, 0) + 1
    index_to_word = sorted(counts.keys(), key=lambda ch: (-counts[ch], ch))
    word_to_index = {word: idx for idx, word in enumerate(index_to_word)}
    return word_to_index, index_to_word


def encode_poems(poems, word_to_index):
    return [[word_to_index[char] for char in poem] for poem in poems]


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    path,
    model,
    optimizer,
    next_epoch,
    global_step,
    config,
    word_to_index,
    index_to_word,
    history,
):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "next_epoch": next_epoch,
        "global_step": global_step,
        "config": asdict(config),
        "word_to_index": word_to_index,
        "index_to_word": index_to_word,
        "history": history,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer, device, word_to_index, index_to_word):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("word_to_index") != word_to_index or checkpoint.get("index_to_word") != index_to_word:
        print("Checkpoint vocab differs from current data; start a fresh run.")
        return 0, 0, {"train_loss": [], "val_loss": [], "perplexity": [], "format_accuracy": [], "distinct_1": [], "distinct_2": []}

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    _optimizer_to_device(optimizer, device)
    history = checkpoint.get(
        "history",
        {"train_loss": [], "val_loss": [], "perplexity": [], "format_accuracy": [], "distinct_1": [], "distinct_2": []},
    )
    next_epoch = int(checkpoint.get("next_epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    print(f"Resumed from {path}: next_epoch={next_epoch + 1}, global_step={global_step}")
    return next_epoch, global_step, history


def evaluate(model, loader, device, max_batches=30):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs)
            flat_labels = labels.reshape(-1)
            loss = model.loss(outputs, flat_labels)
            total_loss += float(loss.detach().cpu()) * flat_labels.numel()
            total_tokens += flat_labels.numel()
    if total_tokens == 0:
        return {"val_loss": 0.0, "perplexity": 0.0}
    val_loss = total_loss / total_tokens
    return {"val_loss": val_loss, "perplexity": math.exp(min(val_loss, 20.0))}


def _sample_next_id(logits, temperature=0.9, top_k=8):
    logits = logits.detach().to("cpu")
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
        probs = torch.softmax(values, dim=-1)
        pick = torch.multinomial(probs, 1)
        return int(indices[pick].item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_poetry(
    model,
    index_to_word,
    word_to_index,
    poem_type=7,
    start_words="",
    device=None,
    temperature=0.9,
    top_k=8,
):
    model.eval()
    device = device or next(model.parameters()).device
    target_len = 24 if poem_type == 5 else 32
    forbidden = {"，", "。", "？", "！", "、", "；", "："}
    candidate_ids = [i for i, word in enumerate(index_to_word) if word not in forbidden]
    if not candidate_ids:
        candidate_ids = list(range(len(index_to_word)))

    result = []
    hidden = None
    input_id = None

    with torch.no_grad():
        for char in start_words:
            if char not in word_to_index:
                continue
            result.append(char)
            input_id = torch.tensor([[word_to_index[char]]], dtype=torch.long, device=device)
            _, hidden = model(input_id, hidden)
            if len(result) >= target_len:
                return "".join(result[:target_len])

        if input_id is None:
            first_id = random.choice(candidate_ids)
            result.append(index_to_word[first_id])
            input_id = torch.tensor([[first_id]], dtype=torch.long, device=device)
            _, hidden = model(input_id, hidden)

        while len(result) < target_len:
            output, hidden = model(input_id, hidden)
            next_id = _sample_next_id(output[-1], temperature=temperature, top_k=top_k)
            result.append(index_to_word[next_id])
            input_id = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return "".join(result[:target_len])


def format_accuracy(samples, poem_type):
    if not samples:
        return 0.0
    target_len = 24 if poem_type == 5 else 32
    if poem_type == 5:
        marks = {5: "，", 11: "。", 17: "，", 23: "。"}
    else:
        marks = {7: "，", 15: "。", 23: "，", 31: "。"}
    correct = 0
    for poem in samples:
        if len(poem) != target_len:
            continue
        if all(poem[pos] == mark for pos, mark in marks.items()):
            correct += 1
    return correct / len(samples)


def distinct_n(samples, n):
    total = 0
    uniq = set()
    for sample in samples:
        chars = [ch for ch in sample if ch.strip()]
        if len(chars) < n:
            continue
        for i in range(len(chars) - n + 1):
            gram = tuple(chars[i : i + n])
            uniq.add(gram)
            total += 1
    return len(uniq) / total if total else 0.0


def generation_metrics(model, index_to_word, word_to_index, config, device):
    samples = [
        generate_poetry(
            model,
            index_to_word,
            word_to_index,
            poem_type=config.poem_type,
            start_words=config.start_words if i == 0 else "",
            device=device,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        for i in range(config.sample_count)
    ]
    return {
        "samples": samples,
        "format_accuracy": format_accuracy(samples, config.poem_type),
        "distinct_1": distinct_n(samples, 1),
        "distinct_2": distinct_n(samples, 2),
    }


def save_samples(path, epoch, metrics):
    lines = [f"epoch={epoch}", f"format_accuracy={metrics['format_accuracy']:.4f}", f"distinct_1={metrics['distinct_1']:.4f}", f"distinct_2={metrics['distinct_2']:.4f}", ""]
    lines.extend(metrics["samples"])
    _write_lines(path, lines)


def plot_metrics(history, figure_path, poem_type):
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Poetry LSTM training metrics ({poem_type}-char quatrain)")

    axes[0, 0].plot(epochs, history.get("train_loss", []), marker="o", label="train loss")
    axes[0, 0].plot(epochs, history.get("val_loss", []), marker="o", label="val loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epochs, history.get("perplexity", []), marker="o", color="#d95f02")
    axes[0, 1].set_title("Validation perplexity")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(alpha=0.25)

    labels = ["format", "distinct-1", "distinct-2"]
    values = [
        history.get("format_accuracy", [0.0])[-1] if history.get("format_accuracy") else 0.0,
        history.get("distinct_1", [0.0])[-1] if history.get("distinct_1") else 0.0,
        history.get("distinct_2", [0.0])[-1] if history.get("distinct_2") else 0.0,
    ]
    axes[1, 0].bar(labels, values, color=["#1b9e77", "#7570b3", "#e7298a"])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Generation quality")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].axis("off")
    summary = "\n".join(
        [
            f"last train loss: {history.get('train_loss', [0])[-1]:.4f}" if history.get("train_loss") else "last train loss: n/a",
            f"last val loss: {history.get('val_loss', [0])[-1]:.4f}" if history.get("val_loss") else "last val loss: n/a",
            f"last perplexity: {history.get('perplexity', [0])[-1]:.2f}" if history.get("perplexity") else "last perplexity: n/a",
            f"format accuracy: {values[0]:.2%}",
            f"distinct-1: {values[1]:.2%}",
            f"distinct-2: {values[2]:.2%}",
        ]
    )
    axes[1, 1].text(0.02, 0.95, summary, va="top", fontsize=12)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def _empty_history():
    return {
        "train_loss": [],
        "val_loss": [],
        "perplexity": [],
        "format_accuracy": [],
        "distinct_1": [],
        "distinct_2": [],
    }


def run_training(config):
    set_seed(config.seed)
    data_dir = Path(config.data_dir)
    checkpoint_dir = data_dir / "checkpoints"
    figure_dir = data_dir / "figures"
    sample_dir = data_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    ensure_dataset(
        config.raw_dir,
        max_files=config.max_files,
        force_download=config.force_download,
        download_retries=config.download_retries,
        strict_download=config.strict_download,
    )
    process_1(config.raw_dir, config.data_dir, force=config.force_process, max_files=config.max_files)

    poem_path = data_dir / f"poem_{config.poem_type}.txt"
    poems = read_poems(poem_path)
    if not poems:
        raise RuntimeError(f"No {config.poem_type}-character poems found in {poem_path}")

    word_to_index, index_to_word = build_vocab(poems)
    encoded = encode_poems(poems, word_to_index)
    dataset = classmodel.MyDataset(encoded)

    val_size = max(1, int(len(dataset) * config.val_ratio)) if len(dataset) > 10 else 0
    train_size = len(dataset) - val_size
    if val_size:
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        if val_dataset is not None
        else None
    )

    device = choose_device(config.device)
    print(f"Device: {device}")
    print(f"Poems: train={train_size}, val={val_size}, vocab={len(index_to_word)}")

    model = classmodel.PoemLstm(
        vocab_size=len(index_to_word),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    latest_checkpoint = checkpoint_dir / f"poem_{config.poem_type}_latest.pt"
    start_epoch = 0
    global_step = 0
    history = _empty_history()
    if config.resume and latest_checkpoint.exists():
        start_epoch, global_step, history = load_checkpoint(
            latest_checkpoint, model, optimizer, device, word_to_index, index_to_word
        )

    try:
        for epoch in range(start_epoch, config.epochs):
            model.train()
            total_loss = 0.0
            total_tokens = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs, _ = model(inputs)
                flat_labels = labels.reshape(-1)
                loss = model.loss(outputs, flat_labels)
                loss.backward()
                optimizer.step()

                global_step += 1
                total_loss += float(loss.detach().cpu()) * flat_labels.numel()
                total_tokens += flat_labels.numel()

                if config.log_interval and batch_idx % config.log_interval == 0:
                    print(
                        f"epoch {epoch + 1}/{config.epochs} "
                        f"batch {batch_idx}/{len(train_loader)} "
                        f"loss={float(loss.detach().cpu()):.4f}"
                    )

                if config.checkpoint_every_batches and global_step % config.checkpoint_every_batches == 0:
                    save_checkpoint(
                        latest_checkpoint,
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        config,
                        word_to_index,
                        index_to_word,
                        history,
                    )

            train_loss = total_loss / max(1, total_tokens)
            val_stats = evaluate(model, val_loader, device, config.eval_batches) if val_loader else {"val_loss": 0.0, "perplexity": 0.0}
            gen_stats = generation_metrics(model, index_to_word, word_to_index, config, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_stats["val_loss"])
            history["perplexity"].append(val_stats["perplexity"])
            history["format_accuracy"].append(gen_stats["format_accuracy"])
            history["distinct_1"].append(gen_stats["distinct_1"])
            history["distinct_2"].append(gen_stats["distinct_2"])

            save_samples(sample_dir / f"poem_{config.poem_type}_epoch_{epoch + 1:03d}.txt", epoch + 1, gen_stats)
            save_checkpoint(
                latest_checkpoint,
                model,
                optimizer,
                epoch + 1,
                global_step,
                config,
                word_to_index,
                index_to_word,
                history,
            )
            save_checkpoint(
                checkpoint_dir / f"poem_{config.poem_type}_epoch_{epoch + 1:03d}.pt",
                model,
                optimizer,
                epoch + 1,
                global_step,
                config,
                word_to_index,
                index_to_word,
                history,
            )
            figure_path = plot_metrics(history, figure_dir / f"training_metrics_{config.poem_type}.png", config.poem_type)
            print(
                f"epoch {epoch + 1} done: train_loss={train_loss:.4f}, "
                f"val_loss={val_stats['val_loss']:.4f}, ppl={val_stats['perplexity']:.2f}, "
                f"format={gen_stats['format_accuracy']:.2%}, figure={figure_path}"
            )
            print("sample:", gen_stats["samples"][0])

    except KeyboardInterrupt:
        print("\nInterrupted. Saving latest checkpoint before exit ...")
        save_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            epoch,
            global_step,
            config,
            word_to_index,
            index_to_word,
            history,
        )
        raise

    print(f"Training finished. Latest checkpoint: {latest_checkpoint}")
    print(f"Metrics figure: {figure_dir / f'training_metrics_{config.poem_type}.png'}")
    return model


def train(path1="tangshi", path2="data", peo=7, batchsize=64):
    config = TrainConfig(raw_dir=path1, data_dir=path2, poem_type=peo, batch_size=batchsize)
    return run_training(config)


def gen_poetry(wordsize, index_2_word, type1, wvec, model):
    raise RuntimeError("gen_poetry now uses generate_poetry(model, index_to_word, word_to_index, ...).")
