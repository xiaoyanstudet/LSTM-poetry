import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import function


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM Tang poem generator.")
    parser.add_argument("--raw-dir", default="tangshi", help="folder for downloaded/raw Tang poem JSON files")
    parser.add_argument("--data-dir", default="data", help="folder for processed data, checkpoints, figures")
    parser.add_argument("--poem-type", type=int, choices=[5, 7], default=7, help="train five-character or seven-character quatrains")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--checkpoint-every-batches", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--no-simplify-text", action="store_true", help="keep original traditional/variant characters")
    parser.add_argument("--no-format-constraint", action="store_true", help="let generation sample punctuation freely")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-words", default="湖光秋月两相和")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=0, help="download/process first N JSON files; 0 means full dataset")
    parser.add_argument("--download-retries", type=int, default=3, help="retry count for each download mirror")
    parser.add_argument("--strict-download", action="store_true", help="fail if any dataset file cannot be downloaded")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-process", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="start from scratch instead of latest checkpoint")
    parser.add_argument("--quick-test", action="store_true", help="1 epoch on 2 JSON files, useful for checking the pipeline")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick_test:
        args.epochs = 1
        args.max_files = args.max_files or 2
        args.batch_size = min(args.batch_size, 32)

    config = function.TrainConfig(
        raw_dir=args.raw_dir,
        data_dir=args.data_dir,
        poem_type=args.poem_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        eval_batches=args.eval_batches,
        sample_count=args.sample_count,
        checkpoint_every_batches=args.checkpoint_every_batches,
        log_interval=args.log_interval,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        simplify_text=not args.no_simplify_text,
        constrain_format=not args.no_format_constraint,
        max_files=args.max_files,
        download_retries=args.download_retries,
        strict_download=args.strict_download,
        force_download=args.force_download,
        force_process=args.force_process,
        resume=not args.no_resume,
        device=args.device,
        seed=args.seed,
        start_words=args.start_words,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    function.run_training(config)


if __name__ == "__main__":
    main()
