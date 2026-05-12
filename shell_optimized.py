"""Shell.ai Hackathon entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.pipeline import ShellPipeline
from src.utils import configure_logging

logger = logging.getLogger("shell_hackathon.cli")


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description="Shell Hackathon fuel blend property predictor"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data"),
        help="Directory containing train.csv and test.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs"),
        help="Directory for submission CSV and pipeline pickle",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Hold-out fraction for validation",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable Optuna HPO (slow; ~40 trials per target)",
    )
    parser.add_argument(
        "--no-submission",
        action="store_true",
        help="Skip submission generation (validation only)",
    )
    parser.add_argument(
        "--load-pipeline",
        type=Path,
        default=None,
        help="Load a saved pipeline.pkl instead of retraining",
    )
    return parser


def _infer_from_pipeline(
    pipeline: ShellPipeline, data_path: Path, output_path: Path
) -> None:
    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")

    feature_cols = [
        col
        for col in train_df.columns
        if col not in ["ID", *[f"BlendProperty{i}" for i in range(1, 11)]]
    ]
    X_test = pipeline.feature_engineer.transform(test_df[feature_cols])
    X_test = X_test.replace([float("inf"), float("-inf")], float("nan"))
    X_test = X_test.fillna(X_test.median())

    if pipeline.feature_columns:
        X_test = X_test[pipeline.feature_columns]

    predictions = pipeline.predict(X_test)
    output_path.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"ID": test_df["ID"]})
    for col in predictions.columns:
        submission[col] = predictions[col]
    submission.to_csv(output_path / "submission.csv", index=False)


def main(argv: Optional[list[str]] = None) -> int:
    """Run training or inference based on CLI arguments."""
    args = build_parser().parse_args(argv)
    configure_logging()

    pipeline = ShellPipeline(
        data_path=args.data_path,
        output_path=args.output_path,
        val_split=args.val_split,
        tune=args.tune,
    )

    if args.load_pipeline:
        logger.info("Loading pipeline from %s", args.load_pipeline)
        pipeline.load(args.load_pipeline)
        _infer_from_pipeline(pipeline, args.data_path, args.output_path)
        return 0

    pipeline.run(generate_submission=not args.no_submission)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())