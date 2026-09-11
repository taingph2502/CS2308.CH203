from __future__ import annotations

import argparse

from .data_prep import prepare_data
from .download import download_all
from .generation import generate_all
from .pissa import train_pissa
from .report import build_report
from .retrieval import build_index, evaluate_retrieval
from .scoring import score_generation
from .settings import load_config


COMMANDS = {"download": download_all, "prepare-data": prepare_data, "train": train_pissa, "build-index": build_index,
            "eval-retrieval": evaluate_retrieval, "generate": generate_all, "score": score_generation}

def main() -> None:
    parser = argparse.ArgumentParser(description="CS2308 PiSSA + RAG pipeline")
    parser.add_argument("command", choices=[*COMMANDS, "report", "run-all"])
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, paths = load_config(args.config)
    if args.command == "report": build_report(config, paths); return
    if args.command == "run-all":
        for command in COMMANDS.values(): command(config, paths, args.force)
        build_report(config, paths); return
    COMMANDS[args.command](config, paths, args.force)

if __name__ == "__main__": main()
