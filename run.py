"""Single entry point for the independent ASTER minimal prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="ASTER minimal: WBC detection and session-level AML-vs-control inference."
    )
    command.add_argument("--input", required=True, help="Session image or directory")
    command.add_argument("--session-id", required=True, help="Identifier written to result.json")
    command.add_argument("--output", required=True, help="Directory for generated results")
    command.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return command


def main() -> int:
    args = parser().parse_args()
    from backend import ASTERBackend, AnalysisRequest

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    try:
        backend = ASTERBackend(ROOT / "config" / "inference.yaml", device=args.device)
        result = backend.analyze(AnalysisRequest(args.input, args.session_id, output))
    except Exception as exc:
        failure = {
            "session_id": args.session_id,
            "status": "failed",
            "final_label": "no_decision",
            "warnings": [str(exc)],
            "timing_ms": {"total": round((perf_counter() - started) * 1000, 3)},
        }
        (output / "result.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(f"Analysis failed: {exc}")
        print(f"Result: {output / 'result.json'}")
        return 1

    print(f"Session: {result.session_id}")
    print(f"Status: {result.status}")
    print(f"Fields / WBC: {result.number_of_fields} / {result.number_of_detected_wbc}")
    if result.status == "completed":
        print(f"Decision: {result.label} (tier={result.tier}, final_label={result.final_label})")
    else:
        print(f"Decision: withheld ({result.status})")
    print(f"Result: {output / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
