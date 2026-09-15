"""Everything a run must leave behind: to debug it, to report it, to deploy it.

A Colab session dies without warning. Anything not written to Drive as it happens is lost,
including the traceback that would explain why. This module makes one directory per run
under Drive and writes into it continuously.

    run = RunDir.create(Path("/content/drive/MyDrive/aster_block2"), repo=REPO)
    with run.section("04_cell_head"):
        for epoch in range(epochs):
            ...
            run.metric("cell_head", epoch=epoch, loss=loss, val_acc=acc)
            run.checkpoint("cell_head", {"encoder": ..., "cell_head": ...}, epoch=epoch)
    run.artifact("results/caitomorph_409.csv", frame.to_csv())
    run.finalise()

Layout
    runs/<run_id>/
      manifest.json        versions, seeds, GPU, timings, pre-registration digests
      env/                 pip freeze, nvidia-smi, python version
      logs/<section>.log   stdout+stderr of each section, flushed line by line
      metrics/<name>.jsonl one line per epoch - training curves survive a disconnect
      checkpoints/         per-epoch and final weights
      results/             every CSV/JSON the paper quotes
      figures/
      bundle/              exactly what integration/PATCH.md needs on the Jetson
      REPORT_INPUTS.md     which artifact feeds which section of the paper
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUBDIRS = ("env", "logs", "metrics", "checkpoints", "results", "figures", "bundle")


class _Tee(io.TextIOBase):
    """Write to the notebook AND to a file, flushing every line."""

    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.handle.write(text)
        self.handle.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.handle.flush()


@dataclass
class RunDir:
    root: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    # -- creation -----------------------------------------------------------
    @classmethod
    def create(cls, drive: Path, repo: Path | None = None, run_id: str | None = None) -> "RunDir":
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        root = Path(drive) / "runs" / run_id
        for name in SUBDIRS:
            (root / name).mkdir(parents=True, exist_ok=True)
        run = cls(root=root)
        run.manifest = {
            "run_id": run_id,
            "started": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sections": {},
        }
        run._capture_environment(repo)
        run._write_manifest()
        print(f"[run] {root}")
        return run

    def _capture_environment(self, repo: Path | None) -> None:
        env = self.root / "env"
        for name, command in (("pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"]),
                              ("nvidia_smi.txt", ["nvidia-smi"])):
            try:
                out = subprocess.run(command, capture_output=True, text=True, timeout=120)
                (env / name).write_text(out.stdout or out.stderr)
            except Exception as exc:                       # noqa: BLE001 - best effort
                (env / name).write_text(f"unavailable: {exc}\n")
        try:
            import torch
            self.manifest["torch"] = torch.__version__
            self.manifest["cuda"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                self.manifest["gpu"] = torch.cuda.get_device_name(0)
        except Exception:                                   # noqa: BLE001
            pass
        # the frozen pre-registration travels with the run: a result is only readable
        # next to the rules that produced it
        if repo is not None:
            digests = {}
            for name in ("PREREGISTRATION.md", "src/aster_block2/decision_grid.yaml",
                         "src/aster_block2/proportion_test.py"):
                path = Path(repo) / name
                if path.exists():
                    digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    (self.root / "env" / Path(name).name).write_bytes(path.read_bytes())
            self.manifest["preregistration_sha256"] = digests

    def _write_manifest(self) -> None:
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    # -- sections -----------------------------------------------------------
    @contextlib.contextmanager
    def section(self, name: str):
        """Tee stdout/stderr into logs/<name>.log and record timing and any traceback."""
        path = self.root / "logs" / f"{name}.log"
        started = time.time()
        record = {"started": datetime.now(timezone.utc).isoformat(), "status": "running"}
        self.manifest["sections"][name] = record
        self._write_manifest()
        handle = path.open("a", encoding="utf-8")
        handle.write(f"\n===== {name} @ {record['started']} =====\n")
        try:
            with contextlib.redirect_stdout(_Tee(sys.stdout, handle)), \
                 contextlib.redirect_stderr(_Tee(sys.stderr, handle)):
                yield self
            record["status"] = "ok"
        except BaseException as exc:                        # noqa: BLE001 - record then re-raise
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            handle.write("\n" + traceback.format_exc())
            (self.root / "logs" / f"{name}.traceback.txt").write_text(traceback.format_exc())
            raise
        finally:
            record["seconds"] = round(time.time() - started, 1)
            record["ended"] = datetime.now(timezone.utc).isoformat()
            handle.flush()
            handle.close()
            self._write_manifest()

    # -- alerts -------------------------------------------------------------
    def alert(self, message: str, detail: str = "", severity: str = "WARNING") -> None:
        """Anything the morning review must not miss, in ONE file.

        An unattended `Run all` scrolls every warning past an empty chair. Falsifiers,
        unquantifiable criteria and uncaught exceptions all land in ALERTS.md so the first
        thing to read is one page, not 46 cells of output.
        """
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        block = f"\n## {severity} — {stamp}\n\n{message}\n"
        if detail:
            block += f"\n```\n{detail.strip()}\n```\n"
        path = self.root / "ALERTS.md"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(block)
        print(f"[{severity}] {message}")

    def install_exception_hook(self) -> None:
        """Capture any uncaught exception from any cell into the run directory.

        `Run all` stops at the first error; without this the traceback lives only in the
        browser, and a disconnected session takes it away.
        """
        try:
            shell = get_ipython()          # type: ignore[name-defined]  # noqa: F821
        except NameError:
            return
        if shell is None:
            return

        def handler(shell_self, etype, value, tb, tb_offset=None):
            self.alert(f"uncaught exception: {etype.__name__}: {value}",
                       detail="".join(traceback.format_exception(etype, value, tb)),
                       severity="FAILED")
            return shell_self.showtraceback((etype, value, tb), tb_offset=tb_offset)

        shell.set_custom_exc((Exception,), handler)
        print("[run] exception hook installed - any error is written to ALERTS.md")

    # -- artifacts ----------------------------------------------------------
    def metric(self, name: str, **values: Any) -> None:
        """One JSON line per epoch. The training curve survives a disconnect."""
        line = {"t": datetime.now(timezone.utc).isoformat(), **values}
        with (self.root / "metrics" / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")

    def checkpoint(self, name: str, state: dict, epoch: int | None = None,
                   keep_last: int = 2) -> Path:
        """Save per-epoch, keep the last few, and always refresh <name>_final.pt.

        Per-epoch matters: a crash at epoch 9 of 12 otherwise costs the whole run.
        """
        import torch

        target = self.root / "checkpoints"
        final = target / f"{name}_final.pt"
        torch.save(state, final)
        if epoch is not None:
            stamped = target / f"{name}_epoch{epoch:03d}.pt"
            torch.save(state, stamped)
            old = sorted(target.glob(f"{name}_epoch*.pt"))[:-keep_last]
            for path in old:
                path.unlink(missing_ok=True)
        return final

    def result(self, relative: str, payload: Any) -> Path:
        """Write a CSV string, a DataFrame or a JSON-able object under results/."""
        path = self.root / "results" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(payload, "to_csv") and relative.endswith(".csv"):
            payload.to_csv(path, index=False)
        elif isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def figure(self, name: str, fig) -> Path:
        path = self.root / "figures" / name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        return path

    def bundle(self, source: Path, name: str | None = None) -> Path:
        import shutil

        target = self.root / "bundle" / (name or Path(source).name)
        shutil.copy2(source, target)
        return target

    # -- closing ------------------------------------------------------------
    def finalise(self, notes: str = "") -> Path:
        self.manifest["ended"] = datetime.now(timezone.utc).isoformat()
        failed = [k for k, v in self.manifest["sections"].items() if v.get("status") == "failed"]
        self.manifest["failed_sections"] = failed
        self._write_manifest()

        checksums = []
        for path in sorted((self.root / "bundle").rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS.txt":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                checksums.append(f"{digest}  {path.relative_to(self.root / 'bundle')}")
        (self.root / "bundle" / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")

        alerts = (self.root / "ALERTS.md")
        alert_count = alerts.read_text().count("\n## ") if alerts.exists() else 0
        index = [
            f"# Run {self.manifest['run_id']} — what feeds what",
            "",
            f"**{alert_count} alert(s)** — read `ALERTS.md` first."
            if alert_count else "No alerts raised.",
            "",
            f"Started {self.manifest['started']}, ended {self.manifest['ended']}.",
            f"Failed sections: {', '.join(failed) if failed else 'none'}.",
            "",
            "| Artifact | Feeds |",
            "|---|---|",
            "| `results/cell_head_per_class.csv` | per-class precision/recall — paper, cell-head table |",
            "| `results/differential_spearman.json` | differential validated vs AML-MLL — §6.2, and falsifier 6.5 |",
            "| `results/caitomorph_409_predictions.csv` | the primary test, one row per patient |",
            "| `results/caitomorph_409_confusion.csv` | confusion matrix by `diagnosis_fine` — main results table |",
            "| `results/caitomorph_tierR_confusion.csv` | secondary analysis at reference count — §3.4 |",
            "| `results/x40_stress.csv` | ×40 stress test — the add-on's own acquisitions |",
            "| `results/reference_intervals.json` | each `[REF]` threshold: floor, percentile, which was adopted |",
            "| `results/resolved_thresholds.json` | every `[FIT]` value and the split it was fitted on |",
            "| `results/sensitivity*.csv` | the `[OPS]` sensitivity analyses |",
            "| `metrics/*.jsonl` | training curves |",
            "| `logs/*.log`, `logs/*.traceback.txt` | what actually happened, including failures |",
            "| `env/`, `manifest.json` | reproducibility: versions, GPU, seeds, pre-registration digests |",
            "| `bundle/` + `bundle/SHA256SUMS.txt` | **integration/PATCH.md** — what goes on the Jetson |",
            "",
            notes,
        ]
        path = self.root / "REPORT_INPUTS.md"
        path.write_text("\n".join(index) + "\n", encoding="utf-8")
        print(f"[run] finalised: {self.root}")
        if failed:
            print(f"[run] FAILED SECTIONS: {failed} - see logs/*.traceback.txt")
        return path
