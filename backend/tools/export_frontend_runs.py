"""Build the small run index used by the static frontend."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT.parent


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_line(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| agent |"):
            return line
    return ""


def _algorithm_from_run(run_name: str, source: str | None) -> str:
    if source and run_name.endswith(f"_{source}"):
        head = run_name[: -(len(source) + 1)]
        return head.split("_", 1)[1] if "_" in head else ""
    return ""


def export_frontend_runs(
    results_dir: Path | None = None, frontend_dir: Path | None = None
) -> Path:
    """Write frontend/data/runs.json with available completed result folders."""
    results_dir = PROJECT_ROOT / "results" if results_dir is None else Path(results_dir)
    frontend_dir = APP_ROOT / "frontend" if frontend_dir is None else Path(frontend_dir)
    data_dir = frontend_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    if results_dir.exists():
        for run_dir in sorted(results_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            manifest = _read_json(run_dir / "manifest.json")
            if manifest.get("status") != "complete":
                continue
            test_dir = run_dir / "test"
            required = [
                run_dir / "ohlcv.csv",
                test_dir / "agent_trades.csv",
                test_dir / "equity.csv",
                test_dir / "metrics.json",
            ]
            if not all(path.exists() for path in required):
                continue
            runs.append(
                {
                    "id": run_dir.name,
                    "label": run_dir.name,
                    "mode": manifest.get("mode"),
                    "algorithm": manifest.get("algorithm")
                    or _algorithm_from_run(run_dir.name, manifest.get("source")),
                    "source": manifest.get("source"),
                    "symbol": manifest.get("symbol"),
                    "created_at_utc": manifest.get("created_at_utc"),
                    "best_episode": manifest.get("best_episode"),
                    "period": manifest.get("periods", {}).get("test", {}),
                    "summary": _summary_line(run_dir / "summary.md"),
                    "paths": {
                        "ohlcv": f"/results/{run_dir.name}/ohlcv.csv",
                        "trades": f"/results/{run_dir.name}/test/agent_trades.csv",
                        "equity": f"/results/{run_dir.name}/test/equity.csv",
                        "metrics": f"/results/{run_dir.name}/test/metrics.json",
                        "summary": f"/results/{run_dir.name}/summary.md",
                    },
                }
            )

    output_path = data_dir / "runs.json"
    output_path.write_text(
        json.dumps({"runs": runs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    print(export_frontend_runs())
