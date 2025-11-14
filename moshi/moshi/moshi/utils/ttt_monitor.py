from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


class TTTInferenceMonitor:
    """Collects and visualizes fast-weight updates during inference.

    The monitor records every chunk update emitted by TTT-enabled MLPs and can
    export:
      * A JSON payload with per-layer events and aggregate statistics.
      * Optional PNG plots (delta norm, relative change, w_down norm per layer).
      * A short Markdown report summarizing the run.
    """

    def __init__(
        self,
        output_dir: str | Path,
        run_label: str | None = None,
        *,
        auto_plot: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_label = run_label or f"ttt_run_{int(time.time())}"
        self.events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.layer_metadata: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}
        self.auto_plot = auto_plot
        self.start_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_metadata(self, **kwargs: Any) -> None:
        self.metadata.update(kwargs)

    def register_layer(self, layer_name: str, metadata: dict[str, Any]) -> None:
        self.layer_metadata[layer_name] = metadata

    def record_event(self, layer_name: str, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload["elapsed_ms"] = (time.time() - self.start_time) * 1000.0
        self.events[layer_name].append(_to_serializable(payload))

    def summarize(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for layer, events in self.events.items():
            if not events:
                continue
            total_tokens = sum(float(ev.get("tokens", 0.0)) for ev in events)
            summary[layer] = {
                "updates": len(events),
                "total_tokens": total_tokens,
                "max_relative_delta": max(float(ev.get("relative_delta", 0.0)) for ev in events),
                "clip_events": sum(1 for ev in events if ev.get("clip_applied")),
                "final_post_norm": float(events[-1].get("post_norm", events[-1].get("pre_norm", 0.0))),
            }
        return summary

    def finalize(self) -> dict[str, Any]:
        payload = {
            "run_label": self.run_label,
            "metadata": self.metadata,
            "layers": self.layer_metadata,
            "events": {k: v for k, v in self.events.items()},
            "summary": self.summarize(),
        }
        json_path = self.output_dir / f"{self.run_label}_ttt_monitor.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        plot_paths: list[Path] = []
        if self.auto_plot:
            plot_paths = self._write_plots()
        report_path = self._write_report(json_path, plot_paths)
        return {
            "json": json_path,
            "plots": plot_paths,
            "report": report_path,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write_plots(self) -> list[Path]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - depends on optional dep
            print(f"[TTT Monitor] Matplotlib unavailable, skipping plots: {exc}")
            return []

        plot_paths: list[Path] = []
        for layer, events in self.events.items():
            if not events:
                continue
            updates = [ev.get("chunk_index", idx + 1) for idx, ev in enumerate(events)]
            delta_norms = [ev.get("delta_norm", 0.0) for ev in events]
            rel = [ev.get("relative_delta", 0.0) for ev in events]
            post_norms = [ev.get("post_norm", ev.get("pre_norm", 0.0)) for ev in events]

            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
            axes[0].plot(updates, delta_norms, color="tab:orange")
            axes[0].set_ylabel("Δ||W||₂")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(updates, rel, color="tab:purple")
            axes[1].set_ylabel("Relative Δ")
            axes[1].grid(True, alpha=0.3)
            axes[2].plot(updates, post_norms, color="tab:blue")
            axes[2].set_ylabel("||W||₂")
            axes[2].set_xlabel("Chunk update #")
            axes[2].grid(True, alpha=0.3)
            fig.suptitle(f"TTT dynamics for {layer}")
            fig.tight_layout()

            png_path = self.output_dir / f"{self.run_label}_{layer}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plot_paths.append(png_path)
            plt.close(fig)
        return plot_paths

    def _write_report(self, json_path: Path, plot_paths: list[Path]) -> Path:
        report_path = self.output_dir / f"{self.run_label}_ttt_report.md"
        lines = [f"# TTT Inference Report: {self.run_label}", ""]
        if self.metadata:
            lines.append("## Run metadata")
            for key, value in sorted(self.metadata.items()):
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        summary = self.summarize()
        if summary:
            lines.append("## Layer summaries")
            for layer, stats in summary.items():
                lines.append(f"### {layer}")
                for key, value in stats.items():
                    lines.append(f"- {key.replace('_', ' ').title()}: {value}")
                lines.append("")
        if plot_paths:
            lines.append("## Plots")
            for path in plot_paths:
                lines.append(f"![{path.name}]({path.name})")
            lines.append("")
        lines.append("## Artifacts")
        lines.append(f"- Raw events: `{json_path.name}`")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return report_path


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        try:
            return obj.item()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fallback path
            pass
    return obj