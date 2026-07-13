from __future__ import annotations

import csv
import os

from ..core import BaseVisualTask, FrameContext, GetTime, HookEvent


class RuntimeProfileTask(BaseVisualTask):
    def __init__(self, task_name: str, cfg: dict, mode: str, root_dir: str):
        super().__init__(task_name, cfg, mode, root_dir)
        names = cfg.get("names", ["Model_forward"])
        self.names = {str(v) for v in names} if isinstance(names, (list, tuple, set)) else {str(names)}
        self.save_txt = bool(cfg.get("save_txt", True))
        self.save_csv = bool(cfg.get("save_csv", True))

    def required_time_switches(self) -> set[str]:
        return set(self.names)

    def update(self, frame: FrameContext, hook_events: list[HookEvent]):
        del frame, hook_events
        return

    def close(self):
        super().close()
        if not self.enabled:
            return

        rows = GetTime.summary()
        rows = [row for row in rows if str(row.get("name", "")) in self.names]
        if not rows:
            return

        out_dir = os.path.join(self.root_dir, self.task_name)
        os.makedirs(out_dir, exist_ok=True)

        rows = sorted(rows, key=lambda x: float(x.get("avg_ms", 0.0)), reverse=True)

        if self.save_txt:
            self._save_txt(os.path.join(out_dir, "runtime_profile_summary.txt"), rows)
        if self.save_csv:
            self._save_csv(os.path.join(out_dir, "runtime_profile_summary.csv"), rows)

    @staticmethod
    def _save_txt(path: str, rows: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            f.write("Runtime Profile Summary\n")
            for row in rows:
                name = str(row.get("name", "unknown"))
                count = int(row.get("count", 0))
                avg_ms = float(row.get("avg_ms", 0.0))
                max_ms = float(row.get("max_ms", 0.0))
                total_s = float(row.get("total_s", 0.0))
                f.write(
                    f"- {name}: count={count}, avg_ms={avg_ms:.4f}, "
                    f"max_ms={max_ms:.4f}, total_s={total_s:.4f}\n"
                )

    @staticmethod
    def _save_csv(path: str, rows: list[dict]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "count", "avg_ms", "max_ms", "total_s"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "name": str(row.get("name", "unknown")),
                        "count": int(row.get("count", 0)),
                        "avg_ms": float(row.get("avg_ms", 0.0)),
                        "max_ms": float(row.get("max_ms", 0.0)),
                        "total_s": float(row.get("total_s", 0.0)),
                    }
                )
