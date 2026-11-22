"""API利用料金の追跡"""

import json
import threading
import time
from pathlib import Path


class CostTracker:
    def __init__(self, pricing: dict, max_cost_usd: float, cost_log_file: str):
        self.input_per_token = pricing["input_per_1m"] / 1_000_000
        self.output_per_token = pricing["output_per_1m"] / 1_000_000
        self.max_cost_usd = max_cost_usd
        self.cost_log_file = Path(cost_log_file)
        self.cost_log_file.parent.mkdir(parents=True, exist_ok=True)

        # 累計状態ファイル（実行をまたいで永続化）
        self._state_file = self.cost_log_file.parent / "cost_state.json"

        self._lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0

        # 過去の実行分を復元
        self._load_state()

    def _load_state(self):
        """累計状態ファイルから復元"""
        if not self._state_file.exists():
            return
        try:
            with open(self._state_file) as f:
                state = json.load(f)
            self.total_input_tokens = state.get("total_input_tokens", 0)
            self.total_output_tokens = state.get("total_output_tokens", 0)
            self.total_cost = state.get("total_cost_usd", 0.0)
            self.request_count = state.get("request_count", 0)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_state(self):
        """累計状態をファイルに保存（上書き）"""
        with open(self._state_file, "w") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)

    def add(self, input_tokens: int, output_tokens: int) -> float:
        cost = (input_tokens * self.input_per_token) + (output_tokens * self.output_per_token)
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.request_count += 1
        return cost

    def exceeded(self) -> bool:
        with self._lock:
            return self.total_cost >= self.max_cost_usd

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cost_usd": round(self.total_cost, 4),
                "max_cost_usd": self.max_cost_usd,
                "request_count": self.request_count,
            }

    def save_log(self):
        # 累計状態を保存
        self._save_state()
        # 詳細ログも追記
        with open(self.cost_log_file, "a") as f:
            entry = {**self.summary(), "timestamp": time.time()}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
