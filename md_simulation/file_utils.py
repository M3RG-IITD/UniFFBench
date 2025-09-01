import json
import time
from datetime import datetime


class InProgressExperimentTracker:
    def __init__(self, track_file="runs_in_progress.json"):
        self.track_file = track_file

    def _load_runs(self):
        try:
            with open(self.track_file, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def _save_runs(self, runs):
        with open(self.track_file, "w") as file:
            json.dump(runs, file, indent=4)

    def start_run(self, index):
        runs = self._load_runs()
        timestamp = datetime.now().isoformat()
        runs[index] = timestamp
        self._save_runs(runs)
        print(f"Run {index} started at {timestamp}")

    def complete_run(self, index):
        runs = self._load_runs()
        if index in runs:
            del runs[str(index)]
            self._save_runs(runs)
            print(f"Run {index} completed and removed from tracking")
        else:
            print(f"Run {index} not found in tracking")
