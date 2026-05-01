import csv
from pathlib import Path

from config import OUTPUT_CSV

FIELDNAMES = ["status", "product_area", "response", "justification", "request_type"]


def init_output(path: Path = OUTPUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()


def write_row(row: dict, path: Path = OUTPUT_CSV) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
