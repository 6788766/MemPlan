from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = []
        for row in reader:
            if not isinstance(row, dict):
                continue
            rows.append({str(k): ("" if v is None else str(v)) for k, v in row.items()})
        return fieldnames, rows


def _merge_fieldnames(primary: Sequence[str], other: Sequence[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for name in list(primary) + list(other):
        name = str(name)
        if not name or name in seen:
            continue
        seen.add(name)
        merged.append(name)
    return merged


def _fill_missing(row: Dict[str, str], fieldnames: Sequence[str]) -> Dict[str, str]:
    out = dict(row)
    for key in fieldnames:
        out.setdefault(key, "")
    return out


def build_full_csv(
    *,
    train_path: Path,
    test_path: Path,
    out_path: Path,
    dedupe_key: str = "plan_id",
) -> None:
    train_fields, train_rows = _read_csv_rows(train_path)
    test_fields, test_rows = _read_csv_rows(test_path)
    out_fields = _merge_fieldnames(train_fields, test_fields)

    seen: set[str] = set()

    def _iter_all_rows() -> Iterable[Dict[str, str]]:
        for row in train_rows + test_rows:
            key = row.get(dedupe_key, "")
            if dedupe_key and key:
                if key in seen:
                    continue
                seen.add(key)
            yield _fill_missing(row, out_fields)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=out_fields, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(_iter_all_rows())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge WorkBench train/test CSVs into a single full.csv.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("artifacts/input/work/dataset/train.csv"),
        help="Input train.csv path.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("artifacts/input/work/dataset/test.csv"),
        help="Input test.csv path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/input/work/dataset/full.csv"),
        help="Output full.csv path.",
    )
    parser.add_argument(
        "--dedupe-key",
        type=str,
        default="plan_id",
        help="Optional unique key to dedupe rows (default: plan_id). Use empty string to disable dedupe.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train_path = args.train
    test_path = args.test
    out_path = args.out
    dedupe_key = str(args.dedupe_key or "")

    if not train_path.exists():
        raise SystemExit(f"Missing train CSV: {train_path}")
    if not test_path.exists():
        raise SystemExit(f"Missing test CSV: {test_path}")

    build_full_csv(train_path=train_path, test_path=test_path, out_path=out_path, dedupe_key=dedupe_key)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
