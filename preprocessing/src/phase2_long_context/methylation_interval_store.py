import csv
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class IntervalKey:
    chrom: str
    start: int
    end: int
    mark: str


class LongContextMethylationStore:
    """
    Interval-indexed methylation label store.
    Expects manifest rows with: chrom,start,end,mark,npz_path
    and npz payload containing methyl_ids.
    """

    def __init__(self, manifest_tsv: str):
        self.manifest_tsv = manifest_tsv
        self._index: Dict[IntervalKey, str] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        with open(self.manifest_tsv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = IntervalKey(
                    chrom=row["chrom"],
                    start=int(row["start"]),
                    end=int(row["end"]),
                    mark=row.get("mark", "unknown"),
                )
                self._index[key] = row["npz_path"]

    def load(self, chrom: str, start: int, end: int, mark: str) -> np.ndarray:
        key = IntervalKey(chrom=chrom, start=int(start), end=int(end), mark=mark)
        path = self._index.get(key)
        if path is None:
            return np.full((int(end) - int(start),), 2, dtype=np.uint8)
        payload = np.load(path, allow_pickle=False)
        return payload["methyl_ids"].astype(np.uint8)


def make_interval_loader(manifest_tsv: str, mark: str):
    """
    Returns a callable compatible with:
      load_methylation_for_interval(chrom, start, end)
    used by the Hyena dataset hook.
    """
    store = LongContextMethylationStore(manifest_tsv)

    def _loader(chrom: str, start: int, end: int):
        return store.load(chrom=chrom, start=start, end=end, mark=mark)

    return _loader
