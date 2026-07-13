from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pysam

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_utils import (  # noqa: E402
    Region,
    collapse_windows,
    load_prediction_cache,
    load_regions,
    read_to_reference_map,
    save_prediction_cache,
    supported_enclosing_interval,
)
from reference_tracks import build_reference_track  # noqa: E402


def alignment(cigar: list[tuple[int, int]], query_length: int, reverse: bool = False):
    record = pysam.AlignedSegment()
    record.query_name = "read1"
    record.query_sequence = "A" * query_length
    record.flag = 16 if reverse else 0
    record.reference_id = 0
    record.reference_start = 100
    record.mapping_quality = 60
    record.cigar = cigar
    return record


class BenchmarkTests(unittest.TestCase):
    def test_crlf_region_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regions.tsv"
            path.write_bytes(
                b"region_id\tchrom\tstart\tend\tname\tsplit\r\n"
                b"4\tchr16\t4300000\t4400000\twrong_prefix\ttest\r\n"
            )
            regions = load_regions(path, split="test")
            self.assertEqual(regions[0].chrom, "chr16")
            self.assertEqual(regions[0].start, 4_300_000)

    def test_overlap_collapse(self):
        targets = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
        masks = np.ones_like(targets, dtype=bool)
        metadata = [
            {"window_start": "0", "window_length": "2"},
            {"window_start": "1", "window_length": "2"},
        ]
        collapsed, duplicates = collapse_windows(targets, masks, metadata, [0, 1])
        self.assertAlmostEqual(collapsed[1], 0.7)
        self.assertEqual(duplicates, 1)

    def test_supported_interval_expands_100kb_region(self):
        region = Region("4", "chr16", 4_300_000, 4_400_000, "region", "test")
        self.assertEqual(
            supported_enclosing_interval(region, 90_338_345),
            (4_284_464, 4_415_536),
        )

    def test_supported_interval_clamps_at_chromosome_start(self):
        region = Region("0", "chr16", 100, 100_100, "region", "test")
        self.assertEqual(
            supported_enclosing_interval(region, 90_338_345),
            (0, 131_072),
        )

    def test_supported_interval_clamps_at_chromosome_end(self):
        region = Region("0", "chr16", 90_238_345, 90_338_345, "region", "test")
        self.assertEqual(
            supported_enclosing_interval(region, 90_338_345),
            (90_207_273, 90_338_345),
        )

    def test_reverse_strand_mapping(self):
        record = alignment([(0, 4)], 4, reverse=True)
        mapped = read_to_reference_map(record, {0, 3})
        self.assertEqual(mapped, {3: 100, 0: 103})

    def test_indel_and_softclip_are_not_mapped(self):
        record = alignment([(4, 2), (0, 3), (1, 1), (0, 3)], 9)
        mapped = read_to_reference_map(record, set(range(9)))
        self.assertNotIn(0, mapped)
        self.assertNotIn(1, mapped)
        self.assertNotIn(5, mapped)
        self.assertEqual(len(mapped), 6)

    def test_cache_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npz"
            provenance = {
                "resolution": 128,
                "returned_interval": {"chrom": "chr16", "start": 0, "end": 256},
            }
            save_prediction_cache(
                path,
                np.ones((2, 1), dtype=np.float32),
                [{"name": "A549 H3K4me3"}],
                provenance,
            )
            values, metadata, loaded = load_prediction_cache(path)
            self.assertEqual(values.shape, (2, 1))
            self.assertEqual(metadata[0]["name"], "A549 H3K4me3")
            self.assertEqual(loaded["resolution"], 128)

    def test_synthetic_reference_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bam_path = root / "reads.bam"
            header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr16", "LN": 1000}]}
            with pysam.AlignmentFile(bam_path, "wb", header=header) as bam:
                record = alignment([(0, 4)], 4)
                record.reference_start = 0
                bam.write(record)
            pysam.index(str(bam_path))

            metadata_path = root / "metadata.tsv"
            fields = [
                "read_id", "chrom", "window_start", "window_length", "sample", "region_id"
            ]
            with open(metadata_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerow(
                    {
                        "read_id": "read1", "chrom": "chr16", "window_start": 0,
                        "window_length": 4, "sample": "merged_c1", "region_id": "0",
                    }
                )

            out = root / "track.tsv"
            build_reference_track(
                metadata_path,
                [Region("0", "chr16", 0, 256, "region", "test")],
                {"merged_c1": bam_path},
                lambda _: ({0: 0.2, 1: 0.8, 2: 0.6, 3: 0.4}, 0),
                out,
                root / "summary.json",
            )
            with open(out, "r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            first = next(
                row for row in rows if row["sample"] == "merged_c1" and row["bin_start"] == "0"
            )
            self.assertAlmostEqual(float(first["mean_signal"]), 0.5)
            self.assertEqual(int(first["observed_positions"]), 4)
            self.assertEqual(int(first["unique_reads"]), 1)


if __name__ == "__main__":
    unittest.main()
