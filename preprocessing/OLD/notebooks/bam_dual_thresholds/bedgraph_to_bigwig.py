#!/usr/bin/env python3
from pathlib import Path
import pyBigWig

tracks_dir = Path("/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/fraction_tracks")
chrom_sizes_path = tracks_dir / "hg38.chrom.sizes"

chrom_sizes = []
with chrom_sizes_path.open() as fh:
    for line in fh:
        chrom, size = line.rstrip().split("\t")
        chrom_sizes.append((chrom, int(size)))

for bg in tracks_dir.glob("*.bedGraph"):
    bw_path = bg.with_suffix(".bw")
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader(chrom_sizes)

    chroms = []
    starts = []
    ends = []
    values = []

    with bg.open() as fh:
        for line in fh:
            chrom, start, end, value = line.rstrip().split("\t")
            chroms.append(chrom)
            starts.append(int(start))
            ends.append(int(end))
            values.append(float(value))

    bw.addEntries(chroms, starts, ends=ends, values=values)
    bw.close()
    print(f"wrote {bw_path}")
