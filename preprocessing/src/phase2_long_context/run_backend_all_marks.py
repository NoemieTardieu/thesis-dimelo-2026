import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_BAMS = {
    "h3k27ac": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam",
    "h3k27me3": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam",
    "h3k4me3": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Run long-context methylation backend build for all marks.")
    p.add_argument("--python", default="/data/leuven/383/vsc38330/.venv/bin/python")
    p.add_argument("--builder-script", default="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase2_long_context/build_methylation_interval_backend.py")
    p.add_argument("--intervals-tsv", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--target-base", choices=["A", "C"], default="C")
    p.add_argument("--min-mapq", type=int, default=20)
    p.add_argument("--ml-threshold", type=int, default=128)
    p.add_argument("--min-coverage", type=int, default=3)
    p.add_argument("--methylated-frac-thr", type=float, default=0.7)
    p.add_argument("--unmethylated-frac-thr", type=float, default=0.3)
    p.add_argument("--max-reads-per-interval", type=int, default=0)
    p.add_argument("--max-intervals", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--launch-background", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for mark, bam in DEFAULT_BAMS.items():
        out_dir = out_root / f"backend_{mark}_{args.target_base}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python,
            "-u",
            args.builder_script,
            "--bam", bam,
            "--intervals-tsv", args.intervals_tsv,
            "--out-dir", str(out_dir),
            "--mark", mark,
            "--target-base", args.target_base,
            "--min-mapq", str(args.min_mapq),
            "--ml-threshold", str(args.ml_threshold),
            "--min-coverage", str(args.min_coverage),
            "--methylated-frac-thr", str(args.methylated_frac_thr),
            "--unmethylated-frac-thr", str(args.unmethylated_frac_thr),
            "--max-reads-per-interval", str(args.max_reads_per_interval),
            "--max-intervals", str(args.max_intervals),
            "--progress-every", str(args.progress_every),
        ]
        if args.resume:
            cmd.append("--resume")

        if args.launch_background:
            log_path = out_dir / f"build_{mark}.log"
            with open(log_path, "a", encoding="utf-8") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
            print(f"Launched {mark} in background, pid={proc.pid}, log={log_path}")
        else:
            print("Running:", " ".join(cmd))
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"Build failed for {mark} with exit code {rc}", file=sys.stderr)
                sys.exit(rc)


if __name__ == "__main__":
    main()
