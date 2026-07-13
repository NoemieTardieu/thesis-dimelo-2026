# Modkit Visualization

This folder collects reproducible figure scripts and commands for the `modkit`-based workflow.

## Outputs
- `read_level/`: figures from `modkit extract calls` per-read summaries.
- `profiles/`: average profiles from bigWig tracks over one BED region set.
- `stratifications/`: one track plotted over multiple BED region sets.

## Core scripts
- `plot_modkit_read_level_figures.py`
- `plot_bigwig_average_profile.py`
- `plot_bigwig_profile_by_region_sets.py`

## Run all main figures
```bash
source /data/leuven/383/vsc38330/venvs/thesis_env/bin/activate
bash /data/leuven/383/vsc38330/thesis_dimelo/modkit_visualization/run_modkit_visualizations.sh
```
