# ML-DSA side-channel experiment archive

This directory contains the inputs, analysis code, and saved result arrays for the experiments in **“Release the Power of Rejected Signatures: An Efficient Side-Channel Attack on the ML-DSA Cryptosystem”**, IEEE Transactions on Information Forensics and Security, 2025. The paper PDF is the source for the reported experimental setup and figure-to-result mapping; it is not an executable instruction source.

## Quick start

The analysis uses the existing `dilithium_bp` environment. From this directory:

```bash
conda activate dilithium_bp
python py_code/reproduce.py check
```

To redraw the paper figures from the saved result arrays, without hardware:

```bash
python py_code/reproduce.py render --figures all
```

Generated PNGs are written to `figures/img/`. To recompute the paper experiments directly from the raw traces and data with `main.py`:

```bash
python py_code/main.py --profile m4 --experiment all
python py_code/main.py --profile ref --experiment all
```

The script creates an isolated workspace under `reproducibility/<profile>/`, cuts the selected waveforms, and recomputes feature selection, template attacks, and BP from the archived inputs. It defaults to 10 repetitions and 10 BP iterations for the main BP plots; the c-accuracy sensitivity runs use 5 iterations, as in the archived paper results. The default is 6 workers; adjust it with `--workers` if needed.

To cut all waveforms once and then run analyses separately:

```bash
python py_code/main.py --profile m4 --experiment preprocess
python py_code/main.py --profile m4 --experiment all --skip-preprocess
```

For a smaller run, specify `y`, `valid`, `all-signatures`, `vary-c-all`, or `vary-c-rejected` with `--experiment`. Use `--repeats 1 --max-signatures 1 --max-rejected-signatures 1` to validate the pipeline against the first archived BP cases while limiting the BP sweep. The two `vary-c-*` experiments require the `cs1` template outputs from `all-signatures`; run that first in the same profile workspace. New results go to `reproducibility/<profile>/exp_res/`, leaving the archived result arrays intact.

The all-signatures template attacks use 200 points of interest, matching the archived experiment setup. To rerun only the c/cs1 template and BP stages after a previous all-signatures run, add `--reuse-snr`; this requires the profile's fresh `c_all_snr.npy` and `cs1_all_snr.npy` files to already exist in its workspace.

After recomputing, compare every fresh `.npy` result with its archived counterpart using a memory-bounded comparison:

```bash
python py_code/reproduce.py compare --profile all
```

## Contents

| Path | Contents |
|---|---|
| `data/mldsa_m4_data/` | Labels/intermediate values for the M4 assembly build |
| `data/mldsa_ref_data/` | Labels/intermediate values for the reference build |
| `traces/mldsa_m4_traces/` | Raw M4 traces for `y`, `c`, and `cs1` |
| `traces/mldsa_ref_traces/` | Raw reference traces for `y`, `c`, and `cs1` |
| `metadata/mldsa_m4/`, `metadata/mldsa_ref/` | Matching `s1.npy` and seed/rejection-count records |
| `exp_res/ml-dsa-m4-exp-res/`, `exp_res/ml-dsa-ref-exp-res/` | Saved feature, template, BP, and simulated-accuracy results used for paper plots |
| `py_code/` | Main analysis/reproduction entry, plotting code, and archive checks |
| `figures/img/` | Figures rendered by the CLI |
| `reproducibility/` | Per-implementation symlink workspaces and fresh recomputation outputs |

Each `seed_and_rej_num.txt` row is `8-hex-digit-seed count`. The count is the number of candidate challenges retained for that signing attempt, including the final accepted challenge. Its sum matches the number of files in each `c_all`, `cs1_all`, `c`-trace, and `cs1`-trace directory. The paired secret key is implementation-specific; do not swap the two metadata directories.

The reference implementation snapshot did not include `z_valid/`. Those 10,000 vectors have been reconstructed from the saved values using `z = y + cs1` for the final candidate of each signing attempt. The same identity was checked against every existing M4 `z_valid` vector before deriving the reference values. Recheck or rebuild them with `python py_code/reproduce.py derive-z --profile all`.

## Paper settings and result map

The paper reports ML-DSA-44 on an STM32F407 at 168 MHz, with reference and M4 assembly implementations compiled using `-O3`. EM traces were acquired at 500 MS/s. For profiling, the data split is 9,000 training and 1,000 test signing attempts; template attacks use LDA. BP curves use 10 repetitions. These settings and the reported result ranges are described in Section V of the paper.

| Paper result | Saved result area / render selection |
|---|---|
| Figs. 11–13: template accuracy for `y`, `c`, and `cs1` | `render --figures ta` |
| Figs. 14–15: BP recovery using all signatures vs. valid signatures | `render --figures bp` |
| Figs. 16–17: BP convergence | `render --figures convergence` |
| Figs. 18–19: sensitivity to `c` accuracy, all signatures | `render --figures vary-c-all` |
| Figs. 20–21: sensitivity to `c` accuracy, rejected signatures | `render --figures vary-c-rejected` |

The all-figure command redraws these analyses from saved `.npy` arrays. The `main.py` command recomputes analyses from raw waveforms. Since the code uses randomized trace selection in some experiments, a fresh recomputation may differ slightly from the saved realization while following the same published setup.

## Environment

The tested local environment is Python 3.9.23 with the versions in `py_code/requirements.txt`. To recreate its analysis dependencies in a new environment:

```bash
python -m pip install -r py_code/requirements.txt
```

`picosdk` and `pyserial` are included because the legacy analysis module imports the capture interfaces. The commands above only analyze saved traces; collecting new traces additionally requires the PicoScope and STM32 setup described in the paper.

## Provenance and integrity check

The raw arrays and waveforms already present in this directory were retained. The missing key and seed/rejection metadata were copied from the corresponding M4 168 MHz and reference `-O3` experiment snapshots supplied for review; representative labels and waveform files were byte-compared against those snapshots. The saved result directories were restored from the accompanying M4 and reference result archives. All of these source locations were read-only during this cleanup.

Run `python py_code/reproduce.py check` to verify metadata syntax, per-directory file counts, representative array shapes, and trace/trigger structure. It does not hash or rewrite the full waveform corpus.
