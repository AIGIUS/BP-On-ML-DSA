#!/usr/bin/env python3
"""Prepare, validate, run, and render the ML-DSA side-channel experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

EXP_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = EXP_ROOT / "py_code"
WORK_ROOT = EXP_ROOT / "reproducibility"
FIGURE_ROOT = EXP_ROOT / "figures"

PROFILES = {
    "m4": {
        "label": "ML-DSA M4 assembly implementation",
        "data": "mldsa_m4_data",
        "traces": "mldsa_m4_traces",
        "metadata": "mldsa_m4",
        "expected_candidates": 43030,
    },
    "ref": {
        "label": "ML-DSA reference implementation",
        "data": "mldsa_ref_data",
        "traces": "mldsa_ref_traces",
        "metadata": "mldsa_ref",
        "expected_candidates": 44294,
    },
}

CODE_FILES = ("main.py", "Template_Attack.py", "utils.py")


def _link(src: Path, dest: Path) -> None:
    """Create a relative symlink, while refusing to replace real files."""
    src = src.resolve(strict=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        if dest.resolve(strict=True) != src:
            raise RuntimeError(f"Existing link points elsewhere: {dest}")
        return
    if dest.exists():
        raise RuntimeError(f"Refusing to replace existing path: {dest}")
    dest.symlink_to(os.path.relpath(src, dest.parent), target_is_directory=src.is_dir())


def _workspace(profile: str) -> Path:
    return WORK_ROOT / profile


def setup_workspace(profile: str) -> Path:
    """Create an isolated legacy path layout using links to the archived data."""
    spec = PROFILES[profile]
    root = _workspace(profile)
    (root / "py_code").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "traces").mkdir(parents=True, exist_ok=True)
    (root / "exp_res").mkdir(parents=True, exist_ok=True)

    for name in CODE_FILES:
        _link(CODE_ROOT / name, root / "py_code" / name)

    metadata_dir = EXP_ROOT / "metadata" / spec["metadata"]
    for source, dest_name in (
        (metadata_dir / "s1.npy", "s1.npy"),
        (metadata_dir / "seed_and_rej_num.txt", "seed and rej num.txt"),
    ):
        _link(source, root / "py_code" / dest_name)

    data_root = EXP_ROOT / "data" / spec["data"]
    for name in ("c_all", "cs1_all", "y_valid", "z_valid"):
        _link(data_root / name, root / "data" / name)

    trace_root = EXP_ROOT / "traces" / spec["traces"]
    for name in ("c", "cs1", "y"):
        _link(trace_root / name, root / "traces" / name)

    print(f"Prepared {profile} workspace: {root}")
    return root


def _count_files(path: Path, suffix: str) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.name.endswith(suffix))


def check_profile(profile: str) -> bool:
    """Check inventory, seed counts, secret shape, and representative arrays."""
    spec = PROFILES[profile]
    data_root = EXP_ROOT / "data" / spec["data"]
    trace_root = EXP_ROOT / "traces" / spec["traces"]
    metadata_root = EXP_ROOT / "metadata" / spec["metadata"]
    errors: list[str] = []

    seed_file = metadata_root / "seed_and_rej_num.txt"
    try:
        rows = [line.split() for line in seed_file.read_text().splitlines()]
        if len(rows) != 10_000 or any(len(row) != 2 for row in rows):
            errors.append(f"{profile}: expected 10,000 seed/count rows in {seed_file}")
            rejection_counts = []
        else:
            rejection_counts = [int(row[1]) for row in rows]
            if any(len(row[0]) != 8 or any(c not in "0123456789abcdefABCDEF" for c in row[0]) for row in rows):
                errors.append(f"{profile}: malformed 4-byte seed in {seed_file}")
            if any(count < 1 for count in rejection_counts):
                errors.append(f"{profile}: rejection counts must be positive")
    except (OSError, ValueError) as exc:
        errors.append(f"{profile}: cannot read seed/count metadata: {exc}")
        rejection_counts = []

    candidates = sum(rejection_counts)
    if candidates != spec["expected_candidates"]:
        errors.append(
            f"{profile}: seed metadata sums to {candidates} candidates; "
            f"expected {spec['expected_candidates']}"
        )

    expected = {
        data_root / "c_all": (candidates, ".npy"),
        data_root / "cs1_all": (candidates, ".npy"),
        data_root / "y_valid": (10_000, ".npy"),
        data_root / "z_valid": (10_000, ".npy"),
        trace_root / "c": (candidates, ".npz"),
        trace_root / "cs1": (candidates, ".npz"),
        trace_root / "y": (10_000, ".npz"),
    }
    for folder, (expected_count, suffix) in expected.items():
        try:
            actual = _count_files(folder, suffix)
        except OSError as exc:
            errors.append(f"{profile}: cannot read {folder}: {exc}")
            continue
        if actual != expected_count:
            errors.append(f"{profile}: {folder} has {actual} {suffix} files; expected {expected_count}")

    candidate_pairs = [
        (sign_idx, candidate_idx)
        for sign_idx, count in enumerate(rejection_counts)
        for candidate_idx in range(count)
    ]
    named_expectations = {
        data_root / "c_all": {f"c{sign_idx:06d}rej{candidate_idx:06d}.npy" for sign_idx, candidate_idx in candidate_pairs},
        data_root / "cs1_all": {f"cs{sign_idx:06d}rej{candidate_idx:06d}.npy" for sign_idx, candidate_idx in candidate_pairs},
        data_root / "y_valid": {f"y{sign_idx:06d}.npy" for sign_idx in range(10_000)},
        data_root / "z_valid": {f"z{sign_idx:06d}.npy" for sign_idx in range(10_000)},
        trace_root / "c": {f"trace{sign_idx:06d}rej{candidate_idx:06d}.npz" for sign_idx, candidate_idx in candidate_pairs},
        trace_root / "cs1": {f"trace{sign_idx:06d}rej{candidate_idx:06d}.npz" for sign_idx, candidate_idx in candidate_pairs},
        trace_root / "y": {f"trace{sign_idx:06d}.npz" for sign_idx in range(10_000)},
    }
    for folder, expected_names in named_expectations.items():
        try:
            actual_names = {item.name for item in folder.iterdir() if item.is_file() and item.name.endswith((".npy", ".npz"))}
        except OSError:
            continue
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)[:3]
            extra = sorted(actual_names - expected_names)[:3]
            errors.append(f"{profile}: filename mismatch in {folder}; missing={missing}, extra={extra}")

    try:
        import numpy as np

        s1 = np.load(metadata_root / "s1.npy", allow_pickle=False)
        if s1.shape != (4, 256) or not np.isin(s1, [-2, -1, 0, 1, 2]).all():
            errors.append(f"{profile}: s1.npy has unexpected shape or values: {s1.shape}")
        for sample in (
            data_root / "y_valid" / "y000000.npy",
            data_root / "c_all" / "c000000rej000000.npy",
            data_root / "cs1_all" / "cs000000rej000000.npy",
        ):
            values = np.load(sample, allow_pickle=False)
            if values.shape != (256,):
                errors.append(f"{profile}: unexpected array shape in {sample}: {values.shape}")
        for sample in (
            trace_root / "y" / "trace000000.npz",
            trace_root / "c" / "trace000000rej000000.npz",
            trace_root / "cs1" / "trace000000rej000000.npz",
        ):
            with np.load(sample, allow_pickle=False) as archive:
                if not {"trace", "trigger"}.issubset(archive.files):
                    errors.append(f"{profile}: trace archive lacks trace/trigger: {sample}")
                elif archive["trace"].shape != archive["trigger"].shape:
                    errors.append(f"{profile}: trace and trigger lengths differ: {sample}")
    except (ImportError, OSError, ValueError) as exc:
        errors.append(f"{profile}: sample-array check failed: {exc}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return False
    print(
        f"OK {profile}: 10,000 attempts, {candidates:,} candidate signatures, "
        "labels, raw traces, trigger arrays, and s1 metadata are present."
    )
    return True


def derive_z(profile: str) -> None:
    """Rebuild z=y+cs1 from the saved valid y and final candidate cs1 arrays."""
    import numpy as np

    spec = PROFILES[profile]
    data_root = EXP_ROOT / "data" / spec["data"]
    metadata_root = EXP_ROOT / "metadata" / spec["metadata"]
    rows = [line.split() for line in (metadata_root / "seed_and_rej_num.txt").read_text().splitlines()]
    if len(rows) != 10_000 or any(len(row) != 2 for row in rows):
        raise RuntimeError(f"Expected 10,000 seed/count rows for {profile}")
    output_dir = data_root / "z_valid"
    output_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for sign_idx, row in enumerate(rows):
        candidate_count = int(row[1])
        y = np.load(data_root / "y_valid" / f"y{sign_idx:06d}.npy", allow_pickle=False)
        cs1 = np.load(
            data_root / "cs1_all" / f"cs{sign_idx:06d}rej{candidate_count - 1:06d}.npy",
            allow_pickle=False,
        )
        z = y + cs1
        dest = output_dir / f"z{sign_idx:06d}.npy"
        if dest.exists():
            existing = np.load(dest, allow_pickle=False)
            if not np.array_equal(existing, z):
                raise RuntimeError(f"Existing z disagrees with y+cs1: {dest}")
        else:
            np.save(dest, z, allow_pickle=False)
            created += 1
        if (sign_idx + 1) % 1000 == 0:
            print(f"{profile}: checked {sign_idx + 1}/10000 valid z vectors")
    print(f"{profile}: verified/derived 10,000 z vectors ({created} new files) in {output_dir}")


def compare_profile(profile: str) -> bool:
    """Compare fresh .npy outputs with the archived result arrays without loading them all into RAM."""
    import numpy as np

    fresh_root = WORK_ROOT / profile / "exp_res"
    archive_root = EXP_ROOT / "exp_res" / f"ml-dsa-{profile}-exp-res"
    if not fresh_root.is_dir():
        print(f"No fresh result directory found: {fresh_root}")
        return False
    if not archive_root.is_dir():
        print(f"No archived result directory found: {archive_root}")
        return False

    # Vary-c workspaces contain symlinks to cs1 template arrays generated by
    # all-signatures. They are inputs, not fresh outputs to compare a second time.
    variation_dirs = {
        "c_all_cs1_all_bp_vary_c_acc",
        "c_rej_cs1_rej_bp_vary_c_acc",
    }
    housekeeping_names = {"train_sign_idx.npy", "test_sign_idx.npy"}
    files = sorted(
        path
        for path in fresh_root.rglob("*.npy")
        if not path.is_symlink()
        and not (
            path.parent.name in variation_dirs and path.name in housekeeping_names
        )
    )
    exact = 0
    close = 0
    different = []
    unpaired = []
    for fresh_path in files:
        archived_path = archive_root / fresh_path.relative_to(fresh_root)
        if not archived_path.is_file():
            unpaired.append(fresh_path.relative_to(fresh_root))
            continue
        fresh = np.load(fresh_path, mmap_mode="r", allow_pickle=False)
        archived = np.load(archived_path, mmap_mode="r", allow_pickle=False)
        if fresh.shape != archived.shape:
            different.append((fresh_path.relative_to(fresh_root), f"shape {fresh.shape} != {archived.shape}"))
            continue

        exact_match = True
        flat_fresh = fresh.reshape(-1)
        flat_archived = archived.reshape(-1)
        for start in range(0, fresh.size, 1_000_000):
            end = min(start + 1_000_000, fresh.size)
            if not np.array_equal(flat_fresh[start:end], flat_archived[start:end]):
                exact_match = False
                break
        if exact_match:
            exact += 1
            continue

        if not (np.issubdtype(fresh.dtype, np.number) and np.issubdtype(archived.dtype, np.number)):
            different.append((fresh_path.relative_to(fresh_root), "non-numeric values differ"))
            continue
        close_match = True
        max_abs_diff = 0.0
        for start in range(0, fresh.size, 1_000_000):
            end = min(start + 1_000_000, fresh.size)
            a = np.asarray(flat_fresh[start:end], dtype=np.float64)
            b = np.asarray(flat_archived[start:end], dtype=np.float64)
            delta = np.abs(a - b)
            finite_delta = delta[np.isfinite(delta)]
            if finite_delta.size:
                max_abs_diff = max(max_abs_diff, float(finite_delta.max()))
            if not np.allclose(a, b, rtol=1e-7, atol=1e-10, equal_nan=True):
                close_match = False
                break
        if close_match:
            close += 1
        else:
            different.append((fresh_path.relative_to(fresh_root), f"max abs diff >= {max_abs_diff:.3g}"))

    print(
        f"{profile}: {len(files)} fresh .npy arrays; {exact} exact, {close} numerically close, "
        f"{len(different)} different, {len(unpaired)} without archived counterparts."
    )
    for path, reason in different[:20]:
        print(f"DIFFERENT {path}: {reason}")
    for path in unpaired[:20]:
        print(f"NO ARCHIVE MATCH {path}")
    return bool(files) and not different and not unpaired


def render(figures: list[str]) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    (FIGURE_ROOT / "img").mkdir(parents=True, exist_ok=True)

    for profile in PROFILES:
        spec = PROFILES[profile]
        prefix = f"ml-dsa-{profile}"
        _link(EXP_ROOT / "exp_res" / f"{prefix}-exp-res", FIGURE_ROOT / f"{prefix}-exp-res")
        _link(EXP_ROOT / "metadata" / spec["metadata"] / "s1.npy", FIGURE_ROOT / f"{prefix}-s1.npy")
        _link(EXP_ROOT / "data" / spec["data"], FIGURE_ROOT / f"{prefix}-data")

    os.chdir(FIGURE_ROOT)
    sys.path.insert(0, str(CODE_ROOT))
    import draw_fig
    import matplotlib.pyplot as plt

    # The renderer is intentionally non-interactive and writes PNGs to disk.
    plt.show = lambda *args, **kwargs: None

    selected = set(figures)
    if "all" in selected or "bp" in selected:
        for profile in PROFILES:
            draw_fig.draw_bp_fig(f"ml-dsa-{profile}")
    if "all" in selected or "convergence" in selected:
        for profile in PROFILES:
            draw_fig.draw_convergence_speed(f"ml-dsa-{profile}")
    if "all" in selected or "ta" in selected:
        draw_fig.draw_c_ta_res()
        draw_fig.draw_cs1_ta_res()
        draw_fig.draw_y_ta_res()
    if "all" in selected or "vary-c-all" in selected:
        for profile, accuracies in (("ref", [0.99, 0.98, 0.97, 0.96, 0.95]), ("m4", [0.99, 0.98, 0.97, 0.96])):
            draw_fig.view_bp_res_vary_c_acc(
                f"ml-dsa-{profile}", exp_min_sign_num=1, exp_max_sign_num=20,
                exp_repeat_num=10, c_acc_list=accuracies, sig_used="all"
            )
    if "all" in selected or "vary-c-rejected" in selected:
        for profile, accuracies in (("ref", [1.00, 0.99, 0.98, 0.97, 0.96]), ("m4", [1.00, 0.99, 0.98, 0.97])):
            draw_fig.view_bp_res_vary_c_acc(
                f"ml-dsa-{profile}", exp_min_sign_num=1, exp_max_sign_num=30,
                exp_repeat_num=10, c_acc_list=accuracies, sig_used="rej"
            )
    print(f"Rendered figures into {FIGURE_ROOT / 'img'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="validate the archived inputs")
    check_parser.add_argument("--profile", choices=("m4", "ref", "all"), default="all")

    setup_parser = subparsers.add_parser("setup", help="prepare isolated analysis paths")
    setup_parser.add_argument("--profile", choices=("m4", "ref", "all"), default="all")

    derive_parser = subparsers.add_parser("derive-z", help="verify or derive z=y+cs1 vectors")
    derive_parser.add_argument("--profile", choices=("m4", "ref", "all"), default="all")

    compare_parser = subparsers.add_parser(
        "compare", help="compare fresh results with the archived arrays"
    )
    compare_parser.add_argument("--profile", choices=("m4", "ref", "all"), default="all")

    render_parser = subparsers.add_parser("render", help="render figures from archived result arrays")
    render_parser.add_argument(
        "--figures",
        nargs="+",
        choices=("bp", "convergence", "ta", "vary-c-all", "vary-c-rejected", "all"),
        default=["all"],
    )

    args = parser.parse_args()
    if args.command == "check":
        profiles = list(PROFILES) if args.profile == "all" else [args.profile]
        return 0 if all(check_profile(profile) for profile in profiles) else 1
    if args.command == "setup":
        profiles = list(PROFILES) if args.profile == "all" else [args.profile]
        for profile in profiles:
            setup_workspace(profile)
        return 0
    if args.command == "derive-z":
        profiles = list(PROFILES) if args.profile == "all" else [args.profile]
        for profile in profiles:
            derive_z(profile)
        return 0
    if args.command == "compare":
        profiles = list(PROFILES) if args.profile == "all" else [args.profile]
        return 0 if all(compare_profile(profile) for profile in profiles) else 1
    if args.command == "render":
        render(args.figures)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
