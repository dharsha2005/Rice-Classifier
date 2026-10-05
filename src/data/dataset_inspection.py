"""
dataset_inspection.py
=====================
GrainSet Rice Dataset Inspection Script
Project: Rice Quality and Defect Assessment using Hybrid ML and Deep Feature Fusion

Purpose:
  - Parse the official GrainSet split files (rice_train, rice_val, rice_test, rice_train_bal)
  - Enumerate every class label found in the split files (no guessing)
  - Count images per class per split
  - Check whether every listed image path actually exists on disk
  - Detect missing files, duplicate filenames, invalid/corrupt images
  - Check for train/val/test filename overlap
  - Generate dataset_summary.csv and dataset_report.txt in the reports/ folder

Usage:
  python src/data/dataset_inspection.py

Notes:
  - The rice images must be downloaded from figshare and placed at:
      <RICE_IMAGE_ROOT>/train/<class_folder>/<image>.png
      <RICE_IMAGE_ROOT>/test/<class_folder>/<image>.png
  - This script NEVER copies, modifies, or rebalances the original dataset.
  - This script does NOT train any model.
"""

import os
import sys
import csv
import hashlib
import collections
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIGURATION - adjust RICE_IMAGE_ROOT to point at your rice image folder
# ---------------------------------------------------------------------------
# The split files list paths like:  train/0_NOR/<filename>.png
# Those paths are RELATIVE to the rice image root.
# Example: if your images are at C:/FinalData/rice/train/0_NOR/...
# then set RICE_IMAGE_ROOT = r"C:/FinalData/rice"

PROJECT_ROOT     = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR     = PROJECT_ROOT / "datasets"
REPORTS_DIR      = PROJECT_ROOT / "reports"

# Change this to where your actual rice PNGs are located
RICE_IMAGE_ROOT  = PROJECT_ROOT / "rice"

SPLIT_FILES = {
    "train":     "rice_train.txt",
    "val":       "rice_val.txt",
    "test":      "rice_test.txt",
    "train_bal": "rice_train_bal.txt",
}

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def read_split_file(filepath):
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                print(f"  [WARN] Line {line_num} malformed: {repr(line)}")
                continue
            rel_path, label_str = parts
            try:
                label_id = int(label_str)
            except ValueError:
                print(f"  [WARN] Line {line_num} non-integer label: {repr(label_str)}")
                continue
            entries.append((rel_path, label_id))
    return entries


def extract_class_folder(rel_path):
    parts = Path(rel_path).parts
    if len(parts) >= 2:
        return parts[1]
    return "UNKNOWN"


def check_image_valid(abs_path):
    try:
        from PIL import Image
        with Image.open(abs_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def file_md5(abs_path, chunk_size=8192):
    h = hashlib.md5()
    with open(abs_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fmt_bar(count, total, width=30):
    filled = int(width * count / total) if total > 0 else 0
    bar = "#" * filled + "." * (width - filled)
    pct = 100.0 * count / total if total > 0 else 0.0
    return f"[{bar}] {count:>6,}  ({pct:.1f}%)"


def inspect_dataset():
    print("=" * 70)
    print("  GrainSet Rice Dataset Inspection")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n[1] Parsing split files ...")
    split_data = {}
    all_class_folders = set()
    label_to_folder = {}

    for split_name, filename in SPLIT_FILES.items():
        filepath = DATASETS_DIR / filename
        if not filepath.exists():
            print(f"  [ERROR] Split file not found: {filepath}")
            split_data[split_name] = []
            continue
        entries = read_split_file(filepath)
        split_data[split_name] = entries
        for rel_path, label_id in entries:
            folder = extract_class_folder(rel_path)
            all_class_folders.add(folder)
            if label_id not in label_to_folder:
                label_to_folder[label_id] = folder
            elif label_to_folder[label_id] != folder:
                print(f"  [WARN] Label {label_id} maps to multiple folders: "
                      f"'{label_to_folder[label_id]}' and '{folder}'")
        print(f"  {split_name:12s}: {len(entries):>6,} entries  ->  {filename}")

    print("\n[2] Classes identified from split files ...")
    sorted_labels = sorted(label_to_folder.keys())
    print(f"  {'Label ID':<10} {'Class Folder':<20}")
    print(f"  {'-'*10} {'-'*20}")
    for label_id in sorted_labels:
        folder = label_to_folder[label_id]
        print(f"  {label_id:<10} {folder:<20}")

    print("\n[3] Counting images per class per split ...")
    counts = {sn: collections.Counter() for sn in SPLIT_FILES}
    for split_name, entries in split_data.items():
        for rel_path, label_id in entries:
            counts[split_name][label_id] += 1

    print("\n[4] Checking whether image files exist on disk ...")
    images_exist = RICE_IMAGE_ROOT.exists()
    if not images_exist:
        print(f"\n  WARNING: Rice image root NOT found: {RICE_IMAGE_ROOT}")
        print("  -> The actual rice PNG images have NOT been downloaded yet.")
        print("  -> Download from: https://doi.org/10.6084/m9.figshare.22987292.v3")
        print("  -> Update RICE_IMAGE_ROOT in this script after downloading.")
        print("  -> File existence, validity, duplicate checks will be SKIPPED.\n")

    missing_files = []
    invalid_images = []
    found_files = []

    if images_exist:
        print(f"  Rice image root found: {RICE_IMAGE_ROOT}")
        total_to_check = sum(len(v) for v in split_data.values())
        checked = 0
        for split_name, entries in split_data.items():
            for rel_path, label_id in entries:
                abs_path = RICE_IMAGE_ROOT / rel_path
                if not abs_path.exists():
                    missing_files.append((split_name, rel_path, label_id))
                else:
                    found_files.append((split_name, rel_path, label_id, abs_path))
                    if not check_image_valid(abs_path):
                        invalid_images.append((split_name, rel_path, label_id))
                checked += 1
                if checked % 1000 == 0:
                    print(f"    Checked {checked:>6,} / {total_to_check:,} ...")
        print(f"  Done. Checked {checked:,} entries.")
        print(f"  Missing  : {len(missing_files):,}")
        print(f"  Invalid  : {len(invalid_images):,}")
        print(f"  OK       : {len(found_files):,}")
    else:
        print("  (Skipped - image root not found)")

    print("\n[5] Checking for duplicate filenames ...")
    all_filenames = collections.defaultdict(list)
    for split_name, entries in split_data.items():
        for rel_path, label_id in entries:
            basename = Path(rel_path).name
            all_filenames[basename].append((split_name, rel_path))

    duplicate_filenames = {fn: locs for fn, locs in all_filenames.items() if len(locs) > 1}
    print(f"  Total unique filenames : {len(all_filenames):,}")
    print(f"  Filenames appearing >1 : {len(duplicate_filenames):,}")

    duplicate_hashes = {}
    if images_exist and found_files:
        print("  Computing file hashes for content-duplicate detection ...")
        hash_map = collections.defaultdict(list)
        for split_name, rel_path, label_id, abs_path in found_files:
            md5 = file_md5(abs_path)
            hash_map[md5].append((split_name, rel_path))
        duplicate_hashes = {h: locs for h, locs in hash_map.items() if len(locs) > 1}
        print(f"  Identical content pairs: {len(duplicate_hashes):,}")

    print("\n[6] Checking train / val / test filename overlaps ...")

    def get_filenames(split_name):
        return {Path(rp).name for rp, _ in split_data.get(split_name, [])}

    train_fns = get_filenames("train")
    val_fns   = get_filenames("val")
    test_fns  = get_filenames("test")

    train_val_overlap  = train_fns & val_fns
    train_test_overlap = train_fns & test_fns
    val_test_overlap   = val_fns   & test_fns
    all_three_overlap  = train_fns & val_fns & test_fns

    print(f"  Train n Val  overlap : {len(train_val_overlap):,}")
    print(f"  Train n Test overlap : {len(train_test_overlap):,}")
    print(f"  Val   n Test overlap : {len(val_test_overlap):,}")
    print(f"  All three overlap    : {len(all_three_overlap):,}")

    print("\n[7] Writing dataset_summary.csv ...")
    summary_path = REPORTS_DIR / "dataset_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["split", "class_id", "class_folder", "image_count"])
        for split_name in ["train", "val", "test", "train_bal"]:
            for label_id in sorted_labels:
                folder = label_to_folder.get(label_id, "UNKNOWN")
                count  = counts[split_name].get(label_id, 0)
                writer.writerow([split_name, label_id, folder, count])
    print(f"  Saved -> {summary_path}")

    total_train     = len(split_data.get("train", []))
    total_val       = len(split_data.get("val", []))
    total_test      = len(split_data.get("test", []))
    total_train_bal = len(split_data.get("train_bal", []))
    total_all       = total_train + total_val + total_test

    print("\n[8] Writing dataset_report.txt ...")
    report_path = REPORTS_DIR / "dataset_report.txt"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    def w(line=""):
        lines.append(line)

    w("=" * 68)
    w("  GrainSet Rice Dataset Inspection Report")
    w(f"  Generated : {now_str}")
    w("=" * 68)
    w()
    w("DATASET PATHS")
    w(f"  Split files directory : {DATASETS_DIR}")
    w(f"  Rice image root       : {RICE_IMAGE_ROOT}")
    w(f"  Image root exists     : {images_exist}")
    w()
    w("SPLIT FILES")
    for split_name, filename in SPLIT_FILES.items():
        fp = DATASETS_DIR / filename
        exists_str = "FOUND" if fp.exists() else "MISSING"
        w(f"  {split_name:12s} -> {filename}  [{exists_str}]")
    w()
    w("OVERVIEW")
    w(f"  Number of classes     : {len(sorted_labels)}")
    w(f"  Total train images    : {total_train:,}")
    w(f"  Total val images      : {total_val:,}")
    w(f"  Total test images     : {total_test:,}")
    w(f"  Total (train+val+test): {total_all:,}")
    w(f"  Balanced train subset : {total_train_bal:,}  (sub-sample of train, for class balance)")
    w()
    w("CLASS LABELS  (from data; abbreviations are NOT interpreted)")
    w(f"  {'ID':<6} {'Folder Name':<20} Note")
    w(f"  {'--':<6} {'-----------':<20} ----")
    for label_id in sorted_labels:
        folder = label_to_folder[label_id]
        w(f"  {label_id:<6} {folder:<20} (raw label as found in split files)")
    w()
    w("CLASS DISTRIBUTION PER SPLIT")
    for split_name in ["train", "val", "test", "train_bal"]:
        split_total = len(split_data.get(split_name, []))
        if split_total == 0:
            continue
        w()
        w(f"  [{split_name.upper()}]  Total: {split_total:,}")
        w(f"  {'ID':<6} {'Folder':<15} {'Count':>8}  Bar")
        w(f"  {'--':<6} {'------':<15} {'-----':>8}  ---")
        for label_id in sorted_labels:
            folder = label_to_folder[label_id]
            count  = counts[split_name].get(label_id, 0)
            bar    = fmt_bar(count, split_total)
            w(f"  {label_id:<6} {folder:<15} {count:>8,}  {bar}")
    w()
    w("FILE EXISTENCE CHECK")
    if images_exist:
        w(f"  Total entries checked : {total_train + total_val + total_test + total_train_bal:,}")
        w(f"  Files found on disk   : {len(found_files):,}")
        w(f"  Missing files         : {len(missing_files):,}")
        if missing_files:
            w()
            w("  First 20 missing files:")
            for i, (sn, rp, lid) in enumerate(missing_files[:20]):
                w(f"    [{i+1:>3}] [{sn}] {rp}")
            if len(missing_files) > 20:
                w(f"    ... and {len(missing_files)-20} more (see missing_files.txt)")
    else:
        w("  WARNING: Image root not present - file check SKIPPED.")
        w(f"  Expected root: {RICE_IMAGE_ROOT}")
        w("  Action: Download rice images from figshare and update RICE_IMAGE_ROOT.")
    w()
    w("IMAGE VALIDITY CHECK")
    if images_exist:
        w(f"  Invalid / corrupt images : {len(invalid_images):,}")
        if invalid_images:
            for sn, rp, lid in invalid_images[:10]:
                w(f"    [{sn}] {rp}")
    else:
        w("  (Skipped - image root not present)")
    w()
    w("DUPLICATE DETECTION")
    w(f"  Total unique filenames       : {len(all_filenames):,}")
    w(f"  Filenames appearing > 1 time : {len(duplicate_filenames):,}")
    if images_exist:
        w(f"  Content-identical file pairs : {len(duplicate_hashes):,}")
    if duplicate_filenames:
        w()
        w("  First 10 duplicate filenames:")
        for fn, locs in list(duplicate_filenames.items())[:10]:
            w(f"    {fn}")
            for sn, rp in locs:
                w(f"      -> [{sn}] {rp}")
    w()
    w("TRAIN / VAL / TEST OVERLAP")
    w(f"  Train n Val  : {len(train_val_overlap):,} overlapping filenames")
    w(f"  Train n Test : {len(train_test_overlap):,} overlapping filenames")
    w(f"  Val   n Test : {len(val_test_overlap):,} overlapping filenames")
    w(f"  All three    : {len(all_three_overlap):,} overlapping filenames")
    if train_val_overlap or train_test_overlap or val_test_overlap:
        w()
        w("  WARNING: Overlapping filenames found - verify these are not data leaks!")
        if train_val_overlap:
            sample = list(train_val_overlap)[:5]
            w(f"  Train n Val sample : {sample}")
        if train_test_overlap:
            sample = list(train_test_overlap)[:5]
            w(f"  Train n Test sample: {sample}")
    else:
        w()
        w("  OK: No filename overlaps between train / val / test splits.")
    w()
    w("ISSUES SUMMARY")
    issues = []
    if not images_exist:
        issues.append("CRITICAL: Rice image root not found - images not yet downloaded.")
    if missing_files:
        issues.append(f"WARNING : {len(missing_files)} files listed in splits are missing on disk.")
    if invalid_images:
        issues.append(f"WARNING : {len(invalid_images)} invalid/corrupt image files detected.")
    if duplicate_filenames:
        issues.append(f"INFO    : {len(duplicate_filenames)} filenames appear in >1 location.")
    if train_val_overlap or train_test_overlap or val_test_overlap:
        issues.append("WARNING : Filename overlap found between splits - investigate data leakage.")
    if not issues:
        issues.append("OK: No problems detected.")
    for issue in issues:
        w(f"  * {issue}")
    w()
    w("NEXT STEPS")
    if not images_exist:
        w("  1. Download rice images from:")
        w("       https://doi.org/10.6084/m9.figshare.22987292.v3")
        w("  2. Unzip so that paths match:")
        w(f"       {RICE_IMAGE_ROOT}/train/<class_folder>/<image>.png")
        w(f"       {RICE_IMAGE_ROOT}/test/<class_folder>/<image>.png")
        w("  3. Re-run this script to complete the full inspection.")
    else:
        w("  1. Review any missing or invalid files above.")
        w("  2. Proceed to preprocessing (Stage 2).")
    w()
    w("=" * 68)
    w("  End of Report")
    w("=" * 68)

    with open(report_path, "w", encoding="utf-8") as rpt:
        rpt.write("\n".join(lines))
    print(f"  Saved -> {report_path}")

    if images_exist and missing_files:
        missing_path = REPORTS_DIR / "missing_files.txt"
        with open(missing_path, "w", encoding="utf-8") as mf:
            mf.write(f"# Missing files - {len(missing_files)} total\n")
            mf.write("# Format: split | label_id | relative_path\n\n")
            for sn, rp, lid in missing_files:
                mf.write(f"{sn}\t{lid}\t{rp}\n")
        print(f"  Saved -> {missing_path}")

    print("\n" + "=" * 70)
    print("  INSPECTION COMPLETE")
    print("=" * 70)
    print(f"  Classes          : {len(sorted_labels)}")
    print(f"  Train images     : {total_train:,}")
    print(f"  Val images       : {total_val:,}")
    print(f"  Test images      : {total_test:,}")
    print(f"  Train+Val+Test   : {total_all:,}")
    print(f"  Balanced subset  : {total_train_bal:,}")
    print(f"  Image root found : {images_exist}")
    if images_exist:
        print(f"  Missing files    : {len(missing_files):,}")
        print(f"  Invalid images   : {len(invalid_images):,}")
        print(f"  Content dupes    : {len(duplicate_hashes):,}")
    print(f"  Filename dupes   : {len(duplicate_filenames):,}")
    print(f"  Train n Val ovlp : {len(train_val_overlap):,}")
    print(f"  Train n Test ovlp: {len(train_test_overlap):,}")
    print(f"  Val n Test ovlp  : {len(val_test_overlap):,}")
    print(f"\n  Outputs in: {REPORTS_DIR}")
    print("=" * 70)

    return {
        "classes": sorted_labels,
        "label_to_folder": label_to_folder,
        "counts": counts,
        "total_train": total_train,
        "total_val": total_val,
        "total_test": total_test,
        "total_train_bal": total_train_bal,
        "images_exist": images_exist,
        "missing_files": missing_files,
        "invalid_images": invalid_images,
        "duplicate_filenames": duplicate_filenames,
        "duplicate_hashes": duplicate_hashes,
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
    }


if __name__ == "__main__":
    results = inspect_dataset()
    sys.exit(0)
