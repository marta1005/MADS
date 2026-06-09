from __future__ import annotations

import argparse
from pathlib import Path

from multiads.utilities.campaign_export import (
    find_shard_flat_datasets,
    merge_flat_campaign_csvs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN_ROOT = (
    REPO_ROOT / "outputs" / "CTA_case" / "datasets" / "campaign_001_exploration"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CTA DUST DOE shard CSV files into one flat campaign dataset.",
    )
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Merged output CSV. Defaults to <campaign-root>/cta_dust_vlm_dataset_flat.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    campaign_root = args.campaign_root.resolve()
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else campaign_root / "cta_dust_vlm_dataset_flat.csv"
    )
    shard_csvs = find_shard_flat_datasets(campaign_root)
    row_count = merge_flat_campaign_csvs(shard_csvs, output_csv)

    print("CTA DUST shards merged")
    print(f"  campaign_root = {campaign_root}")
    print(f"  n_shards = {len(shard_csvs)}")
    print(f"  n_rows = {row_count}")
    print(f"  output_csv = {output_csv}")


if __name__ == "__main__":
    main()
