import os
import pandas as pd
import xarray as xr

from argparse import ArgumentParser
from typing import List


def main() -> None:
    # Tell pandas to show whole tables and avoid wrapping.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir", "-r", type=str, required=True,
        help="the root directory for the ScanNet dataset"
    )
    parser.add_argument(
        "--sequence_list", "-s", type=str, default="scannetv2_smg.txt",
        help="a file containing the list of ScanNet sequences we want to use (one per line)"
    )
    args: dict = vars(parser.parse_args())

    root_dir: str = args["root_dir"]
    sequence_list: str = args["sequence_list"]

    # Load in the names of the sequences we want to use.
    with open(sequence_list) as f:
        sequence_names: List[str] = [line.rstrip() for line in f.readlines() if not line.lstrip().startswith("#")]

    # Specify the tags for the various different methods we want to consider.
    method_tags: List[str] = [
        "mvdepth_4m_gt", "mvdepth_pp_4m_gt", "dvmvs_4m_gt", "dvmvs_pp_4m_gt", "foo"
    ]

    # Load the results for all the sequences and methods into a big tensor.
    da_sequences: List[xr.DataArray] = []
    for sequence in sequence_names:
        da_methods: List[xr.DataArray] = []
        for method in method_tags:
            filename: str = os.path.join(root_dir, f"{sequence}/recon/c2c_dist-{method}-gt_gt.txt")
            # filename: str = os.path.join(root_dir, f"{sequence}/recon/c2c_dist-gt_gt-{method}.txt")
            da_method: xr.DataArray = xr.DataArray(
                pd.read_csv(filename, delimiter=' ').values[:, 0].astype(float),
                name=method, dims="Metric", coords={"Metric": ["mean", "median", "std", "min", "max"]}
            )
            da_methods.append(da_method)

        da_sequence: xr.DataArray = xr.concat(da_methods, pd.Index(method_tags, name="Method"))
        da_sequence.name = sequence
        da_sequences.append(da_sequence)

    da: xr.DataArray = xr.concat(da_sequences, pd.Index(sequence_names, name="Sequence"))
    da.name = "Data"

    # Print out the tables we want to use for the paper.
    print()
    print(da.sel(Metric="mean").to_pandas())
    print()
    print(da.sel(Metric=["mean", "median", "std"]).mean(dim="Sequence").to_pandas())


if __name__ == "__main__":
    main()
