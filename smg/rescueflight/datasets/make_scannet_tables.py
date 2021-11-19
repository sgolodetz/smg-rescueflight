import pandas as pd
import xarray as xr

from argparse import ArgumentParser
from typing import List


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--sequence_list", "-s", type=str, default="scannetv2_smg.txt",
        help="a file containing the list of ScanNet sequences we want to use (one per line)"
    )
    args: dict = vars(parser.parse_args())

    # Load in the names of the sequences we want to use.
    with open(args["sequence_list"]) as f:
        sequence_names: List[str] = [line.rstrip() for line in f.readlines()]

    # Specify the tags for the various different methods we want to consider.
    method_tags: List[str] = [
        "dvmvs_4m_gt", "dvmvs_pp_4m_gt", "mvdepth_4m_gt", "mvdepth_pp_4m_gt"
    ]

    # Load the results for all the sequences and methods into a big tensor.
    da_sequences: List[xr.DataArray] = []
    for sequence in sequence_names:
        da_methods: List[xr.DataArray] = []
        for method in method_tags:
            # FIXME: The ScanNet root directory should be detected not hard-coded here.
            filename: str = f"C:/datasets/scannet/{sequence}/recon/c2c_dist-{method}-gt_gt.txt"
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
    print(da.sel(Metric="mean").to_pandas())
    print()
    print(da.sel(Metric=["mean", "median", "std"]).mean(dim="Sequence").to_pandas())


if __name__ == "__main__":
    main()
