import numpy as np
import os
import pandas as pd
import xarray as xr

from argparse import ArgumentParser
from typing import List, Optional


def load_inaccuracy_or_incompleteness_results(which: str, sequence_names: List[str], root_dir: str) \
        -> Optional[xr.DataArray]:
    """
    Try to load in the data we need for either the inaccuracy or the incompleteness tensor, and to make the tensor.

    :param which:           Which tensor to make ("Inaccuracy" or "Incompleteness").
    :param sequence_names:  The names of the sequences whose data is to be included in the tensor.
    :param root_dir:        The root directory of the dataset.
    :return:                The tensor, if data for it exists, or None otherwise.
    """
    available_sequence_names: List[str] = []
    da_sequences: List[xr.DataArray] = []
    for sequence in sequence_names:
        available_method_tags: List[str] = []
        da_methods: List[xr.DataArray] = []
        method_tags: List[str] = ["mvdepth_4m_gt", "mvdepth_pp_4m_gt", "dvmvs_4m_gt", "dvmvs_pp_4m_gt", "foo"]
        recon_dir: str = os.path.join(root_dir, sequence, "recon")

        for method in method_tags:
            if which == "Inaccuracy":
                filename: str = os.path.join(recon_dir, f"c2c_dist-{method}-gt_gt.txt")
            else:
                filename: str = os.path.join(recon_dir, f"c2c_dist-gt_gt-{method}.txt")

            if os.path.exists(filename):
                da_method: xr.DataArray = xr.DataArray(
                    pd.read_csv(filename, delimiter=' ').values[:, 0].astype(float),
                    name=method, dims="Metric",
                    coords={"Metric": ["mean", "median", "std", "min", "max"]}
                )
                da_methods.append(da_method)
                available_method_tags.append(method)
            else:
                print(f"- Missing {filename}")

        if len(da_methods) > 0:
            da_sequence: xr.DataArray = xr.concat(da_methods, pd.Index(available_method_tags, name="Method"))
            da_sequence.name = sequence
            da_sequences.append(da_sequence)
            available_sequence_names.append(sequence)

    if len(da_sequences) > 0:
        da: xr.DataArray = xr.concat(da_sequences, pd.Index(available_sequence_names, name="Sequence"))
        da.name = f"{which} Data"
        return da
    else:
        return None


def print_inaccuracy_and_incompleteness_tables(sequence_names: List[str], root_dir: str) -> None:
    """
    Try to print out the inaccuracy and incompleteness tables we need for the paper.

    :param sequence_names:  The names of the sequences whose data is to be included in the tables.
    :param root_dir:        The root directory of the dataset.
    """
    # Load the results into xarray tensors.
    da_inaccuracy: Optional[xr.DataArray] = load_inaccuracy_or_incompleteness_results(
        "Inaccuracy", sequence_names, root_dir
    )
    da_incompleteness: Optional[xr.DataArray] = load_inaccuracy_or_incompleteness_results(
        "Incompleteness", sequence_names, root_dir
    )

    # Try to print out the tables.
    print()

    if da_inaccuracy is not None:
        print("Per-Sequence Mean Inaccuracy (m)")
        print(da_inaccuracy.sel(Metric="mean").to_pandas())
        print()
        print("Dataset Mean Inaccuracy (mean of all sequences, m)")
        print(da_inaccuracy.sel(Metric="mean").mean(dim="Sequence").to_pandas().transpose().to_string(dtype=False))
        print()
    else:
        print("- Couldn't load inaccuracy results (no data)")

    if da_incompleteness is not None:
        print("Per-Sequence Mean Incompleteness (m)")
        print(da_incompleteness.sel(Metric="mean").to_pandas())
        print()
        print("Dataset Mean Incompleteness (mean of all sequences, m)")
        print(da_incompleteness.sel(Metric="mean").mean(dim="Sequence").to_pandas().transpose().to_string(dtype=False))
    else:
        print("- Couldn't load incompleteness results (no data)")


def main() -> None:
    # Tell numpy to avoid using scientific notation for numbers.
    np.set_printoptions(suppress=True)

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

    # Load in the relevant results and print out the tables we need for the paper.
    print_inaccuracy_and_incompleteness_tables(sequence_names, root_dir)


if __name__ == "__main__":
    main()
