import os
import pandas as pd
import xarray as xr

from argparse import ArgumentParser
from typing import List, Optional


def make_inaccuracy_or_incompleteness_table(which: str, sequence_names: List[str]) -> Optional[xr.DataArray]:
    """
    TODO

    :param which:           Which table to make ("Inaccuracy" or "Incompleteness").
    :param sequence_names:  The names of the GTA-IM sequences whose data is to be included in the table.
    :return:                The table.
    """
    da_sequences: List[xr.DataArray] = []
    for sequence in sequence_names:
        da_methods: List[xr.DataArray] = []
        method_tags: List[str] = ["lcrnet", "maskrcnn", "xnect"]
        available_method_tags: List[str] = []

        for method in method_tags:
            da_percents: List[xr.DataArray] = []
            percent_tags: List[str] = ["20", "40", "60", "80", "100"]

            for percent_to_stop in percent_tags:
                # FIXME: The location of the recon directory should be determined not hard-coded here.
                recon_dir: str = f"C:/datasets/gta-im/{sequence}/recon"

                if which == "Inaccuracy":
                    filename: str = os.path.join(
                        recon_dir, f"c2c_dist-gt_{method}_{percent_to_stop}-gt_gt_{percent_to_stop}.txt"
                    )
                else:
                    filename: str = os.path.join(
                        recon_dir, f"c2c_dist-gt_gt_{percent_to_stop}-gt_{method}_{percent_to_stop}.txt"
                    )

                if os.path.exists(filename):
                    da_percent: xr.DataArray = xr.DataArray(
                        pd.read_csv(filename, delimiter=' ').values[:, 0].astype(float),
                        name=f"{percent_to_stop}", dims="Attribute",
                        coords={"Attribute": ["mean", "median", "std", "min", "max"]}
                    )
                    da_percents.append(da_percent)
                else:
                    print(f"- Missing {filename}")

            if len(da_percents) > 0:
                da_method: xr.DataArray = xr.concat(da_percents, pd.Index(percent_tags, name="%"))
                da_method.name = method
                da_methods.append(da_method)
                available_method_tags.append(method)

        if len(da_methods) > 0:
            da_sequence: xr.DataArray = xr.concat(da_methods, pd.Index(available_method_tags, name="Method"))
            da_sequence.name = sequence
            da_sequences.append(da_sequence)

    if len(da_sequences) > 0:
        da: xr.DataArray = xr.concat(da_sequences, pd.Index(sequence_names, name="Sequence"))
        da.name = f"{which} Data"
        return da
    else:
        return None


def print_inaccuracy_and_incompleteness_tables(sequence_names: List[str]) -> None:
    """
    TODO

    :param sequence_names:  The names of the GTA-IM sequences whose data is to be included in the tables.
    """
    # Load the results for all the sequences and variants into two big tensors.
    da_inaccuracy: Optional[xr.DataArray] = make_inaccuracy_or_incompleteness_table("Inaccuracy", sequence_names)
    da_incompleteness: Optional[xr.DataArray] = make_inaccuracy_or_incompleteness_table("Incompleteness", sequence_names)

    # Tell pandas to show the whole tables.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    # TODO: Comment here.
    if da_inaccuracy is not None:
        print()
        print("Inaccuracy Metrics (m)")
        print(da_inaccuracy.sel(Attribute="mean").mean(dim="Sequence").to_pandas().transpose())
    else:
        print("- Cannot construct inaccuracy table (no data)")

    if da_incompleteness is not None:
        print()
        print("Incompleteness Metrics (m)")
        print(da_incompleteness.sel(Attribute="mean").mean(dim="Sequence").to_pandas().transpose())
    else:
        print("- Cannot construct incompleteness table (no data)")


def main() -> None:
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--sequence_list", "-s", type=str, default="gta_im_smg.txt",
        help="a file containing the list of GTA-IM sequences we want to use (one per line)"
    )
    args: dict = vars(parser.parse_args())

    # Load in the names of the sequences we want to use.
    with open(args["sequence_list"]) as f:
        sequence_names: List[str] = [line.rstrip() for line in f.readlines() if not line.lstrip().startswith("#")]

    # TODO: Comment here.
    print_inaccuracy_and_incompleteness_tables(sequence_names)


if __name__ == "__main__":
    main()
