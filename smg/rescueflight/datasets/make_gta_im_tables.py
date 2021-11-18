import os
import pandas as pd
import xarray as xr

from argparse import ArgumentParser
from typing import List


def load_results(sequence_names: List[str], which: str) -> xr.DataArray:
    da_sequences: List[xr.DataArray] = []
    for sequence in sequence_names:
        da_methods: List[xr.DataArray] = []
        method_tags: List[str] = ["lcrnet", "maskrcnn", "xnect"]

        for method in method_tags:
            da_percents: List[xr.DataArray] = []
            percent_tags: List[str] = ["20", "40", "60", "80", "100"]

            for percent_to_stop in percent_tags:
                # FIXME: The location of the recon directory should be detected not hard-coded here.
                recon_dir: str = f"C:/datasets/gta-im/{sequence}/recon"
                if which == "Inaccuracy":
                    filename: str = os.path.join(
                        recon_dir, f"c2c_dist-gt_{method}_{percent_to_stop}-gt_gt_{percent_to_stop}.txt"
                    )
                else:
                    filename: str = os.path.join(
                        recon_dir, f"c2c_dist-gt_gt_{percent_to_stop}-gt_{method}_{percent_to_stop}.txt"
                    )

                da_percent: xr.DataArray = xr.DataArray(
                    pd.read_csv(filename, delimiter=' ').values[:, 0].astype(float),
                    name=f"{percent_to_stop}", dims="Attribute",
                    coords={"Attribute": ["mean", "median", "std", "min", "max"]}
                )
                da_percents.append(da_percent)

            da_method: xr.DataArray = xr.concat(da_percents, pd.Index(percent_tags, name="%"))
            da_method.name = method
            da_methods.append(da_method)

        da_sequence: xr.DataArray = xr.concat(da_methods, pd.Index(method_tags, name="Method"))
        da_sequence.name = sequence
        da_sequences.append(da_sequence)

    da: xr.DataArray = xr.concat(da_sequences, pd.Index(sequence_names, name="Sequence"))
    da.name = f"{which} Data"
    return da


def main() -> None:
    # Parse any command-line arguments.
    # parser = ArgumentParser()
    # parser.add_argument(
    #     "--sequence_list", "-s", type=str, default="scannetv2_smg.txt",
    #     help="a file containing the list of ScanNet sequences we want to use (one per line)"
    # )
    # args: dict = vars(parser.parse_args())
    #
    # # Load in the names of the sequences we want to use.
    # with open(args["sequence_list"]) as f:
    #     sequence_names: List[str] = [line.rstrip() for line in f.readlines()]

    sequence_names: List[str] = ["FPS-5/2020-06-09-17-14-03"]

    # Load the results for all the sequences and variants into two big tensors.
    da_inaccuracy: xr.DataArray = load_results(sequence_names, "Inaccuracy")
    da_incompleteness: xr.DataArray = load_results(sequence_names, "Incompleteness")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(da_inaccuracy.sel(Sequence=sequence_names[0]).sel(Attribute="mean").to_pandas().transpose())
    print(da_incompleteness.sel(Sequence=sequence_names[0]).sel(Attribute="mean").to_pandas().transpose())


if __name__ == "__main__":
    main()
