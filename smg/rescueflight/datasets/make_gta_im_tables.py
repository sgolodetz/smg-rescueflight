import numpy as np
import os
import pandas as pd
import xarray as xr

from argparse import ArgumentParser
# noinspection PyUnresolvedReferences
from numpy import nan
from typing import Dict, List, Optional


def load_inaccuracy_or_incompleteness_results(which: str, sequence_names: List[str], root_dir: str) \
        -> Optional[xr.DataArray]:
    """
    Try to load in the data we need for either the inaccuracy or the incompleteness tensor, and to make the tensor.

    :param which:           Which tensor to make ("Inaccuracy" or "Incompleteness").
    :param sequence_names:  The names of the GTA-IM sequences whose data is to be included in the tensor.
    :param root_dir:        The root directory of the GTA-IM dataset.
    :return:                The tensor, if data for it exists, or None otherwise.
    """
    da_sequences: List[xr.DataArray] = []
    for sequence in sequence_names:
        available_method_tags: List[str] = []
        da_methods: List[xr.DataArray] = []
        method_tags: List[str] = ["lcrnet", "maskrcnn", "nomask", "xnect"]
        recon_dir: str = os.path.join(root_dir, sequence, "recon")

        for method in method_tags:
            da_percents: List[xr.DataArray] = []
            percent_tags: List[str] = ["20", "40", "60", "80", "100"]

            for percent_to_stop in percent_tags:
                if which == "Inaccuracy":
                    filename: str = os.path.join(
                        recon_dir, f"c2c_dist-gt_{method}_{percent_to_stop}-gt_gt_100.txt"
                    )
                else:
                    filename: str = os.path.join(
                        recon_dir, f"c2c_dist-gt_gt_100-gt_{method}_{percent_to_stop}.txt"
                    )

                if os.path.exists(filename):
                    da_percent: xr.DataArray = xr.DataArray(
                        pd.read_csv(filename, delimiter=' ').values[:, 0].astype(float),
                        name=percent_to_stop, dims="Metric",
                        coords={"Metric": ["mean", "median", "std", "min", "max"]}
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


def load_people_mask_results(sequence_names: List[str], root_dir: str) -> Optional[xr.DataArray]:
    """
    Try to load in the data we need for the people mask tensor, and to make the tensor.

    :param sequence_names:  The names of the GTA-IM sequences whose data is to be included in the tensor.
    :param root_dir:        The root directory of the GTA-IM dataset.
    :return:                The tensor, if data for it exists, or None otherwise.
    """
    da_sequences: List[xr.DataArray] = []
    for sequence in sequence_names:
        available_method_tags: List[str] = []
        da_methods: List[xr.DataArray] = []
        method_tags: List[str] = ["lcrnet", "maskrcnn", "xnect"]
        people_dir: str = os.path.join(root_dir, sequence, "people")

        for method in method_tags:
            filename: str = os.path.join(people_dir, f"people_mask_metrics-{method}.txt")
            if os.path.exists(filename):
                da_method: xr.DataArray = xr.DataArray(
                    pd.read_csv(filename, delimiter=':', header=None).values[:, 1].astype(float),
                    name=method, dims="Metric",
                    coords={"Metric": ["IoG Frame Count", "IoU Frame Count", "Mean IoG", "Mean IoU", "Mean F1"]}
                )
                da_methods.append(da_method)
                available_method_tags.append(method)
            else:
                print(f"- Missing {filename}")

        if len(da_methods) > 0:
            da_sequence: xr.DataArray = xr.concat(da_methods, pd.Index(available_method_tags, name="Method"))
            da_sequence.name = sequence
            da_sequences.append(da_sequence)

    if len(da_sequences) > 0:
        da: xr.DataArray = xr.concat(da_sequences, pd.Index(sequence_names, name="Sequence"))
        da.name = f"People Mask Data"
        return da
    else:
        return None


def load_skeleton_results(sequence_names: List[str], root_dir: str) -> Optional[xr.DataArray]:
    """
    Try to load in the data we need for the 3D skeleton tensor, and to make the tensor.

    :param sequence_names:  The names of the GTA-IM sequences whose data is to be included in the tensor.
    :param root_dir:        The root directory of the GTA-IM dataset.
    :return:                The tensor, if data for it exists, or None otherwise.
    """
    available_sequence_names: List[str] = []
    da_sequences: List[xr.DataArray] = []
    for sequence in sequence_names:
        available_method_tags: List[str] = []
        da_methods: List[xr.DataArray] = []
        method_tags: List[str] = ["lcrnet", "xnect"]
        people_dir: str = os.path.join(root_dir, sequence, "people")

        for method in method_tags:
            filename: str = os.path.join(people_dir, f"skeleton_metrics-{method}.txt")
            if os.path.exists(filename):
                da_metrics: List[xr.DataArray] = []
                metric_tags: List[str] = []

                with open(filename) as f:
                    lines: List[str] = [line.strip() for line in f.readlines() if line != ""]
                    for line in lines:
                        metric, joint_to_value_data = line.split(":", 1)
                        joint_to_value_map: Dict[str, float] = eval(joint_to_value_data)
                        da_metric: xr.DataArray = xr.DataArray(
                            np.fromiter(joint_to_value_map.values(), dtype=float),
                            name=metric, dims="Joint", coords={"Joint": list(joint_to_value_map.keys())}
                        )
                        da_metrics.append(da_metric)
                        metric_tags.append(metric)

                da_method: xr.DataArray = xr.concat(da_metrics, pd.Index(metric_tags, name="Metric"))
                da_method.name = method
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
        da.name = f"3D Skeleton Data"
        return da
    else:
        return None


def print_inaccuracy_and_incompleteness_tables(sequence_names: List[str], root_dir: str) -> None:
    """
    Try to print out the inaccuracy and incompleteness tables we need for the paper.

    :param sequence_names:  The names of the GTA-IM sequences whose data is to be included in the tables.
    :param root_dir:        The root directory of the GTA-IM dataset.
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
        print("Mean Inaccuracy (mean of all sequences, m)")
        print(da_inaccuracy.sel(Metric="mean").mean(dim="Sequence").to_pandas().transpose())
        print()
    else:
        print("- Couldn't load inaccuracy results (no data)")

    if da_incompleteness is not None:
        print("Mean Incompleteness (mean of all sequences, m)")
        print(da_incompleteness.sel(Metric="mean").mean(dim="Sequence").to_pandas().transpose())
    else:
        print("- Couldn't load incompleteness results (no data)")


def print_people_mask_tables(sequence_names: List[str], root_dir: str) -> None:
    """
    Try to print out the people mask tables we need for the paper.

    :param sequence_names:  The names of the GTA-IM sequences whose data is to be included in the tables.
    :param root_dir:        The root directory of the GTA-IM dataset.
    """
    # Load the results into an xarray tensor.
    da_people_masks: Optional[xr.DataArray] = load_people_mask_results(sequence_names, root_dir)

    # Try to print out the tables.
    print()

    if da_people_masks is not None:
        print("People Mask Metrics (mean of all sequences)")
        print(
            da_people_masks.sel(Metric=["Mean IoG", "Mean IoU", "Mean F1"]).mean(dim="Sequence").to_pandas().transpose()
        )
    else:
        print("- Couldn't load people mask results (no data)")


def print_skeleton_tables(sequence_names: List[str], root_dir: str) -> None:
    """
    Try to print out the 3D skeleton tables we need for the paper.

    :param sequence_names:  The names of the GTA-IM sequences whose data is to be included in the tables.
    :param root_dir:        The root directory of the GTA-IM dataset.
    """
    # Load the results into an xarray tensor.
    da_skeletons: Optional[xr.DataArray] = load_skeleton_results(sequence_names, root_dir)

    # Try to print out the tables.
    print()

    if da_skeletons is not None:
        print("MPJPEs (mean of all sequences, m)")
        print(da_skeletons.sel(Metric="MPJPEs").mean(dim="Sequence").to_pandas().transpose())
        print()
        print("3DPCKs (mean of all sequences, %)")
        print(da_skeletons.sel(Metric="3DPCKs").mean(dim="Sequence").to_pandas().transpose())
    else:
        print("- Couldn't load 3D skeleton results (no data)")


def main() -> None:
    # Tell numpy to avoid using scientific notation for numbers.
    np.set_printoptions(suppress=True)

    # Tell pandas to show whole tables.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir", "-r", type=str, required=True,
        help="the root directory for the GTA-IM dataset"
    )
    parser.add_argument(
        "--sequence_list", "-s", type=str, default="gta_im_smg.txt",
        help="a file containing the list of GTA-IM sequences we want to use (one per line)"
    )
    args: dict = vars(parser.parse_args())

    root_dir: str = args["root_dir"]
    sequence_list: str = args["sequence_list"]

    # Load in the names of the sequences we want to use.
    with open(sequence_list) as f:
        sequence_names: List[str] = [line.rstrip() for line in f.readlines() if not line.lstrip().startswith("#")]

    # Load in the relevant results and print out the tables we need for the paper.
    print_inaccuracy_and_incompleteness_tables(sequence_names, root_dir)
    print_people_mask_tables(sequence_names, root_dir)
    print_skeleton_tables(sequence_names, root_dir)


if __name__ == "__main__":
    main()
