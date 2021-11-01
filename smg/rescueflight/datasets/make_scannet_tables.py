import pandas as pd
import xarray as xr

from typing import List


def main() -> None:
    sequence_names: List[str] = [
        "scene0707_00", "scene0708_00", "scene0709_00", "scene0710_00"
    ]
    method_names: List[str] = [
        "dvmvs_4m_gt", "dvmvs_pp_4m_gt", "mvdepth_4m_gt", "mvdepth_pp_4m_gt"
    ]

    da_sequences: List[xr.DataArray] = []
    for sequence in sequence_names:
        da_methods: List[xr.DataArray] = []
        for method in method_names:
            filename: str = f"C:/datasets/scannet/{sequence}/recon/c2c_dist-{method}-gt_gt.txt"
            da_method: xr.DataArray = xr.DataArray(
                pd.read_csv(filename, delimiter=' ').values[:, 0].astype(float),
                name=method, dims="Attribute", coords={"Attribute": ["mean", "median", "std", "min", "max"]}
            )
            da_methods.append(da_method)

        da_sequence: xr.DataArray = xr.concat(da_methods, pd.Index(method_names, name="Method"))
        da_sequence.name = sequence
        da_sequences.append(da_sequence)

    da: xr.DataArray = xr.concat(da_sequences, pd.Index(sequence_names, name="Sequence"))
    da.name = "Data"

    # print(da)
    # print(da.sel(Attribute="median").idxmin(dim="Method"))
    # print(da.sel(Attribute="mean"))
    # print(da.sel(Attribute="mean").mean(dim="Sequence").to_pandas())
    print(da.sel(Attribute=["mean", "median", "std"]).mean(dim="Sequence").to_pandas())
    print()
    print(da.sel(Attribute="mean").to_pandas())


if __name__ == "__main__":
    main()
