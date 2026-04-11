import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import slicing_util
from tensorflow_metadata.proto.v0.statistics_pb2 import DatasetFeatureStatisticsList


def generate_sliced_statistics(
    df: pd.DataFrame,
    schema,
    slice_features: dict,
    csv_path: str = 'slice_sample.csv',
):
    """
    Generates per-slice statistics for specified feature values.
    TFDV slicing only works with CSV input, so the DataFrame is
    first written to a temporary CSV file.

    Args:
        df:             DataFrame to slice (typically train_df).
        schema:         Reference schema proto.
        slice_features: Dict mapping feature names to values or None.
                        None means slice over all values of that feature.
                        Example: {'payment_type': None, 'VendorID': None}
        csv_path:       Path to write the temporary CSV.

    Returns:
        DatasetFeatureStatisticsList proto containing all slices.
    """
    # Write to CSV (required by TFDV slicing API)
    df.to_csv(csv_path, index=False)
    print(f"[slicing_utils] Written {len(df)} rows to {csv_path}")

    # Build slice function
    slice_fn = slicing_util.get_feature_value_slicer(features=slice_features)

    # Build stats options with slicing
    stats_options = tfdv.StatsOptions(
        schema=schema,
        experimental_slice_functions=[slice_fn],
        infer_type_from_schema=True,
    )

    sliced_stats = tfdv.generate_statistics_from_csv(
        csv_path,
        stats_options=stats_options,
    )

    dataset_names = [d.name for d in sliced_stats.datasets]
    print(f"[slicing_utils] Generated slices: {dataset_names}")
    return sliced_stats


def get_slice(sliced_stats, slice_name: str) -> DatasetFeatureStatisticsList:
    """
    Extracts a single named slice from sliced statistics and
    converts it to DatasetFeatureStatisticsList for visualization.

    Args:
        sliced_stats: Full sliced statistics proto.
        slice_name:   Name of the slice (e.g. 'payment_type_Credit').

    Returns:
        DatasetFeatureStatisticsList containing only the requested slice.
    """
    for dataset in sliced_stats.datasets:
        if dataset.name == slice_name:
            stats_list = DatasetFeatureStatisticsList()
            stats_list.datasets.extend([dataset])
            return stats_list

    available = [d.name for d in sliced_stats.datasets]
    raise ValueError(
        f"Slice '{slice_name}' not found. Available slices: {available}"
    )


def compare_slices(
    sliced_stats,
    lhs_name: str,
    rhs_name: str,
):
    """
    Compares two named slices side by side using TFDV visualization.

    Args:
        sliced_stats: Full sliced statistics proto.
        lhs_name:     Name of the left-hand slice.
        rhs_name:     Name of the right-hand slice.
    """
    lhs = get_slice(sliced_stats, lhs_name)
    rhs = get_slice(sliced_stats, rhs_name)

    tfdv.visualize_statistics(
        lhs_statistics=lhs,
        rhs_statistics=rhs,
        lhs_name=lhs_name,
        rhs_name=rhs_name,
    )
    print(f"[slicing_utils] Comparing slice '{lhs_name}' vs '{rhs_name}'")


def list_slices(sliced_stats) -> list[str]:
    """
    Returns the names of all available slices in the statistics proto.

    Args:
        sliced_stats: Full sliced statistics proto.

    Returns:
        List of slice names.
    """
    names = [d.name for d in sliced_stats.datasets]
    print(f"[slicing_utils] Available slices: {names}")
    return names