import tensorflow_data_validation as tfdv
import pandas as pd


def generate_statistics(df: pd.DataFrame, dataset_name: str = 'dataset'):
    """
    Generates TFDV statistics from a Pandas DataFrame.

    Args:
        df:           Input DataFrame.
        dataset_name: Label used in print output.

    Returns:
        DatasetFeatureStatisticsList proto.
    """
    stats = tfdv.generate_statistics_from_dataframe(df)
    print(f"[statistics_utils] Generated statistics for '{dataset_name}' ({len(df)} rows)")
    return stats


def visualize_statistics(stats, dataset_name: str = 'dataset'):
    """
    Visualizes statistics for a single dataset using TFDV Facets.

    Args:
        stats:        DatasetFeatureStatisticsList proto.
        dataset_name: Display label.
    """
    tfdv.visualize_statistics(stats)


def compare_statistics(
    lhs_stats,
    rhs_stats,
    lhs_name: str = 'EVAL',
    rhs_name: str = 'TRAIN',
):
    """
    Visualizes two datasets' statistics side by side for comparison.
    Useful for spotting distribution skew between train and eval sets.

    Args:
        lhs_stats: Statistics for the left-hand dataset (typically eval).
        rhs_stats: Statistics for the right-hand dataset (typically train).
        lhs_name:  Display name for lhs dataset.
        rhs_name:  Display name for rhs dataset.
    """
    tfdv.visualize_statistics(
        lhs_statistics=lhs_stats,
        rhs_statistics=rhs_stats,
        lhs_name=lhs_name,
        rhs_name=rhs_name,
    )
    print(f"[statistics_utils] Comparing '{lhs_name}' vs '{rhs_name}'")


def generate_statistics_from_csv(
    csv_path: str,
    stats_options=None,
):
    """
    Generates TFDV statistics directly from a CSV file.
    Required for sliced statistics generation.

    Args:
        csv_path:      Path to the CSV file.
        stats_options: Optional tfdv.StatsOptions (e.g. for slicing).

    Returns:
        DatasetFeatureStatisticsList proto.
    """
    if stats_options:
        stats = tfdv.generate_statistics_from_csv(
            csv_path,
            stats_options=stats_options,
        )
    else:
        stats = tfdv.generate_statistics_from_csv(csv_path)

    print(f"[statistics_utils] Generated statistics from CSV: {csv_path}")
    return stats