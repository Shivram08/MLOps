import pandas as pd
from sklearn.model_selection import train_test_split


# Expected columns after loading
FEATURE_COLUMNS = [
    'VendorID',
    'passenger_count',
    'trip_distance',
    'RatecodeID',
    'payment_type',
    'fare_amount',
    'tip_amount',
    'tolls_amount',
    'total_amount',
    'trip_duration_seconds',
]

TARGET_COLUMN = 'tip_amount'

CATEGORICAL_COLUMNS = ['VendorID', 'RatecodeID', 'payment_type']
NUMERIC_COLUMNS = [
    'passenger_count',
    'trip_distance',
    'fare_amount',
    'tip_amount',
    'tolls_amount',
    'total_amount',
    'trip_duration_seconds',
]


def load_data(path: str = 'data/nyc_taxi.parquet') -> pd.DataFrame:
    """
    Loads NYC Taxi dataset from CSV or Parquet, computes trip_duration_seconds
    from pickup/dropoff timestamps, and returns a cleaned DataFrame
    with only the relevant feature columns.

    Args:
        path: Path to the raw CSV or Parquet file.

    Returns:
        Cleaned DataFrame.
    """
    if path.endswith('.parquet'):
        df = pd.read_parquet(path).head(50000)
    else:
        df = pd.read_csv(path, nrows=50000)

    # Compute trip duration in seconds from timestamp columns
    df['tpep_pickup_datetime']  = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['trip_duration_seconds'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    ).dt.total_seconds().astype(int)

    # Map payment_type integer codes to readable strings
    payment_map = {1: 'Credit', 2: 'Cash', 3: 'No charge', 4: 'Dispute', 5: 'Unknown', 6: 'Voided'}
    df['payment_type'] = df['payment_type'].map(payment_map).fillna('Unknown')

    # Cast categorical columns to string for TFDV compatibility
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)

    # Keep only relevant columns
    df = df[FEATURE_COLUMNS].dropna()

    print(f"[data_loader] Loaded {len(df)} rows | Columns: {list(df.columns)}")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    shuffle: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame into train and eval sets.

    Args:
        df:           Input DataFrame.
        test_size:    Fraction of data for eval set.
        shuffle:      Whether to shuffle before splitting.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, eval_df).
    """
    train_df, eval_df = train_test_split(
        df,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )
    print(f"[data_loader] Train: {len(train_df)} rows | Eval: {len(eval_df)} rows")
    return train_df, eval_df