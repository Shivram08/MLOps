import pandas as pd


# Anomalous rows to inject into the eval set.
# Each row targets a specific type of data quality issue.
ANOMALOUS_ROWS = [
    {
        # Negative fare — physically impossible
        'VendorID':             '1',
        'passenger_count':      2,
        'trip_distance':        3.5,
        'RatecodeID':           '1',
        'payment_type':         'Credit',
        'fare_amount':          -15.0,
        'tip_amount':           0.0,
        'tolls_amount':         0.0,
        'total_amount':         -15.0,
        'trip_duration_seconds': 720,
    },
    {
        # Zero passengers — invalid for a taxi trip
        'VendorID':             '2',
        'passenger_count':      0,
        'trip_distance':        1.2,
        'RatecodeID':           '1',
        'payment_type':         'Cash',
        'fare_amount':          7.5,
        'tip_amount':           0.0,
        'tolls_amount':         0.0,
        'total_amount':         7.5,
        'trip_duration_seconds': 300,
    },
    {
        # Trip duration of 24 hours — impossible for a NYC taxi trip
        'VendorID':             '1',
        'passenger_count':      1,
        'trip_distance':        2.0,
        'RatecodeID':           '1',
        'payment_type':         'Credit',
        'fare_amount':          10.0,
        'tip_amount':           2.0,
        'tolls_amount':         0.0,
        'total_amount':         12.0,
        'trip_duration_seconds': 86400,
    },
    {
        # Unknown payment type not in training schema domain
        'VendorID':             '2',
        'passenger_count':      3,
        'trip_distance':        5.0,
        'RatecodeID':           '1',
        'payment_type':         'CRYPTO',
        'fare_amount':          18.0,
        'tip_amount':           3.0,
        'tolls_amount':         0.0,
        'total_amount':         21.0,
        'trip_duration_seconds': 900,
    },
    {
        # RatecodeID 99 — out of valid range (valid: 1-6)
        'VendorID':             '1',
        'passenger_count':      1,
        'trip_distance':        0.5,
        'RatecodeID':           '99',
        'payment_type':         'Cash',
        'fare_amount':          5.0,
        'tip_amount':           0.0,
        'tolls_amount':         0.0,
        'total_amount':         5.0,
        'trip_duration_seconds': 180,
    },
]


def inject_anomalies(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends anomalous rows to the eval DataFrame to simulate
    real-world data quality issues that TFDV should detect.

    Anomalies injected:
        1. Negative fare_amount           → out-of-range numeric
        2. Zero passenger_count           → out-of-range numeric
        3. trip_duration_seconds = 86400  → extreme outlier
        4. payment_type = 'CRYPTO'        → unknown categorical value
        5. RatecodeID = '99'              → unexpected string value

    Args:
        eval_df: Original eval DataFrame.

    Returns:
        eval_df with anomalous rows appended.
    """
    anomaly_df = pd.DataFrame(ANOMALOUS_ROWS)

    # Ensure dtypes match the eval DataFrame
    for col in eval_df.select_dtypes(include='object').columns:
        if col in anomaly_df.columns:
            anomaly_df[col] = anomaly_df[col].astype(str)

    result = pd.concat([eval_df, anomaly_df], ignore_index=True)
    print(f"[anomaly_injector] Injected {len(ANOMALOUS_ROWS)} anomalous rows → eval set now {len(result)} rows")
    return result