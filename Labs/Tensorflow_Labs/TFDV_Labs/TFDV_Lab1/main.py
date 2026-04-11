"""
NYC Taxi Trip Data Validation Pipeline
======================================
End-to-end TFDV pipeline for the NYC Taxi Trips dataset.

Pipeline steps:
    1. Load and split data
    2. Generate training statistics
    3. Infer schema from training data
    4. Inject anomalies into eval set
    5. Generate eval statistics
    6. Compare train vs eval statistics
    7. Detect anomalies
    8. Fix schema
    9. Validate again (should show no anomalies)
    10. Slice analysis by payment_type and VendorID
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.data_loader       import load_data, split_data
from src.anomaly_injector  import inject_anomalies
from src.statistics_utils  import generate_statistics, compare_statistics
from src.schema_utils      import (
    infer_schema,
    detect_anomalies,
    fix_schema,
    save_schema,
)
from src.slicing_utils     import (
    generate_sliced_statistics,
    compare_slices,
    list_slices,
)


DATA_PATH   = 'data/nyc_taxi.csv'
SCHEMA_PATH = 'schema/schema.pbtxt'
CSV_PATH    = 'slice_sample.csv'


def main():
    print("\n" + "=" * 60)
    print("  NYC Taxi TFDV Pipeline")
    print("=" * 60)

    # ── Step 1: Load and split ─────────────────────────────────────
    print("\n[Step 1] Loading and splitting data...")
    df = load_data(DATA_PATH)
    train_df, eval_df = split_data(df, test_size=0.2, shuffle=False)

    # ── Step 2: Training statistics ────────────────────────────────
    print("\n[Step 2] Generating training statistics...")
    train_stats = generate_statistics(train_df, dataset_name='TRAIN')

    # ── Step 3: Infer schema ───────────────────────────────────────
    print("\n[Step 3] Inferring schema from training data...")
    schema = infer_schema(train_stats)
    save_schema(schema, SCHEMA_PATH)

    # ── Step 4: Inject anomalies into eval ────────────────────────
    print("\n[Step 4] Injecting anomalies into eval set...")
    eval_df_dirty = inject_anomalies(eval_df)

    # ── Step 5: Eval statistics ────────────────────────────────────
    print("\n[Step 5] Generating eval statistics...")
    eval_stats_dirty = generate_statistics(eval_df_dirty, dataset_name='EVAL (dirty)')

    # ── Step 6: Compare train vs eval ─────────────────────────────
    print("\n[Step 6] Comparing train vs eval statistics...")
    compare_statistics(eval_stats_dirty, train_stats, lhs_name='EVAL', rhs_name='TRAIN')

    # ── Step 7: Detect anomalies ───────────────────────────────────
    print("\n[Step 7] Detecting anomalies in eval set...")
    anomalies = detect_anomalies(eval_stats_dirty, schema)

    # ── Step 8: Fix schema ─────────────────────────────────────────
    print("\n[Step 8] Fixing schema...")
    schema = fix_schema(schema)
    save_schema(schema, SCHEMA_PATH)

    # ── Step 9: Re-validate ────────────────────────────────────────
    print("\n[Step 9] Re-validating eval after schema fix...")
    updated_anomalies = detect_anomalies(eval_stats_dirty, schema)

    # ── Step 10: Slice analysis ────────────────────────────────────
    print("\n[Step 10] Running slice analysis...")

    # Slice by payment_type
    payment_slices = generate_sliced_statistics(
        df=train_df,
        schema=schema,
        slice_features={'payment_type': None},
        csv_path=CSV_PATH,
    )
    list_slices(payment_slices)
    compare_slices(
        payment_slices,
        lhs_name='payment_type_Credit',
        rhs_name='payment_type_Cash',
    )

    # Slice by VendorID
    vendor_slices = generate_sliced_statistics(
        df=train_df,
        schema=schema,
        slice_features={'VendorID': None},
        csv_path=CSV_PATH,
    )
    list_slices(vendor_slices)
    compare_slices(
        vendor_slices,
        lhs_name='VendorID_1',
        rhs_name='VendorID_2',
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()