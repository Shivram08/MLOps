# TFDV Lab 1 — NYC Taxi Trip Data Validation

> **Course Context:** MLOps Lab — TensorFlow Data Validation  
> **Dataset:** NYC Taxi Trips (50,000 rows)  
> **TFDV Version:** 1.17.0  
> **Python:** 3.10.20  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Modifications from Original Lab](#2-modifications-from-original-lab)
3. [Project Structure](#3-project-structure)
4. [Dataset](#4-dataset)
5. [Environment Setup](#5-environment-setup)
6. [Running the Notebook](#6-running-the-notebook)
7. [Notebook Walkthrough — Step by Step](#7-notebook-walkthrough--step-by-step)

---

## 1. Project Overview

This lab demonstrates **TensorFlow Data Validation (TFDV)** — Google's open-source library for exploring, validating, and monitoring ML datasets — applied to the **NYC Taxi Trips** dataset. The core workflow follows the standard TFDV pipeline:

1. Compute statistics on training data
2. Infer a schema from those statistics
3. Inject deliberate anomalies into the evaluation set
4. Validate the eval set against the schema to detect the anomalies
5. Fix the schema to allow legitimate edge cases
6. Re-validate to confirm resolution
7. Slice analysis to compare distributions across subgroups

The entire pipeline is **modularized into a `src/` package** — no raw logic lives directly in the notebook cells. The notebook is an orchestrator that calls into well-separated utility modules.

---

## 2. Modifications from Original Lab

| Aspect | Original Lab | This Lab |
|--------|-------------|----------|
| **Dataset** | Census Income | NYC Taxi Trips |
| **Domain** | Socioeconomic | Transportation |
| **Injected Anomalies** | Census-specific | Negative fares, zero passengers, 24hr trips, `CRYPTO` payment type, invalid `RatecodeID=99` |
| **Slicing dimensions** | `sex` and `race` | `payment_type` and `VendorID` |
| **Code structure** | Monolithic notebook | Modular `src/` package |

---

## 3. Project Structure

```
tfdv-lab1/
│
├── TFDV_Lab1.ipynb            # Main notebook (orchestrator only)
│
├── data/
│   └── nyc_taxi.csv           # 50,000 NYC Taxi trip records
│
├── schema/
│   └── schema.pbtxt           # Inferred + fixed schema (auto-generated)
│
├── slice_sample.csv           # Temporary CSV written by slicing utilities
│
└── src/
    ├── __init__.py
    ├── data_loader.py         # load_data(), split_data()
    ├── anomaly_injector.py    # inject_anomalies()
    ├── statistics_utils.py    # generate_statistics(), compare_statistics()
    ├── schema_utils.py        # infer_schema(), detect_anomalies(), fix_schema(), save_schema()
    └── slicing_utils.py       # generate_sliced_statistics(), compare_slices(), list_slices()
```

### Module Responsibilities

**`src/data_loader.py`**
Handles loading the raw CSV and performing the train/eval split. Prints a summary log confirming the number of rows and column names. The split uses `shuffle=False` to maintain temporal ordering of taxi trips, which is appropriate for time-series-like data.

**`src/anomaly_injector.py`**
Injects exactly 5 hand-crafted anomalous rows into the eval DataFrame to simulate real-world data quality failures. The injected anomalies cover five distinct failure modes: a negative fare, a zero-passenger trip, an impossibly long trip (>24 hours), an unknown payment type (`CRYPTO`), and an invalid RatecodeID (`99`).

**`src/statistics_utils.py`**
Wraps `tfdv.generate_statistics_from_dataframe()`. Accepts a named dataset label (`TRAIN`, `EVAL (dirty)`) so logs remain readable. Returns a `DatasetFeatureStatisticsList` proto for downstream use.

**`src/schema_utils.py`**
Wraps four TFDV operations: schema inference (`tfdv.infer_schema`), anomaly detection (`tfdv.validate_statistics`), schema repair (manual proto mutations with `tfdv.get_feature`), and schema persistence (`tfdv.write_schema_text`).

**`src/slicing_utils.py`**
The most complex module. Implements slice-based statistics by writing the DataFrame to a temporary CSV, running `tfdv.generate_statistics_from_csv()` with `tfdv.SlicingConfig`, and wrapping `tfdv.visualize_statistics()` for side-by-side slice comparisons.

---

## 4. Dataset

**Source:** NYC Taxi Trip Records  
**Size:** 50,000 rows loaded, split into 40,000 train / 10,000 eval  

**Features:**

| Column | Type | Description |
|--------|------|-------------|
| `VendorID` | INT | Taxi vendor (1 = Creative Mobile, 2 = VeriFone) |
| `passenger_count` | INT | Number of passengers |
| `trip_distance` | FLOAT | Trip distance in miles |
| `RatecodeID` | FLOAT/INT | Rate code (1=Standard, 2=JFK, 3=Newark, 4=Nassau, 5=Negotiated, 6=Group, 99=invalid) |
| `payment_type` | STRING | `Cash`, `Credit`, `Dispute`, `No charge` |
| `fare_amount` | FLOAT | Base fare in USD |
| `tip_amount` | FLOAT | Tip in USD |
| `tolls_amount` | FLOAT | Tolls in USD |
| `total_amount` | FLOAT | Total charge in USD |
| `trip_duration_seconds` | INT | Derived: dropoff timestamp − pickup timestamp |

`trip_duration_seconds` is a **derived feature** computed from the raw timestamp columns during loading — it is not present verbatim in the original CSV.

---

## 5. Environment Setup

### Requirements

```bash
pip install tensorflow-data-validation==1.17.0
pip install pandas
pip install apache-beam  # required for slicing functionality
```

### Directory Setup

Before running, ensure the following directories exist:

```bash
mkdir -p data schema
```

Place `nyc_taxi.csv` in the `data/` directory. The `schema/` folder is written to automatically by `save_schema()`.

---

## 6. Running the Notebook

```bash
jupyter notebook TFDV_Lab1.ipynb
```

Run all cells in order (top to bottom). Each step builds on the previous one — `schema` from Step 3 is used in Steps 6–8, and `train_df` from Step 1 is used throughout.

**Expected total runtime:** ~3–5 minutes (Beam-based slicing in Steps 9a and 9b takes the longest due to Apache Beam overhead).

---

## 7. Notebook Walkthrough — Step by Step

### Step 0: Imports & Version Check (Cell 1)

Imports all `src/` modules and prints the TFDV version. Suppresses TensorFlow C++ logs via `TF_CPP_MIN_LOG_LEVEL=3`.

**Expected output:** `TFDV Version: 1.17.0`

---

### Step 1: Load and Split Data (Cells 2–3)

```python
df = load_data('data/nyc_taxi.csv')
train_df, eval_df = split_data(df, test_size=0.2, shuffle=False)
train_df.head()
```

Loads 50,000 rows, splits 80/20 without shuffling. `train_df.describe(include='all')` in Cell 3 gives a quick sanity check on value ranges and missing data before any TFDV work begins.

**Expected log:**
```
[data_loader] Loaded 50000 rows | Columns: ['VendorID', 'passenger_count', ...]
[data_loader] Train: 40000 rows | Eval: 10000 rows
```

---

### Step 2: Generate and Visualize Training Statistics (Cells 4–5)

```python
train_stats = generate_statistics(train_df, dataset_name='TRAIN')
tfdv.visualize_statistics(train_stats)
```

`generate_statistics()` calls `tfdv.generate_statistics_from_dataframe()` which computes per-feature statistics: count, mean, std, min, max, missing rate for numerics; top values, unique count, missing rate for categoricals.

`tfdv.visualize_statistics()` renders an **interactive Facets widget** in the notebook showing distributions for every feature.

---

### Step 3: Infer Schema (Cell 6)

```python
schema = infer_schema(train_stats)
save_schema(schema, 'schema/schema.pbtxt')
```

`tfdv.infer_schema()` produces a protobuf `Schema` object capturing:
- Feature types (`INT`, `FLOAT`, `BYTES`)
- Presence constraints (required / optional)
- Value domains (numeric ranges or categorical value sets)

The schema is persisted to `schema/schema.pbtxt` using `tfdv.write_schema_text()`.

**Inferred categorical domains:**
- `RatecodeID`: `1.0`, `2.0`, `3.0`, `4.0`, `5.0`, `99.0` (note: `99.0` was already in training data)
- `payment_type`: `Cash`, `Credit`, `Dispute`, `No charge`

---

### Step 4: Inject Anomalies (Cell 7)

```python
eval_df_dirty = inject_anomalies(eval_df)
eval_df_dirty.tail(6)
```

Five rows are appended to the eval set, each representing a different failure mode:

| Anomaly Type | Feature Affected | Injected Value |
|-------------|-----------------|---------------|
| Negative fare | `fare_amount` | `-70` |
| Zero passengers | `passenger_count` | `0` |
| 24-hour trip | `trip_duration_seconds` | `86400` |
| Unknown payment | `payment_type` | `"CRYPTO"` |
| Invalid rate code | `RatecodeID` | `99` |

The eval set grows from 10,000 to **10,005 rows**.

---

### Step 5: Compare Train vs Eval Statistics (Cells 8–9)

```python
eval_stats_dirty = generate_statistics(eval_df_dirty, dataset_name='EVAL (dirty)')
tfdv.visualize_statistics(
    lhs_statistics=eval_stats_dirty,
    rhs_statistics=train_stats,
    lhs_name='EVAL',
    rhs_name='TRAIN'
)
```

Side-by-side Facets visualization. The injected anomalies are too small (5 out of 10,005) to visually dominate the distributions, but extreme outliers (e.g., `fare_amount = -70`) may be visible in the tail of the histogram.

---

### Step 6: Detect Anomalies (Cell 10)

```python
anomalies = detect_anomalies(eval_stats_dirty, schema)
```

`tfdv.validate_statistics(eval_stats_dirty, schema)` produces an `Anomalies` proto. The display should flag:

| Feature | Anomaly Type | Description |
|---------|-------------|-------------|
| `fare_amount` | `INT_TYPE_SMALL_INT` | Value `-70` below minimum |
| `passenger_count` | Range violation | Value `0` below expected minimum |
| `payment_type` | `ENUM_TYPE_UNEXPECTED_STRING_VALUES` | `CRYPTO` not in schema domain |
| `RatecodeID` | `ENUM_TYPE_UNEXPECTED_STRING_VALUES` | `99` seen in `<1%` of examples |
| `trip_duration_seconds` | `INT_TYPE_BIG_INT` | Value `86400` above expected range |

---

### Step 7: Fix Schema (Cell 11)

```python
schema = fix_schema(schema)
save_schema(schema, 'schema/schema.pbtxt')
```

The `fix_schema()` function makes targeted repairs:
- Sets `fare_amount` minimum to `0.0` (negative fares are impossible)
- Sets `passenger_count` range to `[1, 6]`
- Sets `trip_duration_seconds` range to `[30, 10800]` (30 seconds to 3 hours)
- Relaxes `payment_type` domain to require only 90% match (allows rare/new values)
- Relaxes `RatecodeID` domain to require only 90% match

---

### Step 8: Re-validate After Fix (Cell 12)

```python
updated_anomalies = detect_anomalies(eval_stats_dirty, schema)
```

After fixing, the re-validation still shows residual anomalies for the truly invalid values that the schema fix intentionally did **not** suppress:
- `fare_amount = -70` (still below 0 — invalid)
- `trip_duration_seconds = 86400` (still flagged as unexpectedly large)
- `RatecodeID` domain type mismatch (INT vs FLOAT)

This is **expected behavior** — the schema fix only resolves edge-case valid values, not genuinely bad data.

---

### Step 9a: Slice by Payment Type (Cells 13–14)

```python
payment_slices = generate_sliced_statistics(
    df=train_df,
    schema=schema,
    slice_features={'payment_type': None},
    csv_path='slice_sample.csv'
)
list_slices(payment_slices)
compare_slices(payment_slices, lhs_name='payment_type_Credit', rhs_name='payment_type_Cash')
```

Generates per-slice statistics using Apache Beam. Slices produced:
- `All Examples`
- `payment_type_Cash`
- `payment_type_Credit`
- `payment_type_Dispute`
- `payment_type_No charge`

Side-by-side comparison of `Credit` vs `Cash` visualizes differences in `tip_amount` (Credit card trips typically have higher tips), `fare_amount`, and `trip_distance`.

---

### Step 9b: Slice by VendorID (Cells 15–16)

```python
vendor_slices = generate_sliced_statistics(
    df=train_df,
    schema=schema,
    slice_features={'VendorID': None},
    csv_path='slice_sample.csv'
)
compare_slices(vendor_slices, lhs_name='VendorID_1', rhs_name='VendorID_2')
```

Slices produced:
- `All Examples`
- `VendorID_1` (Creative Mobile Technologies)
- `VendorID_2` (VeriFone Inc.)

Comparison checks for vendor reporting differences in `trip_distance`, `fare_amount`, and `passenger_count` distributions — important for detecting vendor-level data quality inconsistencies.

---
