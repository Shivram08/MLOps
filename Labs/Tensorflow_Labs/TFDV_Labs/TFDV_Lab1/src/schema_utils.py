import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf import text_format
import os


def infer_schema(train_stats):
    """
    Infers a schema from training dataset statistics.

    Args:
        train_stats: DatasetFeatureStatisticsList proto from training data.

    Returns:
        Schema proto.
    """
    schema = tfdv.infer_schema(statistics=train_stats)
    print("[schema_utils] Schema inferred from training statistics.")
    tfdv.display_schema(schema)
    return schema


def detect_anomalies(eval_stats, schema):
    """
    Validates eval statistics against the reference schema
    and returns detected anomalies.

    Args:
        eval_stats: DatasetFeatureStatisticsList proto from eval data.
        schema:     Reference schema proto (inferred from training data).

    Returns:
        Anomalies proto.
    """
    anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)
    tfdv.display_anomalies(anomalies)
    return anomalies


def fix_schema(schema):
    """
    Fixes the schema to handle known anomalies in the NYC Taxi eval set:

    1. Sets valid range for fare_amount       → min = 0.0
    2. Sets valid range for passenger_count   → min = 1, max = 6
    3. Sets valid range for trip_duration     → min = 30, max = 10800 (3hrs)
    4. Relaxes payment_type domain            → allows 90% domain match
    5. Relaxes RatecodeID domain              → allows 90% domain match

    Args:
        schema: Schema proto to modify in place.

    Returns:
        Modified schema proto.
    """
    # Fix 1: fare_amount must be non-negative
    tfdv.set_domain(
        schema, 'fare_amount',
        schema_pb2.FloatDomain(name='fare_amount', min=0.0)
    )
    print("[schema_utils] Set fare_amount min=0.0")

    # Fix 2: passenger_count must be between 1 and 6
    tfdv.set_domain(
        schema, 'passenger_count',
        schema_pb2.IntDomain(name='passenger_count', min=1, max=6)
    )
    print("[schema_utils] Set passenger_count min=1, max=6")

    # Fix 3: trip_duration_seconds — 30s minimum, 3hr maximum
    tfdv.set_domain(
        schema, 'trip_duration_seconds',
        schema_pb2.IntDomain(name='trip_duration_seconds', min=30, max=10800)
    )
    print("[schema_utils] Set trip_duration_seconds min=30, max=10800")

    # Fix 4: Relax payment_type to tolerate unseen values
    payment_feature = tfdv.get_feature(schema, 'payment_type')
    payment_feature.distribution_constraints.min_domain_mass = 0.9
    print("[schema_utils] Relaxed payment_type domain to 90% match")

    # Fix 5: Relax RatecodeID to tolerate unseen values
    ratecode_feature = tfdv.get_feature(schema, 'RatecodeID')
    ratecode_feature.distribution_constraints.min_domain_mass = 0.9
    print("[schema_utils] Relaxed RatecodeID domain to 90% match")

    tfdv.display_schema(schema)
    return schema


def save_schema(schema, path: str = 'schema/schema.pbtxt'):
    """
    Saves the schema proto to disk as a text protobuf file.

    Args:
        schema: Schema proto to save.
        path:   Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text_format.MessageToString(schema))
    print(f"[schema_utils] Schema saved to {path}")


def load_schema(path: str = 'schema/schema.pbtxt'):
    """
    Loads a schema proto from a saved text protobuf file.

    Args:
        path: Path to the saved schema file.

    Returns:
        Schema proto.
    """
    schema = schema_pb2.Schema()
    with open(path, 'r') as f:
        text_format.Parse(f.read(), schema)
    print(f"[schema_utils] Schema loaded from {path}")
    return schema