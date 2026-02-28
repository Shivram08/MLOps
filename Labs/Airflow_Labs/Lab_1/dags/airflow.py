from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow import configuration as conf

from src.lab import load_data, preprocess_data, train_model, evaluate_model

# Enable pickle support for XCom so serialized data can pass between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# ──────────────────────────────────────────────
# Default Arguments
# ──────────────────────────────────────────────
default_args = {
    'owner': 'shiv',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ──────────────────────────────────────────────
# DAG Definition
# ──────────────────────────────────────────────
dag = DAG(
    'heart_disease_classification_dag',
    default_args=default_args,
    description='Binary classification pipeline for Heart Disease prediction using Random Forest',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
)

# ──────────────────────────────────────────────
# Task 1: Load Data
# ──────────────────────────────────────────────
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# ──────────────────────────────────────────────
# Task 2: Preprocess Data
# ──────────────────────────────────────────────
preprocess_data_task = PythonOperator(
    task_id='preprocess_data_task',
    python_callable=preprocess_data,
    op_args=[load_data_task.output],
    dag=dag,
)

# ──────────────────────────────────────────────
# Task 3: Train Model
# ──────────────────────────────────────────────
train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    op_args=[preprocess_data_task.output, 'heart_disease_rf.pkl'],
    provide_context=True,
    dag=dag,
)

# ──────────────────────────────────────────────
# Task 4: Evaluate Model
# ──────────────────────────────────────────────
evaluate_model_task = PythonOperator(
    task_id='evaluate_model_task',
    python_callable=evaluate_model,
    op_args=[train_model_task.output],
    dag=dag,
)

# ──────────────────────────────────────────────
# Task Dependencies
# ──────────────────────────────────────────────
load_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task

# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    dag.cli()