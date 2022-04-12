
from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator

with DAG(
    'weekly_workflow',
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        'depends_on_past': False,
        'email': ['pengfei_ji@general.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'sla': timedelta(hours=2),
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function,
        # 'on_success_callback': some_other_function,
        # 'on_retry_callback': another_function,
        # 'sla_miss_callback': yet_another_function,
        # 'trigger_rule': 'all_success'
    },
    description='A simple Daily DAG',
    schedule_interval=timedelta(days=7),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['weekly_workflow'],
) as dag:

    t1 = BashOperator(
        task_id='pretrain_model',
        bash_command='''sbatch -W /b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/pretrain.sh "{{ execution_date }}"''',
    )
    t2 = BashOperator(
        task_id='finetune_model',
        bash_command='''sbatch -W /b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/finetune.sh "{{ execution_date }}"''',
    )
    t3 = BashOperator(
        task_id='export_model',
        bash_command='''python /b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/model_exporter.py "{{ execution_date }}"''',
    )
    t1 >> t2 >> t3