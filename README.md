## Running Instructions

This is the pretrain and finetune insturctions for sta-v4 model. I will mainly explain how to use this repo to train sta-v4 model on different features.

###Step One: Generate the features.

The model fetch training data stored in a fixed format, which are generated using `FactorDAO` object in `./scripts/workflow_utils/factor_dao.py`. the sample code of how to use this object is below.

```python:
base_path = '/b/home/pengfei_ji/factor_dbs/'
factor_dao = FactorDAO(base_path)
factor_dao.register_factor_info(factor_name='production_factors',
                                    group_type=GroupType.NEW_LEVEL,
                                    store_granularity=StoreGranularity.DAY_SKEY_FILE,
									save_format='pkl')
task_df = factor_dao.read_factor_by_skey_and_day('normed_multi_label', skey, day,
version='v1')
factor_dao.save_factors(data_df=task_df, factor_group='new_task_factors',
                                         skey=skey, day=day, version='v9')
```

###Step Two:  Set pretrain configuration
most of work can be done by modifying `./scripts/weekly_workflow/train_configs/pretrain_config.yaml`
and `./scripts/weekly_workflow/model_pretrainer.py`

For pretain_config.yaml,  it configures the features and how to fetch them.

- base_path: the feature path to init FactorDAO object
- factor_group: the name of the set of features
- factor_version: the version of factor group
- model_save_path: the path to save the pretrained model

For model_pretrainer.py,  there is a global variable `feat_map`,  you should specify the set of features that used for training,  then simply run the following command

```bash:
sbatch pretrain.sh 20200901
```
###Step Three: Finetune the model and export alpha

After pretraining, run the following command to start finetune, the date variable should be the same as pretrain

```bash:
sbatch finetune.sh 20200901
```
Also, we should keep the following stuff the same as pretrain prodcudure as well

- base_path: the feature path to init FactorDAO object
- factor_group: the name of the set of features
- factor_version: the version of factor group
- model_save_path: the path to save the pretrained model
- feat_map: the vairable to configure feature names for training.

finally we can export alpha by runinng `/scripts/workflow_utils/gen.sh`, the details are in  `alpha_gen.py`,  the code is very straighforward.




