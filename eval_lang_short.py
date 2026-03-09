from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mydatasets.flores_gen_806ede import \
        flores_datasets
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import \
        models as hf_qwen2_1_5b_instruct_models

datasets = flores_datasets
models = hf_qwen2_1_5b_instruct_models
