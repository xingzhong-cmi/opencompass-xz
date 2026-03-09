from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mydatasets.flores_gen_aad4fd_dan import \
        flores_datasets
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import \
        models as hf_qwen2_1_5b_instruct_models
    from opencompass.configs.models.mymodels.my_jiutian_api import models as jiutian_api_models
datasets = flores_datasets
models = jiutian_api_models
