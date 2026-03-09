from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mydatasets.flores_gen_806ede import \
        flores_datasets
    from opencompass.configs.datasets.demo.demo_math_chat_gen import \
        math_datasets
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import \
        models as hf_qwen2_1_5b_instruct_models
    from opencompass.configs.models.mymodels.my_deepseekr1_1_5b import models as hf_deepseekv1_1_5b_models

datasets = flores_datasets + math_datasets
models = hf_qwen2_1_5b_instruct_models + hf_deepseekv1_1_5b_models
