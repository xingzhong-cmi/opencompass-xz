from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.tydiqa.tydiqa_gen_978d2a import tydiqa_datasets
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import models as hf_qwen2_1_5b_instruct_models
    from opencompass.configs.models.mymodels.my_deepseek_api import models as deepseek_api_models
    from opencompass.configs.models.mymodels.my_jiutian_api import models as jiutian_api_models
    from opencompass.configs.models.mymodels.my_deepseekr1_1_5b import models as deepseek_r1_models
    from opencompass.configs.models.mymodels.my_Qwen2_5_7B_Instruct import models as qwen2_5_7b_instruct_models
    from opencompass.configs.models.mymodels.my_m2m100_1_2B import models as m2m100_1_2b_models

datasets = tydiqa_datasets
models = jiutian_api_models + deepseek_api_models + deepseek_r1_models + hf_qwen2_1_5b_instruct_models + qwen2_5_7b_instruct_models + m2m100_1_2b_models
