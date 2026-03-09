from mmengine.config import read_base

with read_base():
    # 数据集和模型导入区域
    
    from opencompass.configs.datasets.mydatasets.tydiqa_gen_978d2a import tydiqa_datasets
    from opencompass.configs.models.mymodels.my_SeaLLMs_v3_7B_Chat import models as SeaLLMs_v3_7B_Chat_models
    from opencompass.configs.models.mymodels.my_Qwen2_1_5B_Instruct import models as Qwen2_1_5B_Instruct_models

datasets = tydiqa_datasets
models = Qwen2_1_5B_Instruct_models + SeaLLMs_v3_7B_Chat_models
