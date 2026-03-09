from mmengine.config import read_base

with read_base():
    # 数据集和模型导入区域
    from opencompass.configs.datasets.mydatasets.flores_gen_aad4fd_dan import flores_datasets

    from opencompass.configs.models.mymodels.my_Qwen2_1_5B_Instruct import models as Qwen2_1_5B_Instruct_models

datasets = flores_datasets
models = Qwen2_1_5B_Instruct_models
