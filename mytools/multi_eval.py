from mmengine.config import read_base

with read_base():
    # 数据集和模型导入区域
    from opencompass.configs.models.mymodels.my_GPT2_Indo_Instruct_Tuned import models as GPT2_Indo_Instruct_Tuned_models
    from opencompass.configs.models.mymodels.my_SambaLingo_Thai_Chat import models as SambaLingo_Thai_Chat_models
    from opencompass.configs.datasets.mydatasets.tydiqa_gen_978d2a import tydiqa_datasets
datasets = tydiqa_datasets
models = SambaLingo_Thai_Chat_models + GPT2_Indo_Instruct_Tuned_models
