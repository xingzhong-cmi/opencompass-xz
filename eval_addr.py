from mmengine.config import read_base

with read_base():
    # 导入地址脱敏数据集配置
    from opencompass.configs.datasets.mydatasets.my_addr import addr_desensitize_datasets
    # 导入模型配置（可替换为你需要测试的模型，参考原代码格式）
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import models as hf_qwen2_1_5b_instruct_models
    # 若需测试其他模型，取消注释下方并替换
    # from opencompass.configs.models.mymodels.my_Qwen2_5_7B_Instruct import models as qwen2_5_7b_instruct_models

# 指定测试数据集和模型
datasets = addr_desensitize_datasets
# 可替换为其他模型（如 qwen2_5_7b_instruct_models）
models = hf_qwen2_1_5b_instruct_models
