from mmengine.config import read_base

with read_base():
    # 导入丹麦语地址脱敏数据集配置
    from opencompass.configs.datasets.mydatasets.my_addr_da import addr_desensitize_da_datasets

    from opencompass.configs.models.mymodels.my_Qwen2_5_7B_Instruct import models as qwen2_5_7b_instruct_models
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import models as hf_qwen2_1_5b_instruct_models
    #from opencompass.configs.models.mymodels.my_deepseek_api import models as deepseek_api_models
    from opencompass.configs.models.mymodels.my_jiutian_api import models as jiutian_api_models
    from opencompass.configs.models.mymodels.my_Qwen3_VL_8B_Instruct import models as Qwen3_VL_8B_Instruct
    from opencompass.configs.models.mymodels.my_Qwen3_32B import models as Qwen3_32B
    from opencompass.configs.models.mymodels.my_Qwen3_8B import models as Qwen3_8B
    from opencompass.configs.models.mymodels.my_Meta_Llama_3_70B_Instruct import models as Meta_Llama_3_70B_Instruct
    from opencompass.configs.models.mymodels.my_Meta_Llama_3_8B_Instruct import models as Meta_Llama_3_8B_Instruct
# 指定测试数据集和模型
datasets = addr_desensitize_da_datasets
# 可替换为其他模型（如 qwen2_5_7b_instruct_models）
models = hf_qwen2_1_5b_instruct_models + qwen2_5_7b_instruct_models + jiutian_api_models + Qwen3_32B + Qwen3_8B
