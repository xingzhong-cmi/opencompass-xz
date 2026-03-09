from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mydatasets.tydiqa_gen_978d2a import tydiqa_datasets
    from opencompass.configs.models.mymodels.my_Qwen2_5_7B_Instruct import models as qwen2_5_7b_instruct_models
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import models as hf_qwen2_1_5b_instruct_models
    #from opencompass.configs.models.mymodels.my_deepseek_api import models as deepseek_api_models

    from opencompass.configs.models.mymodels.my_Qwen3_VL_8B_Instruct import models as Qwen3_VL_8B_Instruct
    from opencompass.configs.models.mymodels.my_Qwen3_32B import models as Qwen3_32B
    from opencompass.configs.models.mymodels.my_Qwen3_8B import models as Qwen3_8B
    from opencompass.configs.models.mymodels.my_Meta_Llama_3_70B_Instruct import models as Meta_Llama_3_70B_Instruct
    from opencompass.configs.models.mymodels.my_Meta_Llama_3_8B_Instruct import models as Meta_Llama_3_8B_Instruct
datasets = tydiqa_datasets
models = Qwen3_VL_8B_Instruct + Qwen3_32B + Qwen3_8B + Meta_Llama_3_70B_Instruct + Meta_Llama_3_8B_Instruct
