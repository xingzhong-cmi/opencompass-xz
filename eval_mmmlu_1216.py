from mmengine.config import read_base

with read_base():
    #from opencompass.configs.datasets.mydatasets.my_pmmeval_gen import PMMEval_datasets
    from opencompass.configs.datasets.mydatasets.mmmlu_lite_gen_c51a84 import mmmlu_lite_datasets
    #from opencompass.configs.datasets.mydatasets.tydiqa_gen_978d2a import tydiqa_datasets
    from opencompass.configs.models.mymodels.my_Qwen2_5_7B_Instruct import models as qwen2_5_7b_instruct_models
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import models as hf_qwen2_1_5b_instruct_models
    #from opencompass.configs.models.mymodels.my_deepseek_api import models as deepseek_api_models
    from opencompass.configs.models.mymodels.my_jiutian_api import models as jiutian_api_models
    from opencompass.configs.models.mymodels.my_Qwen3_VL_8B_Instruct import models as Qwen3_VL_8B_Instruct
    from opencompass.configs.models.mymodels.my_Qwen3_32B import models as Qwen3_32B
    from opencompass.configs.models.mymodels.my_Qwen3_8B import models as Qwen3_8B
    from opencompass.configs.models.mymodels.my_Meta_Llama_3_70B_Instruct import models as Meta_Llama_3_70B_Instruct
    from opencompass.configs.models.mymodels.my_Meta_Llama_3_8B_Instruct import models as Meta_Llama_3_8B_Instruct
datasets = mmmlu_lite_datasets
models = Qwen3_8B + qwen2_5_7b_instruct_models + hf_qwen2_1_5b_instruct_models
