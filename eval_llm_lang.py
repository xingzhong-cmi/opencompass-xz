from mmengine.config import read_base

with read_base():
    #from opencompass.configs.datasets.mydatasets.flores_gen_806ede import \
    #    flores_datasets
    from opencompass.configs.datasets.mydatasets.tydiqa_gen_978d2a import tydiqa_datasets
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import \
        models as hf_qwen2_1_5b_instruct_models
    from opencompass.configs.models.mymodels.my_llama_3_1_8b_instruct import models as hf_llama_8b_models
    from opencompass.configs.models.mymodels.my_seallms_v3_7b_chat import models as hf_seallms_7b_models    

    from opencompass.configs.models.mymodels.my_Sailor_1_8B_Chat import models as Sailor_1_8B_Chat_models
    from opencompass.configs.models.mymodels.my_Qwen2_5_Omni_7B import models as Qwen2_5_Omni_7B_models
    from opencompass.configs.models.mymodels.my_Sailor_7B_Chat import models as Sailor_7B_Chat_models

datasets = tydiqa_datasets
models = hf_qwen2_1_5b_instruct_models + hf_llama_8b_models + hf_seallms_7b_models + Sailor_1_8B_Chat_models + Qwen2_5_Omni_7B_models + Sailor_7B_Chat_models
