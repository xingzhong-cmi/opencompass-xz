from mmengine.config import read_base

with read_base():
    #from opencompass.configs.datasets.mydatasets.flores_gen_806ede import \
     #   flores_datasets
    from opencompass.configs.datasets.mydatasets.flores_gen_trans import flores_datasets
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import \
        models as hf_qwen2_1_5b_instruct_models
    #from opencompass.configs.models.mymodels.my_trans_model import models as hf_m2m100_1_2B_models
    from opencompass.configs.models.mymodels.my_m2m100_1_2B import models as hf_m2m100_1_2B_models
    from opencompass.configs.models.mymodels.my_nllb_200_3_3B import models as nllb_200_3_3B_models

datasets = flores_datasets
#models = hf_qwen2_1_5b_instruct_models + hf_m2m100_1_2B_models + nllb_200_3_3B_models
#models = hf_m2m100_1_2B_models + nllb_200_3_3B_models
models = nllb_200_3_3B_models
