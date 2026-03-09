from opencompass.models.huggingface_trans import HuggingFace
from opencompass.models import HuggingFaceBaseModel
models = [
    dict(
        type=HuggingFace,
        abbr='m2m100_1.2B',
        path='facebook/m2m100_1.2B',
        max_out_len=1024,
        batch_size=8,
        tokenizer_path='facebook/m2m100_1.2B',
        tokenizer_kwargs=dict(
            src_lang='id',  #  ~M 尼语 | ~G ~G~F代 | ~A
            tgt_lang='th',  # 泰语 | ~G ~G~F代 | ~A
        ),
        generation_kwargs=dict(
            forced_bos_token_id=128090  # 直接使用泰语的ID
        ),
        run_cfg=dict(num_gpus=1),
    )        
]
