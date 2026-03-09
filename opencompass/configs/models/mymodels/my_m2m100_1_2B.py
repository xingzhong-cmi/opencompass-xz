from opencompass.models.my_huggingface import HuggingFace

models = [
    dict(
        type=HuggingFace,
        abbr='m2m100_1.2B',
        path='facebook/m2m100_1.2B',
        max_seq_len=2048,  # 输入序列长度（已优化）
        max_out_len=2048,  # 全局输出长度限制（已优化）
        batch_size=4,  # 适配显存（已优化）
        tokenizer_path='facebook/m2m100_1.2B',
        tokenizer_kwargs=dict(
            src_lang='id',  # 印尼语
            tgt_lang='th',  # 泰语
        ),
        generation_kwargs=dict(
            forced_bos_token_id=128090,  # 锁定泰语输出
            num_beams=5  # 提升翻译质量
        ),
        run_cfg=dict(num_gpus=2),
    )
]
