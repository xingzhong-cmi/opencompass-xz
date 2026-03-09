from opencompass.models.my_huggingface import HuggingFace

models = [
    dict(
        type=HuggingFace,
        abbr='nllb-200-3.3B',
        path='facebook/nllb-200-3.3B',  # 模型路径正确
        max_seq_len=2048,  # 适配长输入
        max_out_len=512,  # 输出长度（平衡完整性和效率）
        batch_size=1,  # 3.3B 参数模型，8 可能超出 4GPU 显存，建议先设 4
        tokenizer_path='facebook/nllb-200-3.3B',  # 显式指定分词器路径
        # 关键：翻译生成参数
        generation_kwargs=dict(
            forced_bos_token_id=33180,  # 泰语的 NLLB 语言 ID（必须根据目标语言调整）
            num_beams=5,  # 提升翻译质量
            #max_new_tokens=512,  # 与 max_out_len 一致，避免截断
            #no_repeat_ngram_size=3,  # 抑制重复短语
            #repetition_penalty=1.2,  # 进一步减少重复
            #early_stopping=True,  # 语义完整时提前停止
        ),
        tokenizer_kwargs=dict(
            src_lang='ind_Latn',  # 印尼语
            tgt_lang='tha_Thai',  # 泰语
        ),
        run_cfg=dict(num_gpus=4),  # 4GPU 合理
    )
]
