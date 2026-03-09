from opencompass.models.my_huggingface import HuggingFace

models = [
    dict(
        type=HuggingFace,
        abbr='m2m100_1.2B',
        path='facebook/m2m100_1.2B',
        max_seq_len=2048,
        max_out_len=512,  # 建议设为512（足够覆盖长句，避免过度冗余）
        batch_size=8,
        tokenizer_path='facebook/m2m100_1.2B',
        # 补充关键生成参数
        #generation_kwargs=dict(
         #   forced_bos_token_id=128090,  # 泰语语言ID（正确）
          #  num_beams=5,  # 提升翻译质量
            #max_new_tokens=512,  # 控制单条输出的最大token数（与max_out_len呼应）
            #no_repeat_ngram_size=3,  # 抑制重复短语（如连续"แสดงให้เห็นว่า"）
            #repetition_penalty=1.2,  # 额外惩罚重复内容
            #early_stopping=True,  # 语义完整时提前停止
        #),
        generation_kwargs=dict(
            # 动态获取目标语言 ID（避免硬编码，与测试脚本逻辑一致）
            forced_bos_token_id=128090,
            num_beams=4,  # 与测试脚本的 num_beams=4 保持一致
            #max_new_tokens=512,  # 对应测试脚本的 max_length=100（但扩展为 512 适配长句）
            no_repeat_ngram_size=3,  # 抑制重复（如连续短语）
            repetition_penalty=1.5,  # 增强重复惩罚力度
            early_stopping=True  # 语义完整时提前停止
        ),
        run_cfg=dict(num_gpus=2)
    )
]
