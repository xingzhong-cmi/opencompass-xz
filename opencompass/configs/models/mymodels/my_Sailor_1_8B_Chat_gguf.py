from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Sailor-1.8B-Chat-gguf',  # 保留原始名称作为缩写
        path='sail/Sailor-1.8B-Chat-gguf',     # 确保这里没有URL前缀
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
