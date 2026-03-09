from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Qwen2.5-Omni-7B',
        path='Qwen/Qwen2.5-Omni-7B',  # 确保这里没有URL前缀
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
