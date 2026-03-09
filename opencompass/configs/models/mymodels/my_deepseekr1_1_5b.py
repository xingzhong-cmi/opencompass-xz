from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='DeepSeek-R1-Distill-Qwen-1.5B',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
