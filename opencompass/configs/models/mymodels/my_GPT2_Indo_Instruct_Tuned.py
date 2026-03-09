from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='GPT2-Indo-Instruct-Tuned',
        path='IzzulGod/GPT2-Indo-Instruct-Tuned',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
