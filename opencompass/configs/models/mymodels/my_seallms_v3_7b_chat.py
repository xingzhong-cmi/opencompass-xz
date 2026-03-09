from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='SeaLLMs-v3-7B-Chat',
        path='SeaLLMs/SeaLLMs-v3-7B-Chat',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
