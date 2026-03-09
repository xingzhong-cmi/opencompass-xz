from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='distil-medium.en',
        path='distil-whisper/distil-medium.en',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
