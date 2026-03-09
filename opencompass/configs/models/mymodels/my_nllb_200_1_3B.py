from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='nllb-200-1.3B',
        path='facebook/nllb-200-1.3B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
