from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Meta-Llama-3-70B-Instruct',
        path='meta-llama/Meta-Llama-3-70B-Instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
