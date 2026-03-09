from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama_3_1_8b_instruct',
        path='meta-llama/Llama-3.1-8B-Instruct',
        max_out_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=3),
    )
]
