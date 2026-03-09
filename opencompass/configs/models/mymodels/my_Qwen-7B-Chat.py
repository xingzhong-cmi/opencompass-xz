from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='seallms_v3_7b_chat',
        path='SeaLLMs/SeaLLMs-v3-7B-Chat',
        max_out_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=3),
    )
]
