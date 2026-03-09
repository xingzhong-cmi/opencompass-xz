from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='bert-base-indonesian-tydiqa',
        path='cahya/bert-base-indonesian-tydiqa',  # 确保这里没有URL前缀
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
