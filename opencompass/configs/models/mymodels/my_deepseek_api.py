from opencompass.models import DeepseekAPI

models = [
    dict(
        type=DeepseekAPI,
        abbr='DeepSeek-V3.2-Exp_api',
        path='deepseek-chat',  # 模型名改为API支持的"deepseek-chat"（与curl一致）
        key='sk-428249a366a7472f9403396d3c298f10',  # 你的API密钥（与curl一致）
        url='https://api.deepseek.com/chat/completions',  # 修正URL（去掉/v1）
        max_out_len=1024,  # 最大输出长度（与API支持的max_tokens一致）
        batch_size=8,  # 并发批次大小（需参考API的并发限制，避免限流）
        query_per_second=2,  # 每秒最大查询数（根据API限流策略调整，默认2）
        retry=3,  # 调用失败时的重试次数（默认2，可适当增加）
        system_prompt='',  # 可选：全局系统提示（如需要固定前缀可填写）
        max_seq_len=2048,  # API支持的最大序列长度（根据模型实际支持值调整）
        run_cfg=dict(num_gpus=0),  # API模型无需GPU，固定设为0
    )
]
