from opencompass.models import JiutianAPI

models = [
    dict(
        type=JiutianAPI,
        abbr='jiutian-lan-comv3-api',
        path='jiutian-lan-comv3',  # 九天模型名称（与API请求中的"model"字段一致）
        key='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhcGlfa2V5IjoiNjc2ZTAzYzJkNzNmYTg1ZjIyZmY4OGE2IiwiZXhwIjoxNzcwNjAzMzk1LCJ0aW1lc3RhbXAiOjE3NjI4MjczOTV9.Subdy4e0xvQPIyUUKoL4iX036w36SdSlm3Y39RGrEa4',  # 九天API密钥
        url='https://jiutian.10086.cn/largemodel/moma/api/v3/chat/completions',  # 九天API接口地址
        max_out_len=1024,  # 最大输出长度（对应API的"max_tokens"参数）
        batch_size=8,  # 并发批次大小（根据API限流策略调整）
        query_per_second=2,  # 每秒最大查询数（避免触发API限流）
        retry=3,  # API调用失败后的重试次数
        max_seq_len=2048,  # 模型支持的最大序列长度（根据九天模型规格调整）
        run_cfg=dict(num_gpus=0),  # API模型无需本地GPU，固定设为0
    )
]
