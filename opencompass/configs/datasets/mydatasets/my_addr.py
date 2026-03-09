from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import AddrDesensitizeDataset, AddrDesensitizeEvaluator

# 1. 读取配置：指定输入输出字段
addr_desensitize_reader_cfg = dict(
    input_columns=['original_text'],  # 输入字段：原始文本
    output_column='gold_text'  # 输出字段：脱敏标准答案
)

# 2. 推理配置：明确且强约束的脱敏指令 + 少量示例，防止模型改写非地址内容
addr_desensitize_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=(
            "你是信息安全助手。请严格按以下规则处理文本：\n"
            "1) 仅将文本中的具体地址信息替换为「[隐藏地址]」。\n"
            "2) 非地址内容（如人名、时间、电话、价格、描述等）必须与原文完全一致，不得增删或改写。\n"
            "3) 如果存在多个地址，所有地址都必须替换为「[隐藏地址]」。\n"
            "4) 保持原文的标点、格式不变。\n"
            "5) 只输出处理后的文本，不要任何解释或额外说明。\n"
            "\n"
            "示例1：\n"
            "原文：我公司的办公地址在北京市朝阳区建国路88号浦软大厦15层，联系电话是13800138000。\n"
            "输出：我公司的办公地址在[隐藏地址]，联系电话是13800138000。\n"
            "\n"
            "示例2：\n"
            "原文：这家医院的新院区在武汉市洪山区珞喻路1037号，老院区在武昌区解放路234号。\n"
            "输出：这家医院的新院区在[隐藏地址]，老院区在[隐藏地址]。\n"
            "\n"
            "示例3：\n"
            "原文：从西安市雁塔区长安南路陕西师范大学出发，打车20分钟可到西安大雁塔景区。\n"
            "输出：从[隐藏地址]出发，打车20分钟可到[隐藏地址]。\n"
            "\n"
            "现在开始处理：\n"
            "原文：{original_text}\n"
            "输出："
        ),
    ),
    retriever=dict(type=ZeroRetriever),  # 零样本检索
    inferencer=dict(type=GenInferencer, max_out_len=512),  # 生成式推理器
)

# 3. 评估配置：使用自定义评估器（计算两个指标）
addr_desensitize_eval_cfg = dict(
    evaluator=dict(type=AddrDesensitizeEvaluator),
    ds_split='dev',  # 验证集划分
    ds_column=['address_info', 'original_text'],  # 评估所需额外字段
)

# 4. 构建数据集列表
addr_desensitize_datasets = [
    dict(
        abbr='addr_desensitize_cn',  # 数据集简称
        type=AddrDesensitizeDataset,  # 数据集类
        path='/workspace/data/mydata/addr',  # 数据集根路径
        reader_cfg=addr_desensitize_reader_cfg,
        infer_cfg=addr_desensitize_infer_cfg,
        eval_cfg=addr_desensitize_eval_cfg
    )
]
