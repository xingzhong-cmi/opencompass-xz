from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import AddrDesensitizeDatasetDA, AddrDesensitizeEvaluatorDA

# 1. 读取配置：指定输入输出字段
addr_desensitize_reader_cfg = dict(
    input_columns=['original_text'],  # 输入字段：原始文本
    output_column='gold_text'         # 输出字段：脱敏标准答案
)

# 2. 推理配置：丹麦语强约束脱敏指令 + 少量示例，防止模型改写非地址内容
addr_desensitize_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=(
            "Du er en informationssikkerhedsassistent. Følg reglerne nøje:\n"
            "1) Erstat kun konkrete adresseoplysninger i teksten med \"[SKJULT ADRESSE]\".\n"
            "2) Alt ikke-adresseindhold (navne, tidspunkter, telefonnumre, priser, beskrivelser osv.) "
            "skal være fuldstændig identisk med originalteksten. Ingen tilføjelser eller omskrivninger.\n"
            "3) Hvis der er flere adresser, skal alle erstattes med \"[SKJULT ADRESSE]\".\n"
            "4) Bevar original tegnsætning og formatering.\n"
            "5) Returnér kun den behandlede tekst, ingen forklaringer.\n"
            "\n"
            "Eksempel 1:\n"
            "Original: Vores kontor ligger på Østerbrogade 150, 5. sal, 2100 København Ø. Telefon: 12345678.\n"
            "Output: Vores kontor ligger på [SKJULT ADRESSE]. Telefon: 12345678.\n"
            "\n"
            "Eksempel 2:\n"
            "Original: Hospitalets nye afdeling ligger på Nørrebrogade 45, den gamle afdeling på Vester Allé 12.\n"
            "Output: Hospitalets nye afdeling ligger på [SKJULT ADRESSE], den gamle afdeling på [SKJULT ADRESSE].\n"
            "\n"
            "Eksempel 3:\n"
            "Original: Fra Aarhus Universitet på Nordre Ringgade tager det 20 minutter i taxi til Den Gamle By.\n"
            "Output: Fra [SKJULT ADRESSE] tager det 20 minutter i taxi til [SKJULT ADRESSE].\n"
            "\n"
            "Start nu:\n"
            "Original: {original_text}\n"
            "Output:"
        ),
    ),
    retriever=dict(type=ZeroRetriever),                 # 零样本检索
    inferencer=dict(type=GenInferencer, max_out_len=512)  # 生成式推理器
)

# 3. 评估配置：使用丹麦语评估器（计算两个指标）
addr_desensitize_eval_cfg = dict(
    evaluator=dict(type=AddrDesensitizeEvaluatorDA),
    ds_split='da',                           # 验证集划分
    ds_column=['address_info', 'original_text']  # 评估所需额外字段
)

# 4. 构建数据集列表（请确保数据放在 /workspace/data/mydata/addr_da/addr_desensitize_dev.jsonl）
addr_desensitize_da_datasets = [
    dict(
        abbr='addr_desensitize_da',            # 数据集简称
        type=AddrDesensitizeDatasetDA,         # 数据集类（丹麦语版本）
        path='/workspace/data/mydata/addr', # 数据集根路径（丹麦语数据）
        reader_cfg=addr_desensitize_reader_cfg,
        infer_cfg=addr_desensitize_infer_cfg,
        eval_cfg=addr_desensitize_eval_cfg
    )
]
