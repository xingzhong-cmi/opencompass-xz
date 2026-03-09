# 数据集 → 任务类型 映射表（根据提供的大小写和拼写更新）
DATASET_TASK_MAP = {
    # 翻译 & 摘要
    "flores": "翻译任务",
    "iwslt2017": "翻译任务",
    "XLSum": "摘要任务",
    "Xsum": "摘要任务",
    "lcsts": "摘要任务",  # 已按输入拼写调整
    "summscreen": "摘要任务",  # 已按输入拼写调整
    "summedits": "摘要任务",  # 已按输入拼写调整
    "mgsm": "翻译任务",  # 新增：多语言摘要任务

    # 数学推理
    "gsm8k": "数学推理",
    "math": "数学推理",  # 已按输入拼写调整
    "aime2024": "数学推理",
    "aime2025": "数学推理",
    "OlymMATH": "数学推理",
    "OlympiadBench": "数学推理",
    "MathBench": "数学推理",
    "gaokao_math": "数学推理",
    "mastermath2024v1": "数学推理",
    "math401": "数学推理",
    "SVAMP": "数学推理",
    "game24": "数学推理",
    "gsm8k_contamination": "数学推理",  # 新增：污染版本数学推理
    "gsm_hard": "数学推理",  # 新增：难版数学推理
    "omni_math": "数学推理",  # 新增：全能数学推理

    # 代码生成 & 编程
    "humaneval": "代码生成",
    "humaneval_plus": "代码生成",
    "humaneval_pro": "代码生成",
    "humaneval_cn": "代码生成",
    "humaneval_multi": "代码生成",
    "mbpp": "代码生成",
    "mbpp_plus": "代码生成",
    "mbpp_pro": "代码生成",
    "mbpp_cn": "代码生成",  # 新增：中文MBPP
    "apps": "代码生成",
    "codecompass": "代码生成",  # 已按输入拼写调整
    "livecodebench": "代码生成",  # 已按输入拼写调整
    "multipl_e": "代码生成",  # 已按输入拼写调整
    "Py150": "代码生成",
    "humanevalx": "代码生成",  # 新增：扩展版HumanEval

    # 自然语言推理 & 文本蕴含（含各类前缀）
    "cmnli": "自然语言推理",
    "ocnli": "自然语言推理",
    "ocnli_fc": "自然语言推理",
    "afqmc": "自然语言推理",
    "AX_b": "自然语言推理",
    "AX_g": "自然语言推理",
    "RTE": "自然语言推理",
    "ANLI": "自然语言推理",
    "CB": "自然语言推理",
    "COPA": "自然语言推理",
    "GLUE_QQP": "自然语言推理",  # 新增：GLUE句子对匹配
    "GLUE_CoLA": "自然语言推理",  # 新增：GLUE句子可接受性
    "GLUE_MRPC": "自然语言推理",  # 新增：GLUE语义等价
    "SuperGLUE_AX_b": "自然语言推理",  # 新增：SuperGLUE AX-b
    "SuperGLUE_AX_g": "自然语言推理",  # 新增：SuperGLUE AX-g
    "SuperGLUE_COPA": "自然语言推理",  # 新增：SuperGLUE COPA
    "SuperGLUE_RTE": "自然语言推理",  # 新增：SuperGLUE RTE
    "SuperGLUE_CB": "自然语言推理",  # 新增：SuperGLUE CB
    "CLUE_afqmc": "自然语言推理",  # 新增：CLUE-afqmc
    "CLUE_cmnli": "自然语言推理",  # 新增：CLUE-cmnli
    "CLUE_ocnli": "自然语言推理",  # 新增：CLUE-ocnli
    "CLUE_ocnli_fc": "自然语言推理",  # 新增：CLUE-ocnli_fc
    "FewCLUE_bustm": "自然语言推理",  # 新增：FewCLUE短文本匹配
    "XCOPA": "自然语言推理",  # 新增：跨语言COPA
    "adv_glue": "自然语言推理",  # 新增：高级GLUE
    "bbeh": "大模型基准",  # 新增：类似BBH的基准任务
    "bbh": "大模型基准",  # 补充：原有未明确分类，按类型归类

    # 词义消歧 & 代词指代
    "WiC": "词义消歧",
    "SuperGLUE_WiC": "词义消歧",  # 新增：SuperGLUE WiC
    "SuperGLUE_WSC": "词义消歧",  # 新增：SuperGLUE代词指代
    "FewCLUE_cluewsc": "词义消歧",  # 新增：FewCLUE代词指代
    "winograd": "词义消歧",  # 已按输入拼写调整

    # 完形填空
    "ReCoRD": "完形填空",
    "SuperGLUE_ReCoRD": "完形填空",  # 新增：SuperGLUE ReCoRD
    "CHID": "完形填空",
    "FewCLUE_chid": "完形填空",  # 新增：FewCLUE成语填空
    "ClozeTest_maxmin": "完形填空",  # 新增：极值完形填空
    "cmo_fib": "完形填空",  # 新增：CMO完形填空

    # 选择题 & 知识问答
    "ceval": "选择题问答",
    "cmmlu": "选择题问答",
    "mmlu": "选择题问答",
    "mmlu_cf": "选择题问答",
    "mmlu_pro": "选择题问答",
    "mmmlu": "选择题问答",  # 已按输入拼写调整
    "mmmlu_lite": "选择题问答",  # 已按输入拼写调整
    "GaokaoBench": "选择题问答",
    "MMLUArabic": "选择题问答",  # 新增：阿拉伯语MMLU
    "PJExam": "选择题问答",  # 新增：某考试选择题
    "ruler": "综合评估",  # 新增：评估基准工具

    # 医学相关
    "MedQA": "医学选择题",
    "MedMCQA": "医学选择题",
    "MedCalc_Bench": "医学计算",
    "MedXpertQA": "医学问答",
    "Medbullets": "医学知识",
    "MedBench": "医学综合",
    "CIBench": "临床信息",
    "CARDBiomedBench": "生物医学",
    "BioBench": "生物学",
    "PubMedQA": "生物医学问答",
    "NPHardEval": "NP难题",
    "ProteinLMBench": "蛋白质语言",
    "ClinicBench": "临床问答",
    "HealthBench": "健康问答",
    "nejm_ai_benchmark": "医学AI",  # 已按输入拼写调整
    "HLE": "医学综合",  # 新增：健康素养评估
    "civilcomments": "毒性检测",  # 新增：医学/社会评论毒性检测

    # 科学相关
    "ChemBench": "化学知识",
    "chem_exam": "化学知识",  # 新增：化学考试
    "PHYSICS": "物理知识",
    "PHYBench": "物理知识",
    "TheoremQA": "定理推理",
    "ScienceQA": "科学问答",
    "SciKnowEval": "科学知识",
    "SciEval": "科学评估",
    "scicode": "科学代码",  # 已按输入拼写调整
    "matbench": "科学知识",  # 新增：材料科学基准
    "Earth_Silver": "地球科学",
    "ClimaQA": "气候问答",
    "scibench": "科学综合",  # 新增：科学综合基准
    "livestembench": "实时STEM",  # 新增：实时STEM评估

    # 开放域问答
    "triviaqa": "开放域问答",
    "nq": "开放域问答",
    "nq_cn": "开放域问答",
    "qasper": "开放域问答",
    "qaspercut": "开放域问答",
    "HotpotQA": "多跳问答",
    "drop": "数值问答",  # 已按输入拼写调整
    "narrativeqa": "故事问答",  # 已按输入拼写调整
    "QuALITY": "长文问答",
    "leval": "长文问答",  # 已按输入拼写调整
    "longbench": "长文问答",  # 已按输入拼写调整
    "longbenchv2": "长文问答",  # 已按输入拼写调整
    "triviaqarc": "开放域问答",  # 新增：TriviaQA变体
    "gpqa": "开放域问答",  # 新增：通用问答

    # 常识推理
    "winogrande": "常识推理",
    "piqa": "常识推理",
    "hellaswag": "常识推理",
    "WinoGrande": "常识推理",
    "siqa": "社会常识",  # 已按输入拼写调整
    "CSQA": "常识问答",
    "CSQA_CN": "常识问答",
    "obqa": "开放常识",  # 已按输入拼写调整
    "ARC_c": "常识推理",
    "ARC_e": "常识推理",
    "ARC_Prize_Public_Evaluation": "常识推理",
    "strategyqa": "策略推理",  # 已按输入拼写调整
    "babilong": "常识推理",
    "commonsenseqa": "常识推理",  # 新增：常识问答
    "commonsenseqa_cn": "常识推理",  # 新增：中文常识问答

    # 安全 & 伦理 & 偏见
    "safety": "安全评估",
    "realtoxicprompts": "毒性检测",
    "jigsawmultilingual": "毒性检测",
    "crowspairs": "社会偏见",
    "crowspairs_cn": "社会偏见",
    "contamination": "数据污染",
    "s3eval": "安全评估",  # 已按输入拼写调整
    "internsandbox": "安全评估",  # 新增：内部沙箱安全

    # 其他专项
    "IFEval": "指令遵循",
    "PMMEval": "多模态评估",
    "R_Bench": "检索评估",
    "TabMWP": "表格推理",
    "needlebench": "大海捞针",
    "needlebench_v2": "大海捞针",
    "infinitebench": "无限上下文",
    "llm_compression": "压缩评估",
    "promptbench": "提示工程",
    "calm": "校准评估",
    "agieval": "综合评估",
    "compassbench_v1_3": "综合基准",
    "compassbench_20_v1_1": "综合基准",
    "compassbench_20_v1_1_public": "综合基准",
    "livemathbench": "实时数学",  # 已按输入拼写调整
    "livereasonbench": "实时推理",  # 已按输入拼写调整
    "TACO": "工具调用",
    "sage": "专家评估",  # 已按输入拼写调整
    "wikibench": "百科问答",  # 已按输入拼写调整
    "tydiqa": "多语言问答",  # 已按输入拼写调整
    "xiezhi": "专业知识",
    "govrepcrs": "政府报告",  # 已按输入拼写调整
    "lawbench": "法律问答",  # 已按输入拼写调整
    "kaoshi": "考试题",  # 已按输入拼写调整
    "hungarian_exam": "匈牙利考试",  # 已按输入拼写调整
    "OpenFinData": "金融问答",
    "FinanceIQ": "金融知识",
    "dingo": "对话评估",  # 已按输入拼写调整
    "eese": "教育评估",  # 已按输入拼写调整
    "SimpleQA": "简单问答",
    "chinese_simpleqa": "中文简单问答",  # 已按输入拼写调整
    "SmolInstruct": "小型指令",
    "Bustm": "短文本匹配",
    "EPRSTMT": "情感分类",
    "FewCLUE_eprstmt": "情感分类",  # 新增：FewCLUE情感分类
    "TNEWS": "新闻分类",
    "FewCLUE_tnews": "新闻分类",  # 新增：FewCLUE新闻分类
    "CSL": "论文关键词",
    "FewCLUE_csl": "论文关键词",  # 新增：FewCLUE论文关键词
    "storycloze": "故事补全",  # 已按输入拼写调整
    "wikitext": "语言建模",  # 已按输入拼写调整
    "supergpqa": "专家级问答",  # 已按输入拼写调整
    "teval": "文本评估",  # 已按输入拼写调整
    "musr": "多步推理",  # 已按输入拼写调整
    "cvalues": "价值观评估",  # 已按输入拼写调整
    "ds1000": "数据科学",  # 已按输入拼写调整
    "qabench": "问答评估",  # 新增：问答评估基准
    "race": "阅读理解",  # 已按输入拼写调整
    "CLUE_C3": "阅读理解",  # 新增：CLUE阅读理解
    "CLUE_CMRC": "阅读理解",  # 新增：CLUE阅读理解
    "CLUE_DRCD": "阅读理解",  # 新增：CLUE阅读理解
    "squad20": "阅读理解",  # 新增：SQuAD 2.0
    "srbench": "检索评估",  # 新增：检索评估基准
    "collections": "未知任务",  # 新增：数据集集合（未明确任务）
    "inference_ppl": "语言建模",  # 新增：推理困惑度评估
    "anthropics_evals": "综合评估",  # 新增：Anthropic评估基准
    "judge": "主观评估",  # 新增：判断类主观评估
    "kcle": "未知任务",  # 新增：未明确任务类型
    "korbench": "综合评估",  # 新增：韩语综合基准
    "lambada": "语言建模",  # 新增：语言建模数据集
    "CHARM": "对话评估",  # 新增：对话相关
    "LCBench": "综合评估",  # 新增：语言代码基准
    "cmb": "综合评估",  # 新增：综合基准

    # 自定义数据集
    "mydatasets": "自定义数据集",
}

# 如果找不到映射，默认显示“未知任务”
DEFAULT_TASK = "未知任务"
