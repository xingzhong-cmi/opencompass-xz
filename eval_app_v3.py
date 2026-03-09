import streamlit as st
import os
import subprocess
import re
import time
import requests
from pathlib import Path
import json
import pandas as pd
from dataset_task_list import DATASET_TASK_MAP, DEFAULT_TASK

# 页面配置
st.set_page_config(
    page_title="OpenCompass评估工具",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 常量定义
HUGGINGFACE_CACHE_PATH = Path.home() / ".cache" / "huggingface" / "hub"
DATASET_CONFIG_PATH = Path("/workspace/projects/opencompass/opencompass/configs/datasets")
GENERATE_SCRIPT_PATH = Path("/workspace/projects/opencompass/mytools/generate_task.py")
OUTPUT_DIR = Path("/workspace/projects/opencompass/outputs")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 初始化会话状态
if "deepseek_api_key" not in st.session_state:
    st.session_state.deepseek_api_key = "sk-xxxxxx"  # 默认填充密钥
if "custom_selected_models" not in st.session_state:
    st.session_state.custom_selected_models = []
if "selected_display_datasets" not in st.session_state:
    st.session_state.selected_display_datasets = []
if "selected_metrics" not in st.session_state:
    st.session_state.selected_metrics = ["BLEU"]
if "recommended" not in st.session_state:
    st.session_state.recommended = {"models": [], "datasets": [], "metric": "BLEU"}


def get_recommended_models(task):
    try:
        # 检查API密钥是否存在
        if not st.session_state.deepseek_api_key:
            st.error("请先输入DeepSeek API密钥")
            return []

        # 任务提示词优化
        task_prompt_map = {
            "翻译任务": "机器翻译",
            "摘要任务": "文本摘要",
            "数学推理": "数学推理与解题",
            "代码生成": "代码生成与编程",
            "自然语言推理": "自然语言推理与理解"
        }
        specific_task = task_prompt_map.get(task, task)

        # API调用配置
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state.deepseek_api_key}"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": f"请推荐最新的3个用于【{specific_task}】的最佳Hugging Face模型，只返回模型的完整仓库路径（格式为作者/模型名），每个一行，不要额外说明"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            model_content = result["choices"][0]["message"]["content"].strip()
            models = model_content.split("\n")
            valid_models = [model.strip() for model in models if model.strip() and "/" in model]
            return valid_models[:3]
        else:
            st.error(f"API调用失败，状态码: {response.status_code}，响应: {response.text[:200]}")
            return []
    except Exception as e:
        st.error(f"获取推荐模型时出错: {str(e)}")
        return []


# 辅助函数：获取本地缓存的模型列表
def get_cached_models():
    models = []
    if HUGGINGFACE_CACHE_PATH.exists():
        for item in HUGGINGFACE_CACHE_PATH.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                parts = item.name.split("--")[1:]
                if len(parts) >= 2:
                    model_name = "/".join(parts)
                    models.append(model_name)
    return sorted(models)


# 辅助函数：获取数据集配置列表
def get_dataset_configs():
    datasets = []
    if DATASET_CONFIG_PATH.exists():
        for root, _, files in os.walk(DATASET_CONFIG_PATH):
            for file in files:
                if file.endswith(".py") and not file.startswith("_"):
                    dataset_name = os.path.splitext(file)[0]
                    if dataset_name != "__init__":
                        datasets.append(dataset_name)
    return sorted(list(set(datasets)))


# 辅助函数：执行命令并实时输出
def run_command(command, output_area, timeout=7200):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )

    output = []
    try:
        for line in process.stdout:
            line_clean = line.strip()
            output_area.write(line_clean)
            output.append(line)
            st.session_state.process_running = True

        process.wait(timeout=timeout)
        ret_code = process.returncode

    except subprocess.TimeoutExpired:
        process.kill()
        ret_code = -1
        output.append("\n❌ 执行超时，进程已强制终止\n")

    st.session_state.process_running = False
    output_str = "".join(output)

    if ret_code == 0:
        error_keywords = ["ERROR", "fail", "Traceback", "Exception", "No module", "not found"]
        if any(keyword in output_str for keyword in error_keywords):
            st.warning(f"⚠️ 检测到错误日志，尽管返回码为0，仍将视为失败")
            ret_code = 999

    return ret_code, output_str


# 评价指标与Evaluator映射 + 配置文件修改功能
METRIC_TO_EVALUATOR = {
    "BLEU": "BleuEvaluator",
    "ROUGE": "RougeEvaluator",
    "F1分数": "F1Evaluator",
    "困惑度": "PPLEvaluator",
}


def find_dataset_config_files(dataset_names, config_root):
    dataset_to_config = {}
    config_root = Path(config_root)
    if not config_root.exists():
        return dataset_to_config
    all_py_files = list(config_root.glob("*.py"))
    for ds_name in dataset_names:
        found = False
        for py_file in all_py_files:
            if ds_name.lower() in py_file.stem.lower():
                dataset_to_config[ds_name] = py_file
                found = True
                break
        if not found:
            st.warning(f"⚠️ 未找到数据集 '{ds_name}' 对应的配置文件")
    return dataset_to_config


def update_selected_dataset_evaluators(metric_names, dataset_to_config_map):
    if not metric_names or not dataset_to_config_map:
        return

    evaluator_types = [METRIC_TO_EVALUATOR.get(m, "BleuEvaluator") for m in metric_names]
    chosen_evaluator = evaluator_types[0] if evaluator_types else "BleuEvaluator"

    evaluator_import_line = "from opencompass.openicl.icl_evaluator import "
    all_evaluators = list(METRIC_TO_EVALUATOR.values())

    for ds_name, config_path in dataset_to_config_map.items():
        try:
            content = config_path.read_text(encoding='utf-8')
            lines = content.splitlines()
            import_line_index = None
            current_imports = set()

            for i, line in enumerate(lines):
                if line.strip().startswith(evaluator_import_line):
                    import_line_index = i
                    imported_str = line[len(evaluator_import_line):].strip()
                    if imported_str.startswith('(') and imported_str.endswith(')'):
                        imported_str = imported_str[1:-1]
                    current_imports = set(name.strip() for name in imported_str.split(',') if name.strip())
                    break

            if import_line_index is None:
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith('#'):
                        insert_index = i
                        break
                import_line_index = insert_index
                lines.insert(insert_index, evaluator_import_line + chosen_evaluator)
                st.info(f"➕ 已添加 import: {chosen_evaluator}")
            else:
                if chosen_evaluator not in current_imports:
                    current_imports.add(chosen_evaluator)
                    if len(current_imports) > 3:
                        import_str = evaluator_import_line + "(\n    " + ",\n    ".join(sorted(current_imports)) + "\n)"
                    else:
                        import_str = evaluator_import_line + ", ".join(sorted(current_imports))
                    lines[import_line_index] = import_str
                    st.info(f"➕ 已更新 import: 添加 {chosen_evaluator}")

            content = "\n".join(lines)
            pattern = r'(evaluator\s*=\s*dict\s*\(\s*type\s*=\s*)([\'"]?)(\w+)([\'"]?)'
            new_content = re.sub(pattern, rf'\1\2{chosen_evaluator}\4', content)

            if new_content != content:
                config_path.write_text(new_content, encoding='utf-8')
                # st.info(f"✅ 已更新 {config_path.name} ({ds_name}) 的 evaluator 为 {chosen_evaluator}")
            # else:
                # st.info(f"ℹ️ {config_path.name} ({ds_name}) 的 evaluator 已是 {chosen_evaluator}，无需更新")

        except Exception as e:
            st.error(f"❌ 更新 {config_path.name} ({ds_name}) 失败: {e}")


# 页面标题
st.title("OpenCompass 模型评估工具")

recommended_models = []

# 推荐测试计划生成器模块（唯一模块，默认折叠）
with st.expander("推荐测试计划生成器", expanded=False):
    # # API密钥配置
    # api_key = st.text_input(
    #     "DeepSeek API密钥",
    #     value=st.session_state.deepseek_api_key,
    #     type="password",
    #     help="请输入你的DeepSeek API密钥"
    # )
    # if api_key != st.session_state.deepseek_api_key:
    #     st.session_state.deepseek_api_key = api_key
    #     st.success("API密钥已更新")

    # 任务选择和生成按钮
    col_task, col_btn = st.columns([3, 1])
    with col_task:
        unique_tasks = sorted(list(set(DATASET_TASK_MAP.values())))
        selected_task = st.selectbox(
            "选择任务类型",
            unique_tasks,
            index=0,
            key="task_selector_unique"  # 唯一key
        )

    with col_btn:
        st.write("")  # 占位，使按钮垂直居中
        generate_plan = st.button(
            "生成推荐测试计划",
            use_container_width=True,
            key="generate_plan_btn"  # 唯一key
        )

    # 生成推荐计划逻辑
    if generate_plan:
        with st.spinner("正在使用deepseek搜索该任务的头部模型，并生成推荐测试计划..."):
            # 1. 获取推荐模型（保持不变）
            recommended_models = get_recommended_models(selected_task)
            if not recommended_models:
                recommended_models = ["未获取到推荐模型，请检查API密钥或网络"]

            # 2. 优化：先获取实际存在的数据集配置，再筛选匹配任务的数据集（核心修改部分）
            existing_datasets = get_dataset_configs()  # 实际存在的数据集配置名
            # st.warning(f"existing datasets: {existing_datasets}")

            # 新增：为每个数据集计算对应的任务类型（支持前缀匹配，和侧边栏逻辑一致）
            dataset_to_task = {}  # 存储每个数据集对应的任务类型
            for ds in existing_datasets:
                task_type = DEFAULT_TASK
                # 优先精确匹配（ds本身是DATASET_TASK_MAP的键）
                if ds in DATASET_TASK_MAP:
                    task_type = DATASET_TASK_MAP[ds]
                else:
                    # 前缀匹配（找DATASET_TASK_MAP中是ds前缀的键）
                    for key in DATASET_TASK_MAP:
                        if ds.startswith(key):
                            task_type = DATASET_TASK_MAP[key]
                            break
                dataset_to_task[ds] = task_type  # 记录每个数据集的任务类型

            # st.warning(f"数据集-任务类型映射（含前缀匹配）: {dataset_to_task}")
            # st.warning(f"selected_task: {selected_task}")

            # 精确匹配：任务类型与selected_task完全一致
            exact_matches = [ds for ds, task in dataset_to_task.items() if task == selected_task]
            # st.warning(f"exact_matches: {exact_matches}")

            # 模糊匹配：任务类型中包含selected_task（兜底）
            fuzzy_matches = [ds for ds, task in dataset_to_task.items() if selected_task in task]
            # st.warning(f"fuzzy_matches: {fuzzy_matches}")

            # 确定最终推荐数据集
            if exact_matches:
                recommended_datasets = exact_matches
            elif fuzzy_matches:
                recommended_datasets = fuzzy_matches
            else:
                recommended_datasets = []

            # 3. 评价指标（保持不变）
            recommended_metric = "BLEU"

            # 显示推荐表格（处理空数据集场景）
            st.subheader("推荐测试计划")
            if not recommended_datasets:
                st.warning("未找到匹配的数据集，请手动选择")
                table_data = {
                    "模型": recommended_models[:3] + [""] * (3 - len(recommended_models[:3])),
                    "数据集": ["无匹配数据集"] * 3,
                    "评价指标": [recommended_metric] * 3
                }
            else:
                display_models = recommended_models[:3]
                display_datasets = recommended_datasets[:]  # [：3]
                max_rows = max(len(display_models), len(display_datasets), 1)
                table_data = {
                    "模型": display_models + [""] * (max_rows - len(display_models)),
                    "数据集": display_datasets + [""] * (max_rows - len(display_datasets)),
                    "评价指标": [recommended_metric] + [""] * (max_rows - 1)
                }
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

            # 保存推荐结果（无数据集时不保存）
            st.session_state.recommended = {
                "models": recommended_models,
                "datasets": recommended_datasets,
                "metric": recommended_metric
            }

            # 确认按钮
            if st.button("确认并应用推荐配置", key="confirm_recommend_btn"):  # 唯一key
                # 更新模型配置：确保模型来源切换到自定义仓库
                valid_models = [m for m in recommended_models if "/" in m]
                st.session_state.custom_selected_models = valid_models[:3]  # 推荐最多3个模型
                st.session_state.model_source = "自定义HuggingFace仓库"  # 强制切换模型来源

                st.session_state.selected_display_datasets = ["flores_gen_aad4fd"]  #

                # 更新数据集配置
                dataset_configs = get_dataset_configs()
                dataset_display_options = []
                dataset_display_to_raw = {}
                max_len = max((len(ds) for ds in dataset_configs), default=30)
                for ds in dataset_configs:
                    task_type = DEFAULT_TASK
                    # 优先匹配完全一致的键
                    if ds in DATASET_TASK_MAP:
                        task_type = DATASET_TASK_MAP[ds]
                    else:
                        # 否则匹配前缀（如'flores'匹配'flores_gen'）
                        for key in DATASET_TASK_MAP:
                            if ds.startswith(key):  # 核心修改：检查数据集是否以键为前缀
                                task_type = DATASET_TASK_MAP[key]
                                break
                    padding = " " * (max_len - len(ds) + 2)
                    display_name = f"{ds}{padding}← {task_type}"
                    dataset_display_options.append(display_name)
                    dataset_display_to_raw[display_name] = ds

                # 匹配数据集显示名称（优化匹配逻辑）
                st.info("正在传输推荐的数据集作为配置参数.....")  # for debug
                selected_display_datasets = ['flores_gen_trans']  #
                for ds in recommended_datasets[:3]:  # 推荐最多3个数据集
                    # 优先完全匹配
                    exact_match = next((dn for dn, rn in dataset_display_to_raw.items() if rn.lower() == ds.lower()),
                                       None)
                    st.text("正在传输推荐的数据集作为配置参数.")
                    if exact_match:
                        selected_display_datasets.append(exact_match)
                    else:
                        # 模糊匹配（放宽条件）
                        fuzzy_match = next(
                            (dn for dn, rn in dataset_display_to_raw.items() if ds.lower() in rn.lower()), None)
                        if fuzzy_match:
                            selected_display_datasets.append(fuzzy_match)
                        else:
                            st.warning(f"未找到匹配的数据集: {ds}")

                st.session_state.selected_display_datasets = selected_display_datasets
                st.session_state.selected_metrics = [recommended_metric]
                st.warning(f"st.session_state.selected_display_datasets: {st.session_state.selected_display_datasets}")  #

                st.success("推荐配置已应用到左侧参数区！")
                st.rerun()  # 强制刷新页面使配置生效

# 左侧栏 - 配置区域
with st.sidebar:
    st.header("评估配置")

    # 初始化模型来源状态（新增）
    if "model_source" not in st.session_state:
        st.session_state.model_source = "自定义HuggingFace仓库"

    # 1. 模型加载模块
    st.subheader("1. 加载模型")
    model_source = st.radio(
        "选择模型来源",
        ("本地缓存模型", "自定义HuggingFace仓库"),  # 选项列表：索引0→本地缓存，索引1→自定义仓库
        index=1 if st.session_state.model_source == "自定义HuggingFace仓库" else 0,
        # 初始时，session_state的值是"自定义HuggingFace仓库"，所以index=1
        key="model_source_radio"
    )
    # 实时更新模型来源状态（新增）
    st.session_state.model_source = model_source

    selected_models = []
    if model_source == "本地缓存模型":
        cached_models = get_cached_models()
        if cached_models:
            selected_models = st.multiselect(
                "选择要评估的模型",
                cached_models,
                default=[],  # 本地模型不默认选择，避免冲突
                key="local_model_select"
            )
        else:
            st.info("未在~/.cache/huggingface/hub/找到缓存的模型")

    else:
        custom_model = st.text_input(
            "输入HuggingFace仓库路径",
            "Qwen/Qwen2-1.5B-Instruct",
            key="custom_model_input"
        )

        if st.button("添加模型", key="add_model_btn") and custom_model:
            if custom_model not in st.session_state.custom_selected_models:
                st.session_state.custom_selected_models.append(custom_model)
                st.success(f"已添加模型: {custom_model}")
                st.rerun()  # 立即刷新显示新增模型

        if st.session_state.custom_selected_models:
            st.write("已选择的模型:")
            for i, model in enumerate(st.session_state.custom_selected_models):
                if st.button(f"移除 {model}", key=f"rm_model_{i}"):  # 移除模型
                    st.session_state.custom_selected_models.pop(i)
                    st.rerun()

        if recommended_models:  #
            st.session_state.custom_selected_models = recommended_models  #

        selected_models = st.session_state.custom_selected_models

    # 2. 数据集加载模块
    st.subheader("2. 加载数据集")
    dataset_configs = get_dataset_configs()
    dataset_display_options = []
    dataset_display_to_raw = {}  # 显示名称 -> 原始数据集名
    raw_to_display = {}  # 新增：原始数据集名 -> 显示名称（用于反向映射）

    if dataset_configs:
        max_len = max((len(ds) for ds in dataset_configs), default=30)
        for ds in dataset_configs:
            task_type = DATASET_TASK_MAP.get(ds, DEFAULT_TASK)
            padding = " " * (max_len - len(ds) + 2)
            display_name = f"{ds}{padding}← {task_type}"
            dataset_display_options.append(display_name)
            dataset_display_to_raw[display_name] = ds
            raw_to_display[ds] = display_name  # 记录原始数据集到显示名称的映射
    else:
        st.info("未找到数据集配置文件，请检查路径：{}".format(DATASET_CONFIG_PATH))

    # 处理默认值：优先使用推荐数据集中的前三个
    default_datasets = []
    # 1. 从推荐结果中提取前三个数据集
    if "recommended" in st.session_state and st.session_state.recommended.get("datasets"):
        top3_recommended = st.session_state.recommended["datasets"][:4]  # 取前三个推荐数据集
        # 2. 将原始数据集名映射为显示名称
        for ds in top3_recommended:
            if ds in raw_to_display:  # 确保存在对应的显示名称
                display_name = raw_to_display[ds]
                if display_name in dataset_display_options:  # 确保显示名称有效
                    default_datasets.append(display_name)

    # 3. 如果有之前保存的选中状态，补充有效选项（可选，保留用户手动选择的历史）
    if "selected_display_datasets" in st.session_state:
        for d in st.session_state.selected_display_datasets:
            if d in dataset_display_options and d not in default_datasets:
                default_datasets.append(d)

    # 渲染Multiselect（默认值为推荐的前三个数据集）
    selected_display_datasets = st.multiselect(
        "选择要测试的数据集",
        dataset_display_options,
        default=default_datasets,
        key="dataset_select"
    )

    # 调试输出
    # st.text(f"推荐的前三个数据集：{st.session_state.recommended.get('datasets', [])[:3]}")
    # st.text(f"映射后的显示名称：{default_datasets}")
    # st.text(f"当前选中的数据集：{selected_display_datasets}")

    # 更新session状态，保存当前选中的显示名称（可选，用于后续复用）
    st.session_state.selected_display_datasets = selected_display_datasets

    # 映射为原始数据集名（用于后续评估）
    selected_datasets = [dataset_display_to_raw[ds] for ds in selected_display_datasets]

    # 3. 评价指标模块
    st.subheader("3. 评价指标")
    # 优化指标状态读取
    default_metrics = []
    if "selected_metrics" in st.session_state and st.session_state.selected_metrics:
        default_metrics = st.session_state.selected_metrics
        # 延迟删除状态
        # del st.session_state.selected_metrics

    metrics = st.multiselect(
        "选择评价指标",
        ["BLEU", "ROUGE", "F1分数", "困惑度"],
        default=default_metrics if default_metrics else ["BLEU"],
        key="metric_select"
    )

    # 选择后清除推荐状态（新增）
    if "selected_metrics" in st.session_state and metrics:
        del st.session_state.selected_metrics

    # 开始测试按钮
    st.subheader("开始测试")
    run_evaluation = st.button("启动评估", disabled=not (selected_models and selected_datasets), key="run_eval_btn")

# 右侧栏 - 结果展示区域
col1, col2 = st.columns([3, 7])

with col2:
    st.header("评估过程与结果")

    if "process_running" not in st.session_state:
        st.session_state.process_running = False
    if "evaluation_output" not in st.session_state:
        st.session_state.evaluation_output = ""
    if "evaluation_complete" not in st.session_state:
        st.session_state.evaluation_complete = False

    output_area = st.empty()
    result_area = st.empty()

    if run_evaluation and not st.session_state.process_running:
        st.session_state.evaluation_complete = False
        st.session_state.evaluation_output = ""

        with output_area.container():
            st.info("评估正在进行中，请稍候...")

        eval_script = "eval_tmp.py"
        model_args = " ".join(selected_models)
        dataset_args = f"--datasets {' '.join(selected_datasets)}" if selected_datasets else ""

        MY_DATASETS_DIR = Path("/workspace/projects/opencompass/opencompass/configs/datasets/mydatasets")
        if selected_datasets and metrics:
            # st.subheader("🔄 更新所选数据集的评价指标...")
            dataset_to_config = find_dataset_config_files(selected_datasets, MY_DATASETS_DIR)
            if dataset_to_config:
                update_selected_dataset_evaluators(metrics, dataset_to_config)
            else:
                st.warning("未找到任何选中数据集对应的配置文件")
        elif not metrics:
            st.warning("未选择评价指标，将使用配置文件默认 evaluator")
        elif not selected_datasets:
            st.warning("未选择数据集，跳过配置更新")

        generate_cmd = f"python {GENERATE_SCRIPT_PATH} --models {model_args} --output {eval_script} {dataset_args}"
        eval_cmd = f"opencompass {eval_script}"
        eval_cmd_fallback = f"conda run --no-capture-output -n opencompass-old opencompass {eval_script}"

        with output_area.container():
            st.subheader("生成评估配置...")
            ret_code, output = run_command(generate_cmd, st)
            st.session_state.evaluation_output += output

            if ret_code == 0:
                st.subheader("开始模型评估...")
                ret_code, output = run_command(eval_cmd, st)
                st.session_state.evaluation_output += output

                if ret_code != 0:
                    st.warning("主环境评估失败，正在切换ncompass-old 环境重试...")
                    ret_code, output = run_command(eval_cmd_fallback, st)
                    st.session_state.evaluation_output += output

                if ret_code == 0:
                    st.success("✅ 评估成功完成！")
                else:
                    st.error(f"❌ 评估失败，返回代码: {ret_code} —— 但仍可查看输出文件和日志")

                st.session_state.evaluation_complete = True

            else:
                st.error(f"生成评估配置失败，返回代码: {ret_code}")
                st.session_state.evaluation_complete = True


    elif st.session_state.evaluation_output:
        with output_area.container():
            st.text(st.session_state.evaluation_output)

    if st.session_state.evaluation_complete:
        with result_area.container():
            st.subheader("📊 评估结果报告")

            default_dir = OUTPUT_DIR / "default"
            latest_run_dir = None

            if default_dir.exists():
                timestamp_dirs = [
                    d for d in default_dir.iterdir()
                    if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)
                ]
                if timestamp_dirs:
                    latest_run_dir = max(timestamp_dirs, key=lambda d: d.name)
                    st.info(f"📂 检测到最新评估结果目录: `{latest_run_dir.name}`")
                else:
                    st.warning("未找到时间戳格式的结果目录")
            else:
                st.warning(f"`default` 输出目录不存在: {default_dir}")

            st.subheader("📂 原始结果文件浏览器")
            if latest_run_dir and latest_run_dir.exists():
                all_files = []
                for sub_dir in ["configs", "logs", "predictions", "results", "summary"]:
                    target_dir = latest_run_dir / sub_dir
                    if target_dir.exists():
                        for file in target_dir.rglob("*"):
                            if file.is_file() and file.suffix in [".json", ".log", ".txt", ".md", ".csv", ".py"]:
                                rel_path = file.relative_to(latest_run_dir)
                                all_files.append((str(rel_path), file))

                if all_files:
                    all_files.sort(key=lambda x: x[0])
                    file_options = [f[0] for f in all_files]
                    selected_file_path = st.selectbox("选择文件查看", file_options, key="file_viewer")

                    selected_file = next(f[1] for f in all_files if f[0] == selected_file_path)
                    if selected_file.exists() and selected_file.stat().st_size > 0:
                        try:
                            content = selected_file.read_text(encoding='utf-8')
                            if selected_file.suffix == ".json":
                                with st.expander(f"📄 {selected_file_path} (点击展开 JSON)"):
                                    st.json(json.loads(content))
                            elif selected_file.suffix in [".log", ".txt", ".md", ".py"]:
                                st.text_area(f"📄 {selected_file_path}", content, height=300)
                            elif selected_file.suffix == ".csv":
                                df = pd.read_csv(selected_file)
                                st.dataframe(df)
                        except Exception as e:
                            st.error(f"无法读取文件: {e}")
                    else:
                        st.info("文件为空")
                else:
                    st.info("未找到可查看的文件")

            else:
                st.info("请先完成评估以查看结果文件")

# 左侧显示当前配置摘要
with col1:
    st.header("当前配置")

    st.subheader("模型")
    if selected_models:
        for model in selected_models:
            st.text(model)
    else:
        st.info("未选择模型")

    st.subheader("数据集")
    if selected_datasets:
        for dataset in selected_datasets:
            st.text(dataset)
    else:
        st.info("未选择数据集")

    st.subheader("评价指标")
    if metrics:
        for metric in metrics:
            st.text(metric)
    else:
        st.info("未选择评价指标")

    st.subheader("评估状态")
    if st.session_state.process_running:
        st.warning("评估正在进行中...")
        if st.button("取消评估", key="cancel_eval_btn"):
            st.session_state.process_running = False
            st.warning("评估已取消")
    elif st.session_state.evaluation_complete:
        st.success("评估已完成")
        if st.button("重新开始", key="restart_btn"):
            st.session_state.process_running = False
            st.session_state.evaluation_complete = False
            st.session_state.evaluation_output = ""
            st.rerun()
    else:
        st.info("等待评估开始")

    st.subheader("系统信息")
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        st.text(f"GPU数量: {device_count}")

        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            st.text(f"GPU内存使用: {mem_info.used / 1024 ** 3:.2f}GB / {mem_info.total / 1024 ** 3:.2f}GB")
    except:
        st.text("GPU信息获取失败（请安装pynvml）")

    if st.session_state.evaluation_complete or st.session_state.process_running:
        st.subheader("评估文件路径")
        st.text(f"配置脚本: {os.path.abspath('eval_tmp.py')}")
        st.text(f"输出目录: {OUTPUT_DIR.absolute()}")
