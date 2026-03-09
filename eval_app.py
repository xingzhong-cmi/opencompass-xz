import streamlit as st
import os
import subprocess
import re
import time
from pathlib import Path
import json
import pandas as pd  # 如果尚未导入
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

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 辅助函数：获取本地缓存的模型列表
def get_cached_models():
    models = []
    if HUGGINGFACE_CACHE_PATH.exists():
        # 匹配模型仓库目录（格式为 models--{author}--{model_name}）
        for item in HUGGINGFACE_CACHE_PATH.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                # 转换为标准仓库格式（author/model_name）
                parts = item.name.split("--")[1:]
                if len(parts) >= 2:
                    model_name = "/".join(parts)
                    models.append(model_name)
    return sorted(models)


# 辅助函数：获取数据集配置列表
def get_dataset_configs():
    datasets = []
    if DATASET_CONFIG_PATH.exists():
        # 递归查找所有Python配置文件
        for root, _, files in os.walk(DATASET_CONFIG_PATH):
            for file in files:
                if file.endswith(".py") and not file.startswith("_"):
                    # 提取数据集名称（不含.py后缀）
                    dataset_name = os.path.splitext(file)[0]
                    # 排除__init__.py
                    if dataset_name != "__init__":
                        datasets.append(dataset_name)
    return sorted(list(set(datasets)))  # 去重


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
        errors='replace'  # 避免编码错误导致崩溃
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

    # ✅ 增强检测：即使 returncode == 0，如果输出中包含 ERROR 或 fail，也视为失败
    if ret_code == 0:
        error_keywords = ["ERROR", "fail", "Traceback", "Exception", "No module", "not found"]
        if any(keyword in output_str for keyword in error_keywords):
            st.warning(f"⚠️ 检测到错误日志，尽管返回码为0，仍将视为失败")
            ret_code = 999  # 自定义错误码，表示“逻辑失败”

    return ret_code, output_str

# ============= 新增：评价指标与Evaluator映射 + 配置文件修改功能 =============
METRIC_TO_EVALUATOR = {
    "BLEU": "BleuEvaluator",
    "ROUGE": "RougeEvaluator",
    "准确率": "",
    "F1分数": "F1Evaluator",
    "困惑度": "PPLEvaluator",
}

def find_dataset_config_files(dataset_names, config_root):
    """根据数据集名称查找对应的配置文件"""
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
    """仅更新选中数据集对应配置文件的 evaluator 类型，并确保 import 语句存在"""
    if not metric_names or not dataset_to_config_map:
        return

    evaluator_types = [METRIC_TO_EVALUATOR.get(m, "BleuEvaluator") for m in metric_names]
    chosen_evaluator = evaluator_types[0] if evaluator_types else "BleuEvaluator"

    # 需要确保导入的模块
    evaluator_import_line = "from opencompass.openicl.icl_evaluator import "
    all_evaluators = list(METRIC_TO_EVALUATOR.values())  # 所有可能的 Evaluator

    for ds_name, config_path in dataset_to_config_map.items():
        try:
            content = config_path.read_text(encoding='utf-8')

            # 步骤1: 确保 import 语句包含所需 Evaluator
            lines = content.splitlines()
            import_line_index = None
            current_imports = set()

            # 查找现有的 evaluator import 行
            for i, line in enumerate(lines):
                if line.strip().startswith(evaluator_import_line):
                    import_line_index = i
                    # 提取已导入的类名
                    imported_str = line[len(evaluator_import_line):].strip()
                    if imported_str.startswith('(') and imported_str.endswith(')'):
                        imported_str = imported_str[1:-1]  # 去掉括号
                    current_imports = set(name.strip() for name in imported_str.split(',') if name.strip())
                    break

            # 如果没找到 import 行，我们在文件开头添加
            if import_line_index is None:
                # 找到第一个非注释、非空行的位置
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith('#'):
                        insert_index = i
                        break
                import_line_index = insert_index
                lines.insert(insert_index, evaluator_import_line + chosen_evaluator)
                st.info(f"➕ 已添加 import: {chosen_evaluator}")
            else:
                # 如果 import 行存在，但缺少 chosen_evaluator，则添加
                if chosen_evaluator not in current_imports:
                    current_imports.add(chosen_evaluator)
                    # 重新构建 import 行（多行或单行）
                    if len(current_imports) > 3:
                        # 多行格式
                        import_str = evaluator_import_line + "(\n    " + ",\n    ".join(sorted(current_imports)) + "\n)"
                    else:
                        # 单行格式
                        import_str = evaluator_import_line + ", ".join(sorted(current_imports))
                    lines[import_line_index] = import_str
                    st.info(f"➕ 已更新 import: 添加 {chosen_evaluator}")

            # 步骤2: 修改 evaluator=dict(type=...) 行
            content = "\n".join(lines)
            pattern = r'(evaluator\s*=\s*dict\s*\(\s*type\s*=\s*)([\'"]?)(\w+)([\'"]?)'
            new_content = re.sub(pattern, rf'\1\2{chosen_evaluator}\4', content)

            if new_content != content:
                config_path.write_text(new_content, encoding='utf-8')
                st.info(f"✅ 已更新 {config_path.name} ({ds_name}) 的 evaluator 为 {chosen_evaluator}")
            else:
                st.info(f"ℹ️ {config_path.name} ({ds_name}) 的 evaluator 已是 {chosen_evaluator}，无需更新")

        except Exception as e:
            st.error(f"❌ 更新 {config_path.name} ({ds_name}) 失败: {e}")


# 页面标题
st.title("OpenCompass 模型评估工具")

# 左侧栏 - 配置区域
with st.sidebar:
    st.header("评估配置")

    # 1. 模型加载模块
    st.subheader("1. 加载模型")
    model_source = st.radio(
        "选择模型来源",
        ("本地缓存模型", "自定义HuggingFace仓库")
    )

    selected_models = []
    if model_source == "本地缓存模型":
        cached_models = get_cached_models()
        if cached_models:
            selected_models = st.multiselect(
                "选择要评估的模型",
                cached_models,
                default=[] if len(cached_models) > 5 else cached_models[:2]
            )
        else:
            st.info("未在~/.cache/huggingface/hub/找到缓存的模型")

    # 2. 自定义模型处理部分（修改）
    else:
        custom_model = st.text_input(
            "输入HuggingFace仓库路径",
            "Qwen/Qwen2-1.5B-Instruct"
        )

        # 使用session_state保存已添加的自定义模型，避免刷新丢失
        if "custom_selected_models" not in st.session_state:
            st.session_state.custom_selected_models = []

        # 添加模型按钮逻辑
        if st.button("添加模型") and custom_model:
            if custom_model not in st.session_state.custom_selected_models:
                st.session_state.custom_selected_models.append(custom_model)
                st.success(f"已添加模型: {custom_model}")
                # 清空输入框
                st.session_state.custom_input = ""

        # 显示已添加的自定义模型
        if st.session_state.custom_selected_models:
            st.write("已选择的模型:")
            for i, model in enumerate(st.session_state.custom_selected_models):
                if st.button(f"移除 {model}", key=f"rm_{i}"):
                    st.session_state.custom_selected_models.pop(i)
                    st.experimental_rerun()

        # 将session_state中的模型列表赋值给selected_models
        selected_models = st.session_state.custom_selected_models

    # 2. 数据集加载模块
    st.subheader("2. 加载数据集")
    dataset_configs = get_dataset_configs()  # ← 就是这里定义的！
    # selected_datasets = st.multiselect(
    #     "选择要测试的数据集",
    #     dataset_configs,
    #     default=[] if len(dataset_configs) > 5 else dataset_configs[:2]
    # )
    # 构建带任务类型的显示选项
    dataset_display_options = []
    dataset_display_to_raw = {}  # 映射：显示名 → 原始名

    max_len = max((len(ds) for ds in dataset_configs), default=30)
    for ds in dataset_configs:
        task_type = DEFAULT_TASK
        # 尝试模糊匹配（如 flores_100_ind-tha → 匹配 "flores"）
        for key in DATASET_TASK_MAP:
            if key in ds:  # 支持部分匹配
                task_type = DATASET_TASK_MAP[key]
                break
        # 格式化对齐
        padding = " " * (max_len - len(ds) + 2)
        display_name = f"{ds}{padding}← {task_type}"
        dataset_display_options.append(display_name)
        dataset_display_to_raw[display_name] = ds

    # 显示给用户的选择器
    selected_display_datasets = st.multiselect(
        "选择要测试的数据集",
        dataset_display_options,
        default=[] if len(dataset_display_options) > 5 else dataset_display_options[:2]
    )

    # 转换回原始数据集名，用于后续命令
    selected_datasets = [dataset_display_to_raw[ds] for ds in selected_display_datasets]

    # 3. 评价指标模块
    st.subheader("3. 评价指标")
    #st.info("评价指标功能待实现")
    metrics = st.multiselect(
        "选择评价指标",
        ["准确率", "BLEU", "ROUGE", "F1分数", "困惑度"],
        default=["BLEU"]
    )

    # 开始测试按钮
    st.subheader("开始测试")
    run_evaluation = st.button("启动评估", disabled=not (selected_models and selected_datasets))

# 右侧栏 - 结果展示区域
col1, col2 = st.columns([3, 7])

with col2:
    st.header("评估过程与结果")

    # 初始化会话状态
    if "process_running" not in st.session_state:
        st.session_state.process_running = False
    if "evaluation_output" not in st.session_state:
        st.session_state.evaluation_output = ""
    if "evaluation_complete" not in st.session_state:
        st.session_state.evaluation_complete = False

    # 输出区域
    output_area = st.empty()
    result_area = st.empty()

    # 当点击开始测试按钮且没有正在运行的进程时
    if run_evaluation and not st.session_state.process_running:
        st.session_state.evaluation_complete = False
        st.session_state.evaluation_output = ""

        # 显示运行中的信息
        with output_area.container():
            st.info("评估正在进行中，请稍候...")

        eval_script = "eval_tmp.py"  # 生成的评估脚本名
        model_args = " ".join(selected_models)
        dataset_args = f"--datasets {' '.join(selected_datasets)}" if selected_datasets else ""

        # ============= 修复：移除嵌套 container，确保输出可见 =============
        MY_DATASETS_DIR = Path("/workspace/projects/opencompass/opencompass/configs/datasets/mydatasets")
        if selected_datasets and metrics:
            st.subheader("🔄 更新所选数据集的评价指标...")
            dataset_to_config = find_dataset_config_files(selected_datasets, MY_DATASETS_DIR)
            if dataset_to_config:
                update_selected_dataset_evaluators(metrics, dataset_to_config)
            else:
                st.warning("未找到任何选中数据集对应的配置文件")
        elif not metrics:
            st.warning("未选择评价指标，将使用配置文件默认 evaluator")
        elif not selected_datasets:
            st.warning("未选择数据集，跳过配置更新")
        # ============= 修复结束 =============

        generate_cmd = f"python {GENERATE_SCRIPT_PATH} --models {model_args} --output {eval_script} {dataset_args}"

        # 主评估命令（当前环境）
        eval_cmd = f"opencompass {eval_script}"  # ✅ 补上 --work-dir
        # 备用评估命令：在 opencompass-old 环境中运行（使用 --no-capture-output 支持实时输出）
        eval_cmd_fallback = f"conda run --no-capture-output -n opencompass-old opencompass {eval_script}"

        # 执行命令
        with output_area.container():
            st.subheader("生成评估配置...")
            ret_code, output = run_command(generate_cmd, st)
            st.session_state.evaluation_output += output

            if ret_code == 0:
                st.subheader("开始模型评估...")
                ret_code, output = run_command(eval_cmd, st)
                st.session_state.evaluation_output += output

                # 如果主命令失败（包括逻辑失败），尝试在 opencompass-old 环境中重试
                if ret_code != 0:
                    st.warning("主环境评估失败，正在切换至 opencompass-old 环境重试...")
                    ret_code, output = run_command(eval_cmd_fallback, st)
                    st.session_state.evaluation_output += output

                # ✅ 无论成功或失败，都设置为完成状态，确保显示结果板块
                if ret_code == 0:
                    st.success("✅ 评估成功完成！")
                else:
                    st.error(f"❌ 评估失败，返回代码: {ret_code} —— 但仍可查看输出文件和日志")

                # ✅ 关键修改：移到 if/else 外面，确保总是执行！
                st.session_state.evaluation_complete = True

            else:
                st.error(f"生成评估配置失败，返回代码: {ret_code}")
                # ✅ 即使生成失败，也允许查看历史输出
                st.session_state.evaluation_complete = True


    # 显示历史输出
    elif st.session_state.evaluation_output:
        with output_area.container():
            st.text(st.session_state.evaluation_output)

    # 评估完成后显示结果报告
    if st.session_state.evaluation_complete:
        with result_area.container():
            st.subheader("📊 评估结果报告")

            # === 自动查找最新输出目录 ===
            default_dir = OUTPUT_DIR / "default"
            # default_dir = OUTPUT_DIR
            latest_run_dir = None

            if default_dir.exists():
                # 获取所有时间戳格式的子目录（如 20250918_094456）
                timestamp_dirs = [
                    d for d in default_dir.iterdir()
                    if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)
                ]
                if timestamp_dirs:
                    latest_run_dir = max(timestamp_dirs, key=lambda d: d.name)  # 最新时间戳
                    st.info(f"📂 检测到最新评估结果目录: `{latest_run_dir.name}`")
                else:
                    st.warning("未找到时间戳格式的结果目录")
            else:
                st.warning(f"`default` 输出目录不存在: {default_dir}")

            # === 原“选择文件查看”功能保留（增强版）===
            st.subheader("📂 原始结果文件浏览器")
            if latest_run_dir and latest_run_dir.exists():
                # 递归列出所有文件（最多2层深度，避免性能问题）
                all_files = []
                for sub_dir in ["configs", "logs", "predictions", "results", "summary"]:
                    target_dir = latest_run_dir / sub_dir
                    if target_dir.exists():
                        for file in target_dir.rglob("*"):
                            if file.is_file() and file.suffix in [".json", ".log", ".txt", ".md", ".csv", ".py"]:
                                # 显示相对路径
                                rel_path = file.relative_to(latest_run_dir)
                                all_files.append((str(rel_path), file))

                if all_files:
                    # 按路径排序
                    all_files.sort(key=lambda x: x[0])
                    file_options = [f[0] for f in all_files]
                    selected_file_path = st.selectbox("选择文件查看", file_options, key="file_viewer")

                    # 获取选中文件
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
        # 评估状态信息
    st.subheader("评估状态")
    if st.session_state.process_running:
        st.warning("评估正在进行中...")
        if st.button("取消评估"):
            # 这里需要实际实现进程终止逻辑
            st.session_state.process_running = False
            st.warning("评估已取消")
    elif st.session_state.evaluation_complete:
        st.success("评估已完成")
        if st.button("重新开始"):
            # 重置会话状态
            st.session_state.process_running = False
            st.session_state.evaluation_complete = False
            st.session_state.evaluation_output = ""
            st.experimental_rerun()
    else:
        st.info("等待评估开始")

    # 系统信息
    st.subheader("系统信息")
    try:
        # 简单获取GPU信息（需要pynvml库）
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        st.text(f"GPU数量: {device_count}")

        # 显示第一个GPU信息
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            st.text(f"GPU内存使用: {mem_info.used / 1024 ** 3:.2f}GB / {mem_info.total / 1024 ** 3:.2f}GB")
    except:
        st.text("GPU信息获取失败（请安装pynvml）")

    # 评估配置文件路径
    if st.session_state.evaluation_complete or st.session_state.process_running:
        st.subheader("评估文件路径")
        st.text(f"配置脚本: {os.path.abspath('eval_tmp.py')}")
        st.text(f"输出目录: {OUTPUT_DIR.absolute()}")
