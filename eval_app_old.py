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
# def get_dataset_configs():
#     datasets = []
#     if DATASET_CONFIG_PATH.exists():
#         # 递归查找所有Python配置文件
#         for root, _, files in os.walk(DATASET_CONFIG_PATH):
#             for file in files:
#                 if file.endswith(".py") and not file.startswith("_"):
#                     # 提取数据集名称（不含.py后缀）
#                     dataset_name = os.path.splitext(file)[0]
#                     # 排除__init__.py
#                     if dataset_name != "__init__":
#                         datasets.append(dataset_name)
#     return sorted(list(set(datasets)))  # 去重


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

# def run_command(command, output_area):
#     process = subprocess.Popen(
#         command,
#         shell=True,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,
#         text=True,
#         bufsize=1
#     )
#
#     output = []
#     for line in process.stdout:
#         output_area.write(line.strip())
#         output.append(line)
#         # 刷新界面
#         st.session_state.process_running = True
#
#     process.wait()
#     st.session_state.process_running = False
#
#     return process.returncode, "".join(output)


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

    # else:
    #     custom_model = st.text_input(
    #         "输入HuggingFace仓库路径",
    #         "Qwen/Qwen2-1.5B-Instruct"
    #     )
    #     if custom_model:
    #         if st.button("添加模型"):
    #             if custom_model not in selected_models:
    #                 selected_models.append(custom_model)
    #                 st.success(f"已添加模型: {custom_model}")
    #
    #     # 显示已添加的自定义模型
    #     if selected_models:
    #         st.write("已选择的模型:")
    #         for i, model in enumerate(selected_models):
    #             if st.button(f"移除 {model}", key=f"rm_{i}"):
    #                 selected_models.pop(i)
    #                 st.experimental_rerun()

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

    # st.subheader("2. 加载数据集")
    # dataset_configs = get_dataset_configs()
    # selected_datasets = st.multiselect(
    #     "选择要测试的数据集",
    #     dataset_configs,
    #     default=[] if len(dataset_configs) > 5 else dataset_configs[:2]
    # )

    # 3. 评价指标模块
    st.subheader("3. 评价指标")
    st.info("评价指标功能待实现")
    metrics = st.multiselect(
        "选择评价指标",
        ["准确率", "BLEU", "ROUGE", "F1分数", "困惑度"],
        default=["准确率"]
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

        generate_cmd = f"python {GENERATE_SCRIPT_PATH} --models {model_args} --output {eval_script} {dataset_args}"

        # 主评估命令（当前环境）
        eval_cmd = f"opencompass {eval_script}"
        # 备用评估命令：在 opencompass-old 环境中运行
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
                if ret_code != 0:  # ← 现在能捕获 output 中含 ERROR 但 ret_code=0 的情况
                    st.warning("主环境评估失败，正在切换至 opencompass-old 环境重试...")
                    ret_code, output = run_command(eval_cmd_fallback, st)
                    st.session_state.evaluation_output += output

                if ret_code == 0:
                    st.success("评估完成!")
                    st.session_state.evaluation_complete = True
                else:
                    # st.session_state.evaluation_complete = True
                    st.error(f"评估失败，返回代码: {ret_code}")
            # if ret_code == 0:
            #     st.subheader("开始模型评估...")
            #     ret_code, output = run_command(eval_cmd, st)
            #     st.session_state.evaluation_output += output
            #
            #     # 如果主命令失败，尝试在 opencompass-old 环境中重试
            #     if ret_code != 0:
            #         st.warning("主环境评估失败，正在切换至 opencompass-old 环境重试...")
            #         ret_code, output = run_command(eval_cmd_fallback, st)
            #         st.session_state.evaluation_output += output
            #
            #     if ret_code == 0:
            #         st.success("评估完成!")
            #         st.session_state.evaluation_complete = True
            #     else:
            #         st.error(f"评估失败，返回代码: {ret_code}")
            # else:
            #     st.error(f"生成评估配置失败，返回代码: {ret_code}")

    # # 当点击开始测试按钮且没有正在运行的进程时
    # if run_evaluation and not st.session_state.process_running:
    #     st.session_state.evaluation_complete = False
    #     st.session_state.evaluation_output = ""
    #
    #     # 显示运行中的信息
    #     with output_area.container():
    #         st.info("评估正在进行中，请稍候...")
    #
    #     eval_script = "eval_tmp.py"  # 生成的评估脚本名
    #     # 关键修复：移除模型路径的引号，改用原始格式传递
    #     # 因为argparse会自动处理带斜杠的参数，引号反而会导致解析错误
    #     model_args = " ".join(selected_models)  # 直接拼接模型路径，不添加引号
    #     dataset_args = f"--datasets {' '.join(selected_datasets)}" if selected_datasets else ""
    #
    #     # 明确参数顺序：--models 模型列表 --output 脚本名 数据集参数
    #     generate_cmd = f"python {GENERATE_SCRIPT_PATH} --models {model_args} --output {eval_script} {dataset_args}"
    #
    #     # 第二步：运行生成的评估脚本
    #     eval_cmd = f"opencompass {eval_script}"
    #
    #     # 执行命令
    #     with output_area.container():
    #         st.subheader("生成评估配置...")
    #         ret_code, output = run_command(generate_cmd, st)
    #         st.session_state.evaluation_output += output
    #
    #         if ret_code == 0:
    #             st.subheader("开始模型评估...")
    #             ret_code, output = run_command(eval_cmd, st)
    #             st.session_state.evaluation_output += output
    #
    #             if ret_code == 0:
    #                 st.success("评估完成!")
    #                 st.session_state.evaluation_complete = True
    #             else:
    #                 st.error(f"评估失败，返回代码: {ret_code}")
    #         else:
    #             st.error(f"生成评估配置失败，返回代码: {ret_code}")

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

            # # === 提取并展示预测结果（前3条）===
            # if latest_run_dir:
            #     predictions_dir = latest_run_dir / "predictions"
            #     if predictions_dir.exists():
            #         st.subheader("📈 模型预测结果（前3条）")
            #
            #         # 遍历每个模型目录
            #         for model_dir in predictions_dir.iterdir():
            #             if model_dir.is_dir():
            #                 model_name = model_dir.name
            #                 st.markdown(f"### 🤖 模型: `{model_name}`")
            #
            #                 # 查找该模型下的所有 .json 预测文件
            #                 json_files = list(model_dir.glob("*.json"))
            #                 if not json_files:
            #                     st.info(f"未找到预测文件")
            #                     continue
            #
            #                 # 取第一个数据集文件（通常只有一个）
            #                 pred_file = json_files[0]
            #                 try:
            #                     data = pred_file.read_text(encoding='utf-8')
            #                     predictions = json.loads(data)
            #
            #                     if isinstance(predictions, list) and len(predictions) > 0:
            #                         # 取前3条，或全部如果不足3条
            #                         sample_preds = predictions[:3]
            #                         display_data = []
            #
            #                         for i, pred in enumerate(sample_preds, 1):
            #                             # 根据实际结构提取字段，常见字段如：'prompt', 'prediction', 'gold', 'input', 'output'
            #                             # 适配你的数据结构
            #                             row = {
            #                                 "序号": i,
            #                                 "输入/提示": str(pred.get('prompt', pred.get('input', 'N/A'))),
            #                                 "模型输出": str(pred.get('prediction', pred.get('output', 'N/A'))),
            #                                 "参考答案": str(pred.get('gold', pred.get('target', 'N/A'))),
            #                             }
            #                             display_data.append(row)
            #
            #                         # 用 DataFrame 展示
            #                         df = pd.DataFrame(display_data)
            #                         st.dataframe(df, use_container_width=True)
            #
            #                         # 可选：展开查看完整 JSON
            #                         with st.expander("🔍 查看完整 JSON 内容"):
            #                             st.json(predictions)
            #
            #                     else:
            #                         st.warning("预测数据格式异常或为空")
            #
            #                 except Exception as e:
            #                     st.error(f"解析预测文件失败: {e}")
            #
            #     else:
            #         st.warning("未找到 predictions 目录")

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
    # # 评估完成后显示结果报告
    # if st.session_state.evaluation_complete:
    #     with result_area.container():
    #         st.subheader("评估结果报告")
    #
    #         # 这里可以根据实际输出格式解析结果
    #         # 示例：从输出中提取关键信息
    #         output = st.session_state.evaluation_output
    #         result_summary = []
    #
    #         # 简单的结果提取示例（实际应根据OpenCompass输出格式调整）
    #         for model in selected_models:
    #             model_name = model.split("/")[-1]
    #             # 模拟结果提取
    #             match = re.search(f"{model_name}.*?准确率.*?(\d+\.\d+)%", output)
    #             if match:
    #                 acc = match.group(1)
    #                 result_summary.append({
    #                     "模型": model_name,
    #                     "准确率": f"{acc}%"
    #                 })
    #
    #         if result_summary:
    #             st.dataframe(result_summary)
    #         else:
    #             st.info("未找到可解析的评估结果，请查看上面的详细输出")
    #
    #         # 显示预测结果文件
    #         st.subheader("预测结果文件")
    #         if OUTPUT_DIR.exists():
    #             result_files = list(OUTPUT_DIR.glob("*.json")) + list(OUTPUT_DIR.glob("*.log"))
    #             if result_files:
    #                 selected_file = st.selectbox("选择结果文件查看", result_files)
    #                 if selected_file.exists() and selected_file.stat().st_size > 0:
    #                     if selected_file.suffix == ".json":
    #                         st.json(selected_file.read_text())
    #                     else:
    #                         st.text(selected_file.read_text())
    #             else:
    #                 st.info("输出目录中未找到结果文件")
    #         else:
    #             st.warning(f"输出目录不存在: {OUTPUT_DIR}")

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
