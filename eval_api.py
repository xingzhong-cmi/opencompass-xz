# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi import Path as ApiPath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import subprocess
import re
import time
from pathlib import Path
import json
import pandas as pd
import logging
import uuid
from typing import List, Dict, Optional, Tuple, Any

# -------------------------- 1. 基础配置 --------------------------
# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OpenCompass_API")

# FastAPI实例初始化
app = FastAPI(title="OpenCompass评估工具API", description="对接Streamlit前端的模型评估后端服务")

# CORS配置（允许前端跨域访问，需替换为实际前端域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://172.22.152.1:380", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 常量定义（与前端完全一致，确保路径同步）
HUGGINGFACE_CACHE_PATH = Path.home() / ".cache" / "huggingface" / "hub"
DATASET_CONFIG_PATH = Path("/workspace/projects/opencompass/opencompass/configs/datasets")
GENERATE_SCRIPT_PATH = Path("/workspace/projects/opencompass/mytools/generate_task.py")
OUTPUT_DIR = Path("/workspace/projects/opencompass/outputs")
MY_DATASETS_DIR = Path("/workspace/projects/opencompass/opencompass/configs/datasets/mydatasets")

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 业务常量（与前端一致）
DATASET_TASK_MAP = {}  # 需与前端dataset_task_list.py中的映射保持一致
DEFAULT_TASK = "通用任务"
METRIC_TO_EVALUATOR = {
    "BLEU": "BleuEvaluator",
    "ROUGE": "RougeEvaluator",
    "准确率": "",
    "F1分数": "F1Evaluator",
    "困惑度": "PPLEvaluator",
}

# -------------------------- 2. 全局状态管理（任务/模型） --------------------------
# 自定义模型列表（替代前端session_state.custom_selected_models）
custom_models: List[str] = []

# 评估任务状态字典（task_id: 状态信息）
# 状态值：pending(等待中)、running(运行中)、success(成功)、failed(失败)
evaluation_tasks: Dict[str, Dict] = {}


# -------------------------- 3. 数据模型（Pydantic） --------------------------
class EvaluationRequest(BaseModel):
    """启动评估任务的请求参数模型"""
    selected_models: List[str]  # 选中的模型列表
    selected_datasets: List[str]  # 选中的数据集列表
    metrics: List[str]  # 选中的评价指标


class CustomModelRequest(BaseModel):
    """添加自定义模型的请求参数模型"""
    model_repo: str  # HuggingFace仓库路径（如Qwen/Qwen2-1.5B-Instruct）


# -------------------------- 4. 辅助函数（与前端逻辑对齐） --------------------------
def get_cached_models() -> List[str]:
    """获取本地HF缓存的模型列表（与前端get_cached_models一致）"""
    models = []
    if HUGGINGFACE_CACHE_PATH.exists():
        for item in HUGGINGFACE_CACHE_PATH.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                parts = item.name.split("--")[1:]
                if len(parts) >= 2:
                    model_name = "/".join(parts)
                    models.append(model_name)
    return sorted(models)


def get_dataset_configs() -> Tuple[List[str], Dict[str, str]]:
    """获取数据集配置列表及任务类型映射（与前端get_dataset_configs+任务映射逻辑一致）"""
    datasets = []
    if DATASET_CONFIG_PATH.exists():
        for root, _, files in os.walk(DATASET_CONFIG_PATH):
            for file in files:
                if file.endswith(".py") and not file.startswith("_") and file != "__init__.py":
                    dataset_name = os.path.splitext(file)[0]
                    datasets.append(dataset_name)
    datasets = sorted(list(set(datasets)))

    # 构建“数据集名→任务类型”映射
    dataset_to_task = {}
    max_len = max((len(ds) for ds in datasets), default=30)
    for ds in datasets:
        task_type = DEFAULT_TASK
        for key in DATASET_TASK_MAP:
            if key in ds:
                task_type = DATASET_TASK_MAP[key]
                break
        dataset_to_task[ds] = task_type
    return datasets, dataset_to_task


def find_dataset_config_files(dataset_names: List[str]) -> Dict[str, Path]:
    """根据数据集名称查找配置文件（与前端find_dataset_config_files一致）"""
    dataset_to_config = {}
    if not MY_DATASETS_DIR.exists():
        return dataset_to_config
    all_py_files = list(MY_DATASETS_DIR.glob("*.py"))
    for ds_name in dataset_names:
        found = False
        for py_file in all_py_files:
            if ds_name.lower() in py_file.stem.lower():
                dataset_to_config[ds_name] = py_file
                found = True
                break
        if not found:
            logger.warning(f"未找到数据集 '{ds_name}' 对应的配置文件")
    return dataset_to_config


def update_selected_dataset_evaluators(metric_names: List[str], dataset_to_config_map: Dict[str, Path]) -> None:
    """更新数据集配置文件的Evaluator（与前端update_selected_dataset_evaluators一致）"""
    if not metric_names or not dataset_to_config_map:
        return

    evaluator_types = [METRIC_TO_EVALUATOR.get(m, "BleuEvaluator") for m in metric_names]
    chosen_evaluator = evaluator_types[0] if evaluator_types else "BleuEvaluator"
    evaluator_import_line = "from opencompass.openicl.icl_evaluator import "

    for ds_name, config_path in dataset_to_config_map.items():
        try:
            content = config_path.read_text(encoding='utf-8')
            lines = content.splitlines()
            import_line_index = None
            current_imports = set()

            # 查找现有import行
            for i, line in enumerate(lines):
                if line.strip().startswith(evaluator_import_line):
                    import_line_index = i
                    imported_str = line[len(evaluator_import_line):].strip()
                    if imported_str.startswith('(') and imported_str.endswith(')'):
                        imported_str = imported_str[1:-1]
                    current_imports = set(name.strip() for name in imported_str.split(',') if name.strip())
                    break

            # 新增或更新import行
            if import_line_index is None:
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith('#'):
                        insert_index = i
                        break
                lines.insert(insert_index, evaluator_import_line + chosen_evaluator)
                logger.info(f"为数据集 {ds_name} 添加Evaluator导入: {chosen_evaluator}")
            else:
                if chosen_evaluator not in current_imports:
                    current_imports.add(chosen_evaluator)
                    if len(current_imports) > 3:
                        import_str = evaluator_import_line + "(\n    " + ",\n    ".join(sorted(current_imports)) + "\n)"
                    else:
                        import_str = evaluator_import_line + ", ".join(sorted(current_imports))
                    lines[import_line_index] = import_str
                    logger.info(f"为数据集 {ds_name} 更新Evaluator导入: {chosen_evaluator}")

            # 修改evaluator配置
            content = "\n".join(lines)
            pattern = r'(evaluator\s*=\s*dict\s*\(\s*type\s*=\s*)([\'"]?)(\w+)([\'"]?)'
            new_content = re.sub(pattern, rf'\1\2{chosen_evaluator}\4', content)
            if new_content != content:
                config_path.write_text(new_content, encoding='utf-8')
                logger.info(f"更新数据集 {ds_name} 的Evaluator为: {chosen_evaluator}")

        except Exception as e:
            logger.error(f"更新数据集 {ds_name} 配置失败: {str(e)}")


def run_command(command: str, timeout: int = 7200) -> Tuple[int, str]:
    """执行命令并捕获输出（与前端run_command一致，增强错误检测）"""
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
    ret_code = 0
    try:
        for line in process.stdout:
            line_clean = line.strip()
            output.append(line_clean)
            logger.info(f"命令输出: {line_clean}")

        process.wait(timeout=timeout)
        ret_code = process.returncode

    except subprocess.TimeoutExpired:
        process.kill()
        ret_code = -1
        output.append("\n❌ 执行超时，进程已强制终止\n")
        logger.error("命令执行超时")
    except Exception as e:
        ret_code = -2
        output.append(f"\n❌ 命令执行异常: {str(e)}\n")
        logger.error(f"命令执行异常: {str(e)}")

    # 增强错误检测（返回码0但含错误关键词仍视为失败）
    output_str = "\n".join(output)
    if ret_code == 0:
        error_keywords = ["ERROR", "fail", "Traceback", "Exception", "No module", "not found"]
        if any(keyword in output_str for keyword in error_keywords):
            logger.warning(f"命令返回码为0，但检测到错误日志")
            ret_code = 999  # 自定义逻辑失败码

    return ret_code, output_str


def get_latest_result_dir(task_id: str) -> Optional[Path]:
    """获取任务最新的评估结果目录（与前端逻辑一致）"""
    default_dir = OUTPUT_DIR / "default"
    if not default_dir.exists():
        logger.warning(f"任务 {task_id} 未找到default输出目录")
        return None

    # 匹配时间戳格式目录（如20250918_094456）
    timestamp_dirs = [
        d for d in default_dir.iterdir()
        if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)
    ]
    if not timestamp_dirs:
        logger.warning(f"任务 {task_id} 未找到时间戳结果目录")
        return None

    # 返回最新目录（按时间戳排序）
    latest_dir = max(timestamp_dirs, key=lambda d: d.name)
    logger.info(f"任务 {task_id} 最新结果目录: {latest_dir.name}")
    return latest_dir


# -------------------------- 5. 评估任务核心逻辑（异步） --------------------------
def run_evaluation_task(task_id: str, req: EvaluationRequest) -> None:
    """异步执行评估任务（后端核心逻辑，对应前端“启动评估”按钮）"""
    # 初始化任务状态
    evaluation_tasks[task_id] = {
        "status": "running",
        "output": "评估任务启动中...\n",
        "ret_code": None,
        "latest_result_dir": None,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None
    }
    output = [evaluation_tasks[task_id]["output"]]

    try:
        # 1. 验证参数
        if not req.selected_models or not req.selected_datasets:
            raise ValueError("模型和数据集不能为空")

        ## 2. 更新数据集Evaluator配置
        #output.append("🔄 更新数据集评价指标配置...")
        # dataset_to_config = find_dataset_config_files(req.selected_datasets)
        # if dataset_to_config:
        #     update_selected_dataset_evaluators(req.metrics, dataset_to_config)
        # else:
        #     output.append("⚠️ 未找到任何选中数据集的配置文件，将使用默认Evaluator")

        # 3. 生成评估脚本
        eval_script = f"eval_{task_id}.py"  # 按任务ID生成唯一脚本名
        model_args = " ".join(req.selected_models)
        dataset_args = f"--datasets {' '.join(req.selected_datasets)}" if req.selected_datasets else ""
        generate_cmd = f"python {GENERATE_SCRIPT_PATH} --models {model_args} --output {eval_script} {dataset_args}"

        output.append(f"\n📝 生成评估配置脚本: {eval_script}")
        output.append(f"生成命令: {generate_cmd}")
        ret_code, cmd_output = run_command(generate_cmd)
        output.append(cmd_output)

        if ret_code != 0:
            raise RuntimeError(f"生成评估配置失败，返回码: {ret_code}")

        # 4. 执行评估命令（主环境+fallback环境）
        eval_cmd = f"opencompass {eval_script}"
        eval_cmd_fallback = f"conda run --no-capture-output -n opencompass-old opencompass {eval_script}"

        output.append("\n🚀 开始模型评估（主环境）...")
        ret_code, cmd_output = run_command(eval_cmd)
        output.append(cmd_output)

        # 主环境失败则尝试fallback环境
        if ret_code != 0:
            output.append("\n⚠️ 主环境评估失败，切换至opencompass-old环境重试...")
            ret_code, cmd_output = run_command(eval_cmd_fallback)
            output.append(cmd_output)

        # 5. 处理评估结果
        if ret_code == 0:
            output.append("\n✅ 评估任务成功完成！")
            evaluation_tasks[task_id]["status"] = "success"
            # 获取最新结果目录
            latest_dir = get_latest_result_dir(task_id)
            evaluation_tasks[task_id]["latest_result_dir"] = str(latest_dir) if latest_dir else None
        else:
            output.append(f"\n❌ 评估任务失败，返回码: {ret_code}")
            evaluation_tasks[task_id]["status"] = "failed"

        evaluation_tasks[task_id]["ret_code"] = ret_code

    except Exception as e:
        error_msg = f"\n❌ 评估任务异常: {str(e)}"
        output.append(error_msg)
        logger.error(error_msg)
        evaluation_tasks[task_id]["status"] = "failed"
        evaluation_tasks[task_id]["ret_code"] = -3  # 异常失败码

    finally:
        # 更新任务最终状态
        evaluation_tasks[task_id]["output"] = "\n".join(output)
        evaluation_tasks[task_id]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        # 清理临时评估脚本
        eval_script = f"eval_{task_id}.py"
        if Path(eval_script).exists():
            Path(eval_script).unlink()
            logger.info(f"清理任务 {task_id} 的临时脚本: {eval_script}")


# -------------------------- 6. API端点（与前端功能一一对应） --------------------------
@app.get("/api/cached-models", summary="获取本地HF缓存模型列表")
async def api_get_cached_models() -> Dict[str, List[str]]:
    """对应前端“本地缓存模型”下拉框数据"""
    models = get_cached_models()
    return {"cached_models": models}


@app.get("/api/datasets", summary="获取数据集配置及任务类型")
async def api_get_datasets() -> Dict[str, Any]:
    """对应前端“数据集选择”下拉框数据（含任务类型）"""
    datasets, dataset_to_task = get_dataset_configs()
    return {
        "datasets": datasets,
        "dataset_to_task": dataset_to_task  # 数据集→任务类型映射
    }


@app.post("/api/custom-models", summary="添加自定义HF模型")
async def api_add_custom_model(req: CustomModelRequest) -> Dict[str, str]:
    """对应前端“添加自定义模型”功能"""
    if not req.model_repo.strip():
        raise HTTPException(status_code=400, detail="模型仓库路径不能为空")
    if req.model_repo in custom_models:
        raise HTTPException(status_code=400, detail=f"模型已存在: {req.model_repo}")

    custom_models.append(req.model_repo)
    logger.info(f"添加自定义模型: {req.model_repo}")
    return {"message": f"成功添加模型: {req.model_repo}", "custom_models": custom_models}


@app.delete("/api/custom-models/{model_repo}", summary="删除自定义HF模型")
async def api_delete_custom_model(model_repo: str = ApiPath(..., description="要删除的模型仓库路径")) -> Dict[str, str]:
    """对应前端“移除自定义模型”功能"""
    if model_repo not in custom_models:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_repo}")

    custom_models.remove(model_repo)
    logger.info(f"删除自定义模型: {model_repo}")
    return {"message": f"成功删除模型: {model_repo}", "custom_models": custom_models}


@app.get("/api/custom-models", summary="获取自定义模型列表")
async def api_get_custom_models() -> Dict[str, List[str]]:
    """对应前端“已选择的自定义模型”展示"""
    return {"custom_models": custom_models}


@app.post("/api/evaluation/start", summary="启动评估任务")
async def api_start_evaluation(req: EvaluationRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """对应前端“启动评估”按钮，返回任务ID用于查询状态"""
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())[:8]  # 简化为8位UUID
    # 提交异步任务
    background_tasks.add_task(run_evaluation_task, task_id, req)
    logger.info(f"启动评估任务，ID: {task_id}，模型: {req.selected_models}，数据集: {req.selected_datasets}")
    return {"task_id": task_id, "message": "评估任务已启动，可通过task_id查询状态"}


@app.get("/api/evaluation/status/{task_id}", summary="查询评估任务状态")
async def api_get_evaluation_status(task_id: str = ApiPath(..., description="评估任务ID")) -> Dict[str, Any]:
    """对应前端“评估过程与结果”展示，返回任务状态、输出日志等"""
    if task_id not in evaluation_tasks:
        raise HTTPException(status_code=404, detail=f"任务ID不存在: {task_id}")

    return evaluation_tasks[task_id]


@app.get("/api/evaluation/files/{task_id}", summary="获取评估结果文件列表")
async def api_get_evaluation_files(task_id: str = ApiPath(..., description="评估任务ID")) -> Dict[str, Any]:
    """对应前端“原始结果文件浏览器”，返回可查看的文件列表"""
    task = evaluation_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务ID不存在: {task_id}")
    if task["status"] not in ["success", "failed"]:
        raise HTTPException(status_code=400, detail=f"任务未完成，当前状态: {task['status']}")

    latest_dir = Path(task["latest_result_dir"]) if task["latest_result_dir"] else None
    if not latest_dir or not latest_dir.exists():
        return {"message": "未找到结果文件目录", "files": []}

    # 递归获取2层内的目标文件（与前端一致）
    all_files = []
    for sub_dir in ["configs", "logs", "predictions", "results", "summary"]:
        target_dir = latest_dir / sub_dir
        if target_dir.exists():
            for file in target_dir.rglob("*"):
                if file.is_file() and file.suffix in [".json", ".log", ".txt", ".md", ".csv", ".py"]:
                    rel_path = str(file.relative_to(latest_dir))  # 相对路径
                    all_files.append({
                        "relative_path": rel_path,
                        "file_name": file.name,
                        "file_size": f"{file.stat().st_size / 1024:.2f}KB"
                    })

    return {"latest_result_dir": str(latest_dir), "files": all_files}


@app.get("/api/evaluation/files/{task_id}/{file_path:path}", summary="获取具体结果文件内容")
async def api_get_evaluation_file_content(
        task_id: str = ApiPath(..., description="评估任务ID"),
        file_path: str = ApiPath(..., description="文件相对路径（从latest_result_dir开始）")
) -> Dict[str, Any]:
    """对应前端“查看文件内容”功能，返回文件内容及格式"""
    task = evaluation_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务ID不存在: {task_id}")

    latest_dir = Path(task["latest_result_dir"]) if task["latest_result_dir"] else None
    if not latest_dir or not latest_dir.exists():
        raise HTTPException(status_code=404, detail="未找到结果文件目录")

    # 拼接完整文件路径
    full_file_path = latest_dir / file_path
    if not full_file_path.exists() or not full_file_path.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    # 读取文件内容（按格式处理）
    try:
        content = full_file_path.read_text(encoding='utf-8')
        file_suffix = full_file_path.suffix.lower()

        if file_suffix == ".json":
            content = json.loads(content)  # 解析为JSON对象
        elif file_suffix == ".csv":
            content = pd.read_csv(full_file_path).to_dict("records")  # 解析为CSV列表

        return {
            "file_path": file_path,
            "file_suffix": file_suffix,
            "content": content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文件失败: {str(e)}")


# -------------------------- 7. 服务启动入口 --------------------------
if __name__ == "__main__":
    import uvicorn

    # 启动API服务（端�000，与前端8501、TTS API 8502区分）
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
