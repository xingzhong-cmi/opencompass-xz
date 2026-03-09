import requests
import time
import pprint
from typing import Optional, Dict, List

# ===================== 1. 测试配置（请根据你的环境修改）=====================
API_BASE_URL = "http://localhost:8000"  # API服务地址（默认端口8503）
TEST_PARAMS = {
    # 测试用自定义模型（确保HF仓库可访问，如公开模型）
    "test_custom_model": "facebook/m2m100_1.2B",
    # 测试用模型列表（优先选本地已缓存的模型，避免下载耗时）
    "test_models": ["facebook/m2m100_1.2B"],
    # 测试用数据集（需与你的API服务中实际存在的数据集匹配）
    "test_datasets": ["flores_gen_trans"],  # 示例：mmlu、copa等，根据你的配置修改
    # 测试用评价指标
    "test_metrics": ["BLEU"],
    # 评估任务超时时间（单位：秒，默认5分钟）
    "eval_timeout": 300,
    # 轮询任务状态的间隔（单位：秒）
    "poll_interval": 5
}

# 初始化打印工具（让输出更清晰）
pp = pprint.PrettyPrinter(indent=2)
print("=" * 60)
print("OpenCompass API 分步骤测试脚本")
print(f"API服务地址: {API_BASE_URL}")
print("测试参数:")
pp.pprint(TEST_PARAMS)
print("=" * 60)


# ===================== 2. 工具函数（封装重复逻辑）=====================
def send_request(
    method: str,
    url: str,
    params: Optional[Dict] = None,
    json: Optional[Dict] = None
) -> Optional[Dict]:
    """发送API请求并处理响应"""
    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=json,
            timeout=10
        )
        # 检查状态码（200/201为成功，其他为错误）
        if response.status_code in [200, 201]:
            return {"success": True, "data": response.json(), "status_code": response.status_code}
        else:
            return {
                "success": False,
                "error": f"HTTP错误 {response.status_code}",
                "detail": response.text,
                "status_code": response.status_code
            }
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "连接失败", "detail": "API服务未启动或地址错误"}
    except Exception as e:
        return {"success": False, "error": "请求异常", "detail": str(e)}


def print_step_result(step: str, result: Dict) -> None:
    """打印步骤结果（绿色成功/红色失败）"""
    if result["success"]:
        print(f"\033[92m✅ 步骤「{step}」成功\033[0m")  # 绿色
        if "data" in result and result["data"]:
            print("返回数据:")
            pp.pprint(result["data"])
    else:
        print(f"\033[91m❌ 步骤「{step}」失败\033[0m")  # 红色
        print(f"错误原因: {result['error']}")
        if "detail" in result:
            print(f"详细信息: {result['detail']}")
    print("-" * 60)


# ===================== 3. 分步骤测试逻辑 =====================
def test_step1_api_liveness() -> bool:
    """步骤1：测试API服务连通性（通过获取缓存模型接口验证）"""
    step_name = "1. 测试API服务连通性"
    url = f"{API_BASE_URL}/api/cached-models"
    result = send_request(method="GET", url=url)
    print_step_result(step_name, result)
    return result["success"]


def test_step2_get_cached_models() -> bool:
    """步骤2：测试获取本地缓存模型列表"""
    step_name = "2. 获取本地缓存模型列表"
    url = f"{API_BASE_URL}/api/cached-models"
    result = send_request(method="GET", url=url)
    
    # 额外验证：返回数据是否包含"cached_models"字段
    if result["success"]:
        if "cached_models" not in result["data"]:
            result["success"] = False
            result["error"] = "返回格式错误"
            result["detail"] = "未找到 'cached_models' 字段"
    
    print_step_result(step_name, result)
    return result["success"]


def test_step3_custom_model_manage() -> bool:
    """步骤3：测试自定义模型管理（添加→查询→删除）"""
    step_name = "3. 自定义模型管理（添加→查询→删除）"
    test_model = TEST_PARAMS["test_custom_model"]
    success_flag = True

    # 3.1 添加自定义模型
    print(f"子步骤3.1：添加自定义模型「{test_model}」")
    add_url = f"{API_BASE_URL}/api/custom-models"
    add_result = send_request(method="POST", url=add_url, json={"model_repo": test_model})
    if not add_result["success"]:
        # 允许"模型已存在"的错误（避免重复测试残留）
        if "模型已存在" in str(add_result["detail"]):
            print(f"⚠️  模型已存在，跳过添加（非错误）")
        else:
            success_flag = False
            print(f"❌ 添加模型失败: {add_result['error']}")

    # 3.2 查询自定义模型列表
    print(f"子步骤3.2：查询自定义模型列表")
    list_url = f"{API_BASE_URL}/api/custom-models"
    list_result = send_request(method="GET", url=list_url)
    if not list_result["success"]:
        success_flag = False
        print(f"❌ 查询模型列表失败: {list_result['error']}")
    else:
        custom_models = list_result["data"]["custom_models"]
        if test_model not in custom_models:
            success_flag = False
            print(f"❌ 自定义模型列表中未找到「{test_model}」")
        else:
            print(f"✅ 自定义模型列表包含目标模型: {custom_models}")

    # 3.3 删除自定义模型
    print(f"子步骤3.3：删除自定义模型「{test_model}」")
    delete_url = f"{API_BASE_URL}/api/custom-models/{test_model}"
    delete_result = send_request(method="DELETE", url=delete_url)
    if not delete_result["success"]:
        # 允许"模型不存在"的错误（可能之前已删除）
        if "模型不存在" in str(delete_result["detail"]):
            print(f"⚠️  模型不存在，跳过删除（非错误）")
        else:
            success_flag = False
            print(f"❌ 删除模型失败: {delete_result['error']}")

    # 打印步骤总结果
    if success_flag:
        print(f"\033[92m✅ 步骤「{step_name}」成功\033[0m")
    else:
        print(f"\033[91m❌ 步骤「{step_name}」失败\033[0m")
    print("-" * 60)
    return success_flag


def test_step4_get_datasets() -> bool:
    """步骤4：测试获取数据集列表及任务类型"""
    step_name = "4. 获取数据集列表及任务类型"
    url = f"{API_BASE_URL}/api/datasets"
    result = send_request(method="GET", url=url)
    
    # 额外验证：返回是否包含"datasets"和"dataset_to_task"字段
    if result["success"]:
        required_fields = ["datasets", "dataset_to_task"]
        missing_fields = [f for f in required_fields if f not in result["data"]]
        if missing_fields:
            result["success"] = False
            result["error"] = "返回格式错误"
            result["detail"] = f"缺少必要字段: {missing_fields}"
        else:
            # 验证测试数据集是否在返回列表中（可选，避免用户配置差异）
            test_ds = TEST_PARAMS["test_datasets"][0]
            if test_ds not in result["data"]["datasets"]:
                print(f"⚠️  测试数据集「{test_ds}」未在返回列表中，可能影响后续评估测试")
    
    print_step_result(step_name, result)
    return result["success"]


def test_step5_evaluation_flow() -> tuple[bool, Optional[str]]:
    """步骤5：测试评估任务全流程（启动→轮询→查询结果）"""
    step_name = "5. 评估任务全流程（启动→轮询→查询结果）"
    task_id = None
    success_flag = True

    # 5.1 启动评估任务
    print(f"子步骤5.1：启动评估任务")
    eval_url = f"{API_BASE_URL}/api/evaluation/start"
    eval_body = {
        "selected_models": TEST_PARAMS["test_models"],
        "selected_datasets": TEST_PARAMS["test_datasets"],
        "metrics": TEST_PARAMS["test_metrics"]
    }
    start_result = send_request(method="POST", url=eval_url, json=eval_body)
    if not start_result["success"]:
        success_flag = False
        print(f"❌ 启动评估任务失败: {start_result['error']}")
        print(f"详细信息: {start_result['detail']}")
        print_step_result(step_name, start_result)
        return success_flag, task_id
    else:
        task_id = start_result["data"]["task_id"]
        print(f"✅ 评估任务启动成功，任务ID: {task_id}")

    # 5.2 轮询任务状态（直到完成或超时）
    print(f"子步骤5.2：轮询任务状态（任务ID: {task_id}，超时: {TEST_PARAMS['eval_timeout']}秒）")
    status_url = f"{API_BASE_URL}/api/evaluation/status/{task_id}"
    start_time = time.time()
    task_complete = False
    task_status = None

    while time.time() - start_time < TEST_PARAMS["eval_timeout"]:
        time.sleep(TEST_PARAMS["poll_interval"])
        status_result = send_request(method="GET", url=status_url)
        if not status_result["success"]:
            success_flag = False
            print(f"❌ 查询任务状态失败: {status_result['error']}")
            break
        
        task_data = status_result["data"]
        task_status = task_data["status"]
        print(f"⏳ 任务状态: {task_status}（已耗时: {int(time.time()-start_time)}秒）")
        
        # 任务完成（成功/失败均视为完成）
        if task_status in ["success", "failed"]:
            task_complete = True
            print(f"✅ 任务完成，最终状态: {task_status}")
            print(f"任务输出（前500字符）: {task_data['output'][:500]}...")
            break

    if not task_complete:
        success_flag = False
        print(f"❌ 任务超时（超过{TEST_PARAMS['eval_timeout']}秒）")

    # 5.3 查询评估结果文件
    if success_flag and task_status == "success":
        print(f"子步骤5.3：查询评估结果文件列表")
        files_url = f"{API_BASE_URL}/api/evaluation/files/{task_id}"
        files_result = send_request(method="GET", url=files_url)
        if not files_result["success"]:
            success_flag = False
            print(f"❌ 查询结果文件失败: {files_result['error']}")
        else:
            files_data = files_result["data"]
            if not files_data["files"]:
                success_flag = False
                print(f"❌ 未找到评估结果文件")
            else:
                print(f"✅ 找到 {len(files_data['files'])} 个结果文件")
                # 打印前3个文件示例
                for i, file in enumerate(files_data["files"][:3]):
                    print(f"   {i+1}. {file['relative_path']}（大小: {file['file_size']}）")

                # 5.4 读取一个结果文件（选第一个JSON/CSV文件）
                print(f"子步骤5.4：读取第一个结果文件内容")
                target_file = None
                for file in files_data["files"]:
                    if file["relative_path"].endswith((".json", ".csv")):
                        target_file = file["relative_path"]
                        break
                
                if target_file:
                    file_url = f"{API_BASE_URL}/api/evaluation/files/{task_id}/{target_file}"
                    file_result = send_request(method="GET", url=file_url)
                    if not file_result["success"]:
                        success_flag = False
                        print(f"❌ 读取文件「{target_file}」失败: {file_result['error']}")
                    else:
                        print(f"✅ 成功读取文件「{target_file}」，内容（前300字符）:")
                        file_content = file_result["data"]["content"]
                        # 处理JSON/CSV格式的输出
                        if isinstance(file_content, dict) or isinstance(file_content, list):
                            pp.pprint(file_content)
                        else:
                            print(str(file_content)[:300] + "...")
                else:
                    print(f"⚠️  未找到JSON/CSV格式的结果文件，跳过文件读取")

    # 打印步骤总结果
    if success_flag:
        print(f"\033[92m✅ 步骤「{step_name}」成功\033[0m")
    else:
        print(f"\033[91m❌ 步骤「{step_name}」失败\033[0m")
    print("-" * 60)
    return success_flag, task_id


# ===================== 4. 执行测试 =====================
if __name__ == "__main__":
    # 存储各步骤结果
    step_results = []
    task_id = None

    # 按顺序执行步骤（前序步骤失败时，后续步骤可选择性跳过）
    print("开始执行测试...\n")

    # 步骤1：API连通性（核心，失败则后续步骤无法执行）
    step1_ok = test_step1_api_liveness()
    step_results.append(("API连通性", step1_ok))
    if not step1_ok:
        print("\033[91m❌ 核心步骤失败，后续测试无法执行，退出测试\033[0m")
        exit(1)

    # 步骤2：获取缓存模型
    step2_ok = test_step2_get_cached_models()
    step_results.append(("获取缓存模型", step2_ok))

    # 步骤3：自定义模型管理
    step3_ok = test_step3_custom_model_manage()
    step_results.append(("自定义模型管理", step3_ok))

    # 步骤4：获取数据集
    step4_ok = test_step4_get_datasets()
    step_results.append(("获取数据集", step4_ok))

    # 步骤5：评估任务全流程（依赖前序步骤基本正常）
    if step2_ok and step4_ok:
        step5_ok, task_id = test_step5_evaluation_flow()
    else:
        step5_ok = False
        print(f"\033[93m⚠️  跳过步骤5（获取缓存模型/数据集步骤失败）\033[0m")
        print("-" * 60)
    step_results.append(("评估任务全流程", step5_ok))

    # ===================== 6. 测试总结 =====================
    print("\n" + "=" * 60)
    print("📊 OpenCompass API 测试总结")
    print("=" * 60)
    total_steps = len(step_results)
    success_steps = sum(1 for _, ok in step_results if ok)
    failure_steps = total_steps - success_steps

    print(f"总步骤数: {total_steps}")
    print(f"成功步骤数: \033[92m{success_steps}\033[0m")
    print(f"失败步骤数: \033[91m{failure_steps}\033[0m")
    print("\n各步骤详情:")
    for i, (step_name, ok) in enumerate(step_results, 1):
        status = "✅ 成功" if ok else "❌ 失败"
        print(f"{i}. {step_name}: {status}")

    if task_id:
        print(f"\n💡 评估任务ID: {task_id}（可用于后续单独查询结果）")
    print(f"\n💡 测试建议:")
    if failure_steps > 0:
        print("1. 优先修复失败的前序步骤（如API连通性、模型/数据集查询）")
        print("2. 评估任务失败时，可通过任务ID查询详细日志:")
        print(f"   GET {API_BASE_URL}/api/evaluation/status/{task_id}")
    else:
        print("1. 所有API功能测试通过，可正常对接前端使用")
        print("2. 建议保存测试任务ID，用于后续结果复现和分析")
    print("=" * 60)
