import os
import sys
import re

def generate_model_config(hf_repo, eval_script_path):
    """
    根据Hugging Face仓库路径生成配置文件并更新指定的评估脚本
    确保文件名和模型名称仅使用下划线作为分隔符
    """
    # 处理可能包含的URL前缀，提取纯净的仓库路径
    if hf_repo.startswith(('https://huggingface.co/', 'http://huggingface.co/')):
        hf_repo = hf_repo.split('huggingface.co/')[-1].strip('/')
    
    # 提取模型名称（路径中的最后一部分）
    model_name = hf_repo.split('/')[-1]
    
    # 处理模型名称中的点号和连字符，统一替换为下划线
    # 这是关键修改，确保符合Python变量命名规范
    sanitized_model_name = model_name.replace('.', '_').replace('-', '_')
    config_filename = f"my_{sanitized_model_name}.py"
    
    # 配置文件保存目录
    config_dir = "/workspace/projects/opencompass/opencompass/configs/models/mymodels"
    config_path = os.path.join(config_dir, config_filename)
    
    # 生成配置文件内容
    config_content = f"""from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='{model_name}',  # 保留原始名称作为缩写
        path='{hf_repo}',     # 确保这里没有URL前缀
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
"""
    
    # 确保配置目录存在
    os.makedirs(config_dir, exist_ok=True)
    
    # 写入配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"已生成配置文件: {config_path}")
    
    # 更新评估脚本
    update_eval_script(eval_script_path, model_name, sanitized_model_name, config_filename)

def update_eval_script(script_path, original_model_name, sanitized_model_name, config_filename):
    """更新评估脚本，确保导入语句使用有效的Python变量名"""
    script_path = os.path.abspath(script_path)
    
    if not os.path.exists(script_path):
        print(f"错误: 评估脚本 {script_path} 不存在")
        return
    
    # 读取现有脚本内容
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 生成清理后的模型变量名（仅使用下划线）
    model_var_name = f"{sanitized_model_name}_models"
    # 导入语句 - 现在使用合法的变量名
    import_line = f"from opencompass.configs.models.mymodels.{config_filename.split('.')[0]} import models as {model_var_name}"
    
    # 检查模型是否已导入
    if import_line in content:
        print(f"模型 {original_model_name} 已在评估脚本中导入，无需重复添加")
        return
    
    # 找到合适的位置添加导入语句
    pattern = r'(with read_base\(\):\s+#.*?)(?=\n\ndatasets =)'
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    
    if match:
        # 添加新的导入，确保正确缩进
        indent = '    '  # 4个空格缩进
        new_import = f"\n{indent}{import_line}"
        new_content = content[:match.end(1)] + new_import + content[match.end(1):]
    else:
        print("警告: 未找到合适的位置添加导入语句，将使用备用位置")
        with_block_pattern = r'(with read_base\(\):\s+)(.*?)(?=\n\S)'
        with_match = re.search(with_block_pattern, content, re.DOTALL | re.MULTILINE)
        if with_match:
            indent = '    '
            new_import = f"\n{indent}{import_line}"
            new_content = content[:with_match.end(2)] + new_import + content[with_match.end(2):]
        else:
            print("错误: 无法找到添加导入语句的位置")
            return
    
    # 更新models列表
    models_pattern = r'(models = .+?)(?=\n|$)'
    models_match = re.search(models_pattern, new_content, re.DOTALL)
    
    if models_match:
        current_models = models_match.group(1)
        new_models = f"{current_models} + {model_var_name}"
        new_content = new_content[:models_match.start(1)] + new_models + new_content[models_match.end(1):]
    else:
        print("错误: 未找到models定义，无法更新")
        return
    
    # 写入更新后的内容
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"已更新评估脚本: {script_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python generate_model_config.py <hugging_face_repo> <eval_script_path>")
        print("示例: python generate_model_config.py 'Qwen/Qwen2.5-Omni-7B' '/workspace/projects/opencompass/eval_llm_lang.py'")
        sys.exit(1)
    
    hf_repo = sys.argv[1].strip("'\"")
    eval_script_path = sys.argv[2].strip("'\"")
    generate_model_config(hf_repo, eval_script_path)

