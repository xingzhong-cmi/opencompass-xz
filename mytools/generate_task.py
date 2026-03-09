import os
import sys
import re
import shutil
import argparse

def find_dataset_file(dataset_base, src_base):
    """递归查找数据集配置文件（包括子目录）"""
    patterns = [
        re.compile(f"^{dataset_base}\.py$"),  # 精确匹配
        re.compile(f"^{dataset_base}_gen_.*\.py$")  # 带gen后缀的匹配
    ]
    
    for root, dirs, files in os.walk(src_base):
        for file in files:
            for pattern in patterns:
                if pattern.match(file):
                    return os.path.join(root, file)
    return None

def copy_dataset_config(dataset_name):
    """拷贝数据集配置文件到mydatasets目录"""
    src_base = "/workspace/projects/opencompass/opencompass/configs/datasets"
    dst_base = "/workspace/projects/opencompass/opencompass/configs/datasets/mydatasets"
    
    dataset_base = dataset_name[:-3] if dataset_name.endswith(".py") else dataset_name
    
    src_path = find_dataset_file(dataset_base, src_base)
    if not src_path:
        raise FileNotFoundError(
            f"未找到数据集配置文件: 在 {src_base} 及其子目录中未发现 {dataset_base} 相关文件"
        )
    
    src_filename = os.path.basename(src_path)
    dst_path = os.path.join(dst_base, src_filename)
    
    os.makedirs(dst_base, exist_ok=True)
    
    if os.path.exists(dst_path):
        print(f"数据集配置文件 {dst_path} 已存在，跳过拷贝")
    else:
        shutil.copy2(src_path, dst_path)
        print(f"已拷贝数据集配置: {src_path} -> {dst_path}")
    
    dataset_base_name = os.path.splitext(src_filename)[0]
    dataset_var = re.sub(r'_gen_.*', '_datasets', dataset_base_name)
    return dataset_var, src_filename, src_filename[:-3]  # 返回文件名（不含后缀）

def generate_model_config(hf_repo, 
                          eval_script_path,
                          dataset_vars=None,
                          dataset_filenames=None,
                          dataset_basenames=None,
                          model_type="HuggingFacewithChatTemplate",
                          max_out_len=1024,
                          batch_size=8,
                          num_gpus=4,
                          all_model_basenames=None,
                          all_model_vars=None,
                          is_first_model=False):  # 新增：标记是否为第一个模型
    """生成模型配置文件（新增is_first_model参数控制脚本删除）"""
    if hf_repo.startswith(('https://huggingface.co/', 'http://huggingface.co/')):
        hf_repo = hf_repo.split('huggingface.co/')[-1].strip('/')
    
    model_name = hf_repo.split('/')[-1]
    sanitized_model_name = model_name.replace('.', '_').replace('-', '_')
    config_filename = f"my_{sanitized_model_name}.py"
    config_dir = "/workspace/projects/opencompass/opencompass/configs/models/mymodels"
    config_path = os.path.join(config_dir, config_filename)
    
    if os.path.exists(config_path):
        print(f"模型配置文件 {config_path} 已存在，跳过生成")
    else:
        config_content = f"""from opencompass.models import {model_type}

models = [
    dict(
        type={model_type},
        abbr='{model_name}',
        path='{hf_repo}',
        max_out_len={max_out_len},
        batch_size={batch_size},
        run_cfg=dict(num_gpus={num_gpus}),
    )
]
"""
        os.makedirs(config_dir, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"已生成模型配置文件: {config_path}")
    
    current_model_basename = config_filename[:-3]
    
    update_eval_script(
        script_path=eval_script_path,
        model_name=model_name,
        sanitized_model_name=sanitized_model_name,
        config_filename=config_filename,
        current_model_basename=current_model_basename,
        all_model_basenames=all_model_basenames,
        dataset_vars=dataset_vars,
        dataset_filenames=dataset_filenames,
        dataset_basenames=dataset_basenames,
        all_model_vars=all_model_vars,
        is_first_model=is_first_model  # 传递“是否第一个模型”标记
    )

def update_eval_script(script_path, 
                      model_name, 
                      sanitized_model_name, 
                      config_filename,
                      current_model_basename,
                      all_model_basenames,
                      dataset_vars, 
                      dataset_filenames,
                      dataset_basenames,
                      all_model_vars=None,
                      is_first_model=False):  # 新增：仅第一个模型时删除旧脚本
    """更新评估脚本（关键：仅首次处理模型时删除旧脚本）"""
    script_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_path)
    os.makedirs(script_dir, exist_ok=True)

    # >>>>>>>>>> 核心改动：仅在处理第一个模型时删除已有脚本 <<<<<<<<<<
    if is_first_model and os.path.exists(script_path):
        os.remove(script_path)
        print(f"已删除旧评估脚本: {script_path}")

    # 初始化新脚本（仅在脚本不存在时）
    if not os.path.exists(script_path):
        print(f"创建新评估脚本: {script_path}")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("from mmengine.config import read_base\n\n")
            f.write("with read_base():\n")
            f.write("    # 数据集和模型导入区域\n")
            f.write("\n")
            f.write("datasets = []\n")
            f.write("models = []\n")

    # 读取现有脚本内容
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    
    # 1. 清理现有导入（只保留当前指令中需要的）
    # 1.1 清理数据集导入
    if dataset_basenames:
        dataset_import_pattern = re.compile(r'from opencompass\.configs\.datasets\.mydatasets\.[^\s]+ import [^\s]+\n?', re.MULTILINE)
        all_dataset_imports = dataset_import_pattern.findall(new_content)
        
        keep_dataset_imports = []
        for imp in all_dataset_imports:
            for basename in dataset_basenames:
                if basename in imp:
                    keep_dataset_imports.append(imp)
                    break
        
        if all_dataset_imports:
            new_content = dataset_import_pattern.sub('', new_content)
            pattern = r'(with read_base\(\):\s+# 数据集和模型导入区域\s+)(?=\n|    )'
            match = re.search(pattern, new_content, re.DOTALL | re.MULTILINE)
            if match:
                new_content = new_content[:match.end(1)] + ''.join([f"    {imp}" for imp in keep_dataset_imports]) + new_content[match.end(1):]
    
    # 1.2 清理模型导入
    if all_model_basenames:
        model_import_pattern = re.compile(r'from opencompass\.configs\.models\.mymodels\.[^\s]+ import models as [^\s]+\n?', re.MULTILINE)
        all_model_imports = model_import_pattern.findall(new_content)
        
        keep_model_imports = []
        for imp in all_model_imports:
            for basename in all_model_basenames:
                if basename in imp:
                    keep_model_imports.append(imp)
                    break
        
        if all_model_imports:
            new_content = model_import_pattern.sub('', new_content)
            if dataset_basenames and keep_dataset_imports:
                last_dataset_import = keep_dataset_imports[-1]
                pattern = re.escape(last_dataset_import) + r'\n?'
                match = re.search(pattern, new_content)
                if match:
                    new_content = new_content[:match.end(0)] + ''.join([f"    {imp}" for imp in keep_model_imports]) + new_content[match.end(0):]
            else:
                pattern = r'(with read_base\(\):\s+# 数据集和模型导入区域\s+)(?=\n|    )'
                match = re.search(pattern, new_content, re.DOTALL | re.MULTILINE)
                if match:
                    new_content = new_content[:match.end(1)] + ''.join([f"    {imp}" for imp in keep_model_imports]) + new_content[match.end(1):]
    
    # 2. 添加当前会话需要的导入（如果缺失）
    # 2.1 添加数据集导入
    dataset_imports = []
    if dataset_vars and dataset_filenames and dataset_basenames:
        for dataset_var, dataset_filename, dataset_basename in zip(dataset_vars, dataset_filenames, dataset_basenames):
            import_line = f"from opencompass.configs.datasets.mydatasets.{dataset_basename} import {dataset_var}\n"
            dataset_imports.append(import_line)
            
            if import_line not in new_content:
                pattern = r'(with read_base\(\):\s+# 数据集和模型导入区域\s+)(?=\n|    )'
                match = re.search(pattern, new_content, re.DOTALL | re.MULTILINE)
                if match:
                    indent = '    '
                    new_import = f"{indent}{import_line}"
                    new_content = new_content[:match.end(1)] + new_import + new_content[match.end(1):]
                else:
                    print(f"警告: 未找到合适位置添加数据集 {dataset_var} 导入")
        
        # 删除旧的 datasets = 行
        new_content = re.sub(r'^\s*datasets\s*=.*$', '', new_content, flags=re.MULTILINE)
    
    # 2.2 添加模型导入
    model_var_name = f"{sanitized_model_name}_models"
    import_line = f"from opencompass.configs.models.mymodels.{config_filename.split('.')[0]} import models as {model_var_name}\n"
    
    if import_line not in new_content:
        if dataset_imports:
            last_dataset_import = dataset_imports[-1]
            pattern = re.escape(last_dataset_import) + r'\n?'
            match = re.search(pattern, new_content)
            if match:
                indent = '    '
                new_import = f"{indent}{import_line}"
                new_content = new_content[:match.end(0)] + new_import + new_content[match.end(0):]
        else:
            pattern = r'(with read_base\(\):\s+# 数据集和模型导入区域\s+)(?=\n|    )'
            match = re.search(pattern, new_content, re.DOTALL | re.MULTILINE)
            if match:
                indent = '    '
                new_import = f"{indent}{import_line}"
                new_content = new_content[:match.end(1)] + new_import + new_content[match.end(1):]
            else:
                print("警告: 未找到合适位置添加模型导入")
    
    # 3. 更新models变量（只保留当前指令中的模型）
    valid_model_vars = all_model_vars if all_model_vars else []
    
    # 删除旧的 models = 行
    new_content = re.sub(r'^\s*models\s*=.*$', '', new_content, flags=re.MULTILINE)

    # 4. 在文件末尾追加 datasets 和 models 定义
    if dataset_vars:
        datasets_str = " + ".join(dataset_vars)
        new_content = new_content.rstrip() + f"\n\ndatasets = {datasets_str}\n"
    else:
        new_content = new_content.rstrip() + f"\n\ndatasets = []\n"

    if valid_model_vars:
        models_str = " + ".join(valid_model_vars)
        new_content = new_content.rstrip() + f"\nmodels = {models_str}\n"
    else:
        new_content = new_content.rstrip() + f"\nmodels = []\n"

    # 写入更新后的内容
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"已更新评估脚本: {script_path}")


def main():
    global args
    parser = argparse.ArgumentParser(
        description='生成多个模型配置并处理多个数据集（首次生成前删除旧脚本）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--models', dest='hf_repos', nargs='+', required=True,
                      help='Hugging Face模型仓库路径列表（如 Qwen/Qwen2-1.5B-Instruct SeaLLMs/SeaLLMs-v3-7B-Chat）')
    parser.add_argument('--output', dest='eval_script_path', required=True,
                      help='评估脚本输出路径（例如 ./eval_tmp.py）')
    parser.add_argument('--datasets', nargs='+', help='数据集名称列表（如 tydiqa_gen_978d2a）')
    parser.add_argument('--type', dest='model_type', 
                      default='HuggingFacewithChatTemplate', help='模型类型')
    parser.add_argument('--max-out-len', type=int, default=1024,
                      help='最大输出长度')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='批处理大小')
    parser.add_argument('--num-gpus', type=int, default=4,
                      help='GPU数量')
    
    args = parser.parse_args()
    
    # 提前生成所有模型的元信息（变量名、配置文件名）
    model_metas = []
    for hf_repo in args.hf_repos:
        if hf_repo.startswith(('https://huggingface.co/', 'http://huggingface.co/')):
            hf_repo_clean = hf_repo.split('huggingface.co/')[-1].strip('/')
        else:
            hf_repo_clean = hf_repo
        model_name = hf_repo_clean.split('/')[-1]
        sanitized_model_name = model_name.replace('.', '_').replace('-', '_')
        config_filename = f"my_{sanitized_model_name}.py"
        model_metas.append({
            'hf_repo': hf_repo_clean,
            'model_name': model_name,
            'sanitized_model_name': sanitized_model_name,
            'config_filename': config_filename
        })
    all_model_basenames = [meta['config_filename'][:-3] for meta in model_metas]
    all_model_vars = [f"{meta['sanitized_model_name']}_models" for meta in model_metas]

    # 处理多个数据集
    dataset_vars = []
    dataset_filenames = []
    dataset_basenames = []
    if args.datasets:
        for dataset_name in args.datasets:
            try:
                dataset_var, dataset_filename, dataset_basename = copy_dataset_config(dataset_name)
                dataset_vars.append(dataset_var)
                dataset_filenames.append(dataset_filename)
                dataset_basenames.append(dataset_basename)
            except Exception as e:
                print(f"数据集 {dataset_name} 处理错误: {str(e)}")
                sys.exit(1)
    
    # >>>>>>>>>> 关键：循环处理模型时，标记第一个模型并传递is_first_model=True <<<<<<<<<<
    for idx, meta in enumerate(model_metas):
        generate_model_config(
            hf_repo=meta['hf_repo'],
            eval_script_path=args.eval_script_path,
            dataset_vars=dataset_vars,
            dataset_filenames=dataset_filenames,
            dataset_basenames=dataset_basenames,
            model_type=args.model_type,
            max_out_len=args.max_out_len,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
            all_model_basenames=all_model_basenames,
            all_model_vars=all_model_vars,
            is_first_model=(idx == 0)  # 仅第一个模型（idx=0）时删除旧脚本
        )

if __name__ == "__main__":
    main()
