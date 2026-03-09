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
                          num_gpus=4):
    """生成模型配置文件"""
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
    
    # 获取当前模型的基础文件名（不含.py）
    current_model_basename = config_filename[:-3]
    
    update_eval_script(
        script_path=eval_script_path,
        model_name=model_name,
        sanitized_model_name=sanitized_model_name,
        config_filename=config_filename,
        current_model_basename=current_model_basename,
        all_model_basenames=[f"my_{name.replace('.', '_').replace('-', '_')}" for name in [repo.split('/')[-1] for repo in args.hf_repos]],
        dataset_vars=dataset_vars,
        dataset_filenames=dataset_filenames,
        dataset_basenames=dataset_basenames
    )

def update_eval_script(script_path, 
                      model_name, 
                      sanitized_model_name, 
                      config_filename,
                      current_model_basename,
                      all_model_basenames,
                      dataset_vars, 
                      dataset_filenames,
                      dataset_basenames):
    """更新评估脚本，只保留当前指令中指定的模型和数据"""
    script_path = os.path.abspath(script_path)
    
    # >>>>>>>>>> �~V��~^�~Z�~E~H�~H| �~Y�已�~X�~\��~Z~D�~D~Z�~\��~V~G件 <<<<<<<<<<
    if os.path.exists(script_path):
        os.remove(script_path)
        print(f"已�~H| �~Y��~W��~Z~D�~D估�~D~Z�~\�: {script_path}")

    # 确保评估脚本所在目录存在
    script_dir = os.path.dirname(script_path)
    os.makedirs(script_dir, exist_ok=True)
    
    # 初始化脚本内容（如果不存在）
    if not os.path.exists(script_path):
        print(f"创建新评估脚本: {script_path}")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("from mmengine.config import read_base\n\n")
            f.write("with read_base():\n")
            f.write("    # 数据集和模型导入区域\n")
            f.write("\n")
            f.write("datasets = []\n")
            f.write("models = []\n")
    
    # 读取现有内容
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    
    # 1. 清理现有导入（只保留当前指令中需要的）
    # 1.1 清理数据集导入
    if dataset_basenames:
        # 保留指定数据集的导入，移除其他数据集导入
        dataset_import_pattern = re.compile(r'from opencompass\.configs\.datasets\.mydatasets\.[^\s]+ import [^\s]+\n?', re.MULTILINE)
        all_dataset_imports = dataset_import_pattern.findall(new_content)
        
        # 筛选需要保留的数据集导入
        keep_dataset_imports = []
        for imp in all_dataset_imports:
            for basename in dataset_basenames:
                if basename in imp:
                    keep_dataset_imports.append(imp)
                    break
        
        # 替换所有数据集导入为需要保留的
        if all_dataset_imports:
            new_content = dataset_import_pattern.sub('', new_content)
            # 将保留的导入添加回去（放在with块开头）
            pattern = r'(with read_base\(\):\s+# 数据集和模型导入区域\s+)(?=\n|    )'
            match = re.search(pattern, new_content, re.DOTALL | re.MULTILINE)
            if match:
                new_content = new_content[:match.end(1)] + ''.join([f"    {imp}" for imp in keep_dataset_imports]) + new_content[match.end(1):]
    
    # 1.2 清理模型导入
    if all_model_basenames:
        # 保留指定模型的导入，移除其他模型导入
        model_import_pattern = re.compile(r'from opencompass\.configs\.models\.mymodels\.[^\s]+ import models as [^\s]+\n?', re.MULTILINE)
        all_model_imports = model_import_pattern.findall(new_content)
        
        # 筛选需要保留的模型导入
        keep_model_imports = []
        for imp in all_model_imports:
            for basename in all_model_basenames:
                if basename in imp:
                    keep_model_imports.append(imp)
                    break
        
        # 替换所有模型导入为需要保留的
        if all_model_imports:
            new_content = model_import_pattern.sub('', new_content)
            # 将保留的导入添加到数据集导入之后
            if dataset_basenames and keep_dataset_imports:
                last_dataset_import = keep_dataset_imports[-1]
                pattern = re.escape(last_dataset_import) + r'\n?'
                match = re.search(pattern, new_content)
                if match:
                    new_content = new_content[:match.end(0)] + ''.join([f"    {imp}" for imp in keep_model_imports]) + new_content[match.end(0):]
            else:
                # 没有数据集时直接添加到with块
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
                # 插入到with块最上方
                pattern = r'(with read_base\(\):\s+# 数据集和模型导入区域\s+)(?=\n|    )'
                match = re.search(pattern, new_content, re.DOTALL | re.MULTILINE)
                if match:
                    indent = '    '
                    new_import = f"{indent}{import_line}"
                    new_content = new_content[:match.end(1)] + new_import + new_content[match.end(1):]
                else:
                    print(f"警告: 未找到合适位置添加数据集 {dataset_var} 导入")
        
        # >>>>>>>>>> 修改点：先删除旧的 datasets = 行 <<<<<<<<<<
        new_content = re.sub(r'^\s*datasets\s*=.*$', '', new_content, flags=re.MULTILINE)
    
    # 2.2 添加模型导入
    model_var_name = f"{sanitized_model_name}_models"
    import_line = f"from opencompass.configs.models.mymodels.{config_filename.split('.')[0]} import models as {model_var_name}\n"
    
    if import_line not in new_content:
        # 找到最后一个数据集导入的位置
        if dataset_imports:
            last_dataset_import = dataset_imports[-1]
            pattern = re.escape(last_dataset_import) + r'\n?'
            match = re.search(pattern, new_content)
            if match:
                indent = '    '
                new_import = f"{indent}{import_line}"
                new_content = new_content[:match.end(0)] + new_import + new_content[match.end(0):]
        else:
            # 没有数据集时直接导入到with块
            pattern = r'(with read_base\(\):\s+# 数据集和模型导入区域\s+)(?=\n|    )'
            match = re.search(pattern, new_content, re.DOTALL | re.MULTILINE)
            if match:
                indent = '    '
                new_import = f"{indent}{import_line}"
                new_content = new_content[:match.end(1)] + new_import + new_content[match.end(1):]
            else:
                print("警告: 未找到合适位置添加模型导入")
    
    # 3. 更新models变量（只保留当前指令中的模型）
    # 收集所有需要保留的模型变量
    valid_model_vars = [f"{name.replace('.', '_').replace('-', '_')}_models" for name in [repo.split('/')[-1] for repo in args.hf_repos]]
    
    # >>>>>>>>>> 修改点：先删除旧的 models = 行 <<<<<<<<<<
    new_content = re.sub(r'^\s*models\s*=.*$', '', new_content, flags=re.MULTILINE)

    # 4. 在文件末尾追加 datasets = ... 和 models = ...，确保无缩进
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
    global args  # 使args在update_eval_script中可用
    parser = argparse.ArgumentParser(
        description='生成多个模型配置并处理多个数据集，仅保留指定的模型和数据',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- 修改开始 ---
    # 使用 --models 来明确指定模型列表
    parser.add_argument('--models', dest='hf_repos', nargs='+', required=True,
                      help='Hugging Face模型仓库路径列表')
    # 使用 --output 来明确指定输出文件
    parser.add_argument('--output', dest='eval_script_path', required=True,
                      help='评估脚本输出路径（例如 ./eval_tmp.py）')
    # --- 修改结束 ---
    
    parser.add_argument('--datasets', nargs='+', help='数据集名称列表（如tydiqa_gen_978d2a）')
    
    # 其他参数
    parser.add_argument('--type', dest='model_type', 
                      default='HuggingFacewithChatTemplate', help='模型类型')
    parser.add_argument('--max-out-len', type=int, default=1024,
                      help='最大输出长度')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='批处理大小')
    parser.add_argument('--num-gpus', type=int, default=4,
                      help='GPU数量')
    
    args = parser.parse_args()
    
    # 处理多个数据集
    dataset_vars = []
    dataset_filenames = []
    dataset_basenames = []  # 存储数据集文件名（不含.py）
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
    
    # 处理多个模型
    for hf_repo in args.hf_repos:
        generate_model_config(
            hf_repo=hf_repo,
            eval_script_path=args.eval_script_path,
            dataset_vars=dataset_vars,
            dataset_filenames=dataset_filenames,
            dataset_basenames=dataset_basenames,
            model_type=args.model_type,
            max_out_len=args.max_out_len,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus
        )

if __name__ == "__main__":
    main()
    
