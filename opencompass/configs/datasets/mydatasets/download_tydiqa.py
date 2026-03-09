from datasets import load_dataset

dataset = load_dataset("tydiqa", "primary_task")  # 下载主任务数据集
# 保存到本地 OpenCompass 数据目录
dataset.save_to_disk("./data/tydiqa/")
