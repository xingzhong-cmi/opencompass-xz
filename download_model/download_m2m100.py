from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# 重新下载并缓存模型和分词器
tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_1.2B')
model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_1.2B')

# 验证文件是否完整
import os
model_dir = '/root/.cache/huggingface/hub/models--facebook--m2m100_1.2B/snapshots/'
# 找到最新的快照目录（通常只有一个）
snapshot_dir = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))][0]
snapshot_path = os.path.join(model_dir, snapshot_dir)

# 检查关键文件
required_files = ['vocab.json', 'tokenizer.json', 'language_codes.json']
for file in required_files:
    if not os.path.exists(os.path.join(snapshot_path, file)):
        print(f"错误：仍缺少 {file}")
    else:
        print(f"已找到 {file}")
