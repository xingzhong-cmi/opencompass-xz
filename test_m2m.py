import os
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def test_m2m100_translation():
    # 正确的模型快照路径（包含完整文件）
    model_path = "/root/.cache/huggingface/hub/models--facebook--m2m100_1.2B/snapshots/7b36184180524c1a1bbfa37f120a608046250b98"
    
    # 验证路径存在
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在 - {model_path}")
        return
    
    # 加载分词器和模型
    print("加载模型和分词器...")
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(model_path)
        model = M2M100ForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        print(f"加载失败：{str(e)}")
        return
    
    # 语言设置（印尼语→泰语）
    src_lang = "id"  # 印尼语代码
    tgt_lang = "th"  # 泰语代码
    tokenizer.src_lang = src_lang
    print(f"源语言: {src_lang} (印尼语)")
    print(f"目标语言ID: {tokenizer.get_lang_id(tgt_lang)} (泰语)\n")
    
    # 测试用例
    test_cases = [
        "Saya suka makan nasi goreng.",  # 我喜欢吃炒饭
        "Hari ini cuaca sangat cerah.",  # 今天天气很晴朗
        "Universitas ini memiliki perpustakaan besar.",  # 这所大学有一个大图书馆
        "Dia bekerja sebagai dokter di rumah sakit terdekat.",  # 他在最近的医院当医生
        "Kita akan pergi ke pantai besok.",  # 我们明天要去海滩
        "Sebagian pasien kemungkinan ketularan penyakit itu di rumah sakit, menurut perkiraan Dr. Moll, dan paling sedikit dua orang adalah petugas kesehatan di rumah sakit."
        ]
    
    # 执行翻译
    print("开始翻译测试...\n")
    for i, text in enumerate(test_cases, 1):
        print(f"测试用例 {i}:")
        print(f"印尼语原文: {text}")
        
        # 编码输入文本
        encoded_input = tokenizer(text, return_tensors="pt")
        

        print("tokenizer.get_lang_id(tgt_lang) is ", tokenizer.get_lang_id(tgt_lang))
        
    
        print("encoded_input", encoded_input)
        print("tokenizer",tokenizer)
        print("model.generate", model.generate)

        # 生成泰语翻译（强制目标语言）
        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),  # 强制泰语起始
            max_length=100,
            num_beams=4  # 提升翻译流畅度
        )
        
        # 解码并输出结果
        print("tokenizer = ", tokenizer)
        print("generated_tokens = ", generated_tokens)
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print(f"泰语译文: {translated_text}\n")

if __name__ == "__main__":
    test_m2m100_translation()
