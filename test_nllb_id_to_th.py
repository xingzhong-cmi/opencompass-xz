import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def test_nllb_translation():
    # 模型名称（自动下载完整文件）
    model_name = "facebook/nllb-200-1.3B"
    print(f"使用模型: {model_name}")

    # 加载分词器和模型
    print("加载NLLB模型和分词器...")
    try:
        # 加载分词器（显式指定源语言和目标语言）
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            src_lang="ind_Latn",  # 印尼语
            tgt_lang="tha_Thai",  # 泰语
            use_fast=True  # 保留Fast分词器（速度快）
        )
        # 加载模型
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype="auto"
        )
    except Exception as e:
        print(f"加载失败：{str(e)}")
        return

    # 修复：兼容Fast分词器的语言标识格式（关键修改）
    tgt_lang = "tha_Thai"
    try:
        # 方法1：尝试普通分词器的 lang_code_to_id
        if hasattr(tokenizer, "lang_code_to_id"):
            tgt_lang_id = tokenizer.lang_code_to_id[tgt_lang]
        else:
            # 方法2：Fast分词器可能直接用语言代码作为token（无尖括号）
            tgt_lang_id = tokenizer.get_vocab()[tgt_lang]  # 去掉尖括号
    except KeyError:
        print("Two method fail...")

    print(f"源语言: ind_Latn（印尼语）")
    print(f"目标语言: {tgt_lang}（泰语）")
    print(f"目标语言Token ID: {tgt_lang_id}\n")

    # 测试用例
    test_cases = [
        "Saya suka makan nasi goreng.",
        "Hari ini cuaca sangat cerah.",
        "Universitas ini memiliki perpustakaan besar."
    ]

    # 执行翻译
    print("开始翻译测试...\n")
    for i, text in enumerate(test_cases, 1):
        print(f"测试用例 {i}:")
        print(f"印尼语原文: {text}")

        # 编码输入
        encoded_input = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # 生成翻译（强制泰语起始）
        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True
        )

        # 解码输出
        translated_text = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        print(f"泰语译文: {translated_text}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    test_nllb_translation()
