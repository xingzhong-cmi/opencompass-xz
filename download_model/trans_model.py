import streamlit as st
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import time
import uuid
from streamlit.components.v1 import html

# 设置页面配置
st.set_page_config(
    page_title="M2M100 多语言翻译工具",
    page_icon="🌐",
    layout="wide"
)

# 页面样式优化
st.markdown("""
<style>
    .stTextArea textarea {
        font-family: sans-serif;
    }
    .copy-success {
        color: #28a745;
        font-weight: bold;
    }
    .history-item {
        border-left: 3px solid #007bff;
        padding-left: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.title("🌐 M2M100 多语言翻译工具")
st.write("高效多语言翻译，支持连续输入、结果复制和历史记录")

# 支持的语言映射表（扩展更多常用语言）
lang_mapping = {
    "en": "英语",
    "zh": "中文",
    "id": "印尼语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "ru": "俄语",
    "ja": "日语",
    "ko": "韩语",
    "ar": "阿拉伯语",
    "vi": "越南语",
    "th": "泰语",
    "pt": "葡萄牙语",
    "it": "意大利语",
    "nl": "荷兰语",
    "tr": "土耳其语"
}

# 侧边栏 - 模型设置
with st.sidebar:
    st.header("模型设置")
    
    # 模型路径设置
    model_path = st.text_input(
        "模型本地路径",
        value="/root/.cache/modelscope/hub/models/cubeai/m2m100_1.2b"
    )
    
    # 设备选择
    device_option = st.radio(
        "计算设备",
        ("自动选择 (推荐)", "仅使用CPU", "仅使用GPU")
    )
    
    # 根据选择确定设备
    if device_option == "仅使用CPU":
        device = "cpu"
    elif device_option == "仅使用GPU":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            st.warning("未检测到可用GPU，将使用CPU")
    else:  # 自动选择
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    st.info(f"当前使用设备: {device}")
    
    # 翻译参数设置
    st.subheader("翻译参数")
    max_length = st.slider("最大输出长度", 50, 1000, 300)
    num_beams = st.slider("Beam搜索数量", 1, 10, 5)
    temperature = st.slider("生成多样性（temperature）", 0.1, 2.0, 1.0)
    
    # 加载模型按钮
    if st.button("加载模型", use_container_width=True):
        with st.spinner("正在加载模型..."):
            try:
                # 加载分词器和模型
                tokenizer = M2M100Tokenizer.from_pretrained(model_path)
                model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(device)
                
                # 存储到session_state
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.success("模型加载成功！")
            except Exception as e:
                st.error(f"加载模型失败: {str(e)}")
                # 显示更详细的错误信息用于调试
                with st.expander("查看详细错误"):
                    st.exception(e)

# 语言选择
col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox(
        "源语言",
        list(lang_mapping.keys()),
        format_func=lambda x: lang_mapping[x]
    )
with col2:
    target_lang = st.selectbox(
        "目标语言",
        list(lang_mapping.keys()),
        index=1,  # 默认英语
        format_func=lambda x: lang_mapping[x]
    )

# 交换语言按钮
if st.button("🔄 交换语言", use_container_width=False):
    # 交换源语言和目标语言
    source_lang, target_lang = target_lang, source_lang
    # 刷新页面以更新选择框
    st.experimental_rerun()

# 输入区域
st.subheader("输入文本")
input_text = st.text_area(
    "请输入要翻译的文本",
    height=150,
    placeholder="在这里输入文本（支持连续输入多行），然后点击翻译按钮...",
    key="input_text_area"
)

# 翻译按钮
if st.button("翻译", use_container_width=True):
    # 检查模型是否已加载
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        st.error("请先在侧边栏点击'加载模型'按钮")
    elif not input_text.strip():
        st.warning("请输入要翻译的文本")
    else:
        with st.spinner("正在翻译中..."):
            try:
                start_time = time.time()
                
                # 获取模型和分词器
                model = st.session_state.model
                tokenizer = st.session_state.tokenizer
                
                # 设置源语言
                tokenizer.src_lang = source_lang
                
                # 编码输入文本
                encoded_input = tokenizer(
                    input_text, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                ).to(device)
                
                # 生成翻译
                generated_tokens = model.generate(
                    **encoded_input,
                    forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    early_stopping=True
                )
                
                # 解码结果
                translated_text = tokenizer.batch_decode(
                    generated_tokens, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
                
                # 计算耗时
                end_time = time.time()
                duration = end_time - start_time
                
                # 存储当前翻译结果
                st.session_state.current_translation = translated_text
                result_key = f"trans_result_{str(uuid.uuid4())[:8]}"
                
                # 显示结果
                st.subheader("翻译结果（可复制）")
                result_container = st.text_area(
                    label="翻译完成",
                    value=translated_text,
                    height=150,
                    disabled=False,
                    key=result_key,
                    help="点击文本区域后，按Ctrl+C（Windows）或Cmd+C（Mac）复制"
                )
                
                # 一键复制功能（优化版）
                copy_js = f"""
                <script>
                function copyToClipboard() {{
                    // 获取文本区域
                    const textarea = document.querySelector('textarea[aria-label="翻译完成"]');
                    if (textarea) {{
                        // 选中内容
                        textarea.select();
                        // 复制到剪贴板
                        navigator.clipboard.writeText(textarea.value).then(() => {{
                            // 显示成功消息
                            const successDiv = document.createElement('div');
                            successDiv.className = 'copy-success';
                            successDiv.textContent = '✓ 翻译结果已复制到剪贴板！';
                            successDiv.style.margin = '10px 0';
                            textarea.parentNode.parentNode.appendChild(successDiv);
                            
                            // 3秒后移除成功消息
                            setTimeout(() => {{
                                successDiv.remove();
                            }}, 3000);
                        }}).catch(err => {{
                            console.error('无法复制: ', err);
                            alert('复制失败，请手动复制');
                        }});
                    }}
                }}
                </script>
                <button onclick="copyToClipboard()" style="width: 100%; padding: 8px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    📋 一键复制翻译结果
                </button>
                """
                html(copy_js, height=40)
                
                # 显示翻译信息
                st.info(f"翻译完成 | 耗时: {duration:.2f}秒 | 设备: {device} | 语言: {lang_mapping[source_lang]}→{lang_mapping[target_lang]}")
                
                # 更新翻译历史
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    "source": input_text,
                    "translated": translated_text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # 限制历史记录数量，避免占用过多内存
                if len(st.session_state.history) > 50:
                    st.session_state.history.pop(0)  # 移除最早的记录
                
            except Exception as e:
                st.error(f"翻译出错: {str(e)}")
                with st.expander("查看详细错误"):
                    st.exception(e)

# 翻译历史
if "history" in st.session_state and len(st.session_state.history) > 0:
    with st.expander(f"翻译历史 ({len(st.session_state.history)})", expanded=False):
        # 清空历史按钮
        if st.button("清空历史记录", use_container_width=True):
            st.session_state.history = []
            st.experimental_rerun()
            
        # 显示历史记录
        for i, item in enumerate(reversed(st.session_state.history)):
            hist_key = f"hist_result_{i}_{str(uuid.uuid4())[:6]}"
            with st.container():
                st.markdown(f"**{item['time']}** | {lang_mapping[item['source_lang']]} → {lang_mapping[item['target_lang']]}")
                
                # 修复：创建2列而不是1列
                col_source, col_trans = st.columns(2)
                
                with col_source:
                    st.text_area(
                        f"原文 #{len(st.session_state.history)-i}", 
                        item["source"], 
                        height=80, 
                        disabled=True,
                        key=f"source_{hist_key}"
                    )
                
                with col_trans:
                    st.text_area(
                        f"译文 #{len(st.session_state.history)-i}（可复制）", 
                        item["translated"], 
                        height=80, 
                        disabled=False,
                        key=f"trans_{hist_key}"
                    )
                
                # 为历史记录添加复制按钮
                copy_hist_js = f"""
                <script>
                function copyHist{i}() {{
                    const textarea = document.querySelector('textarea[aria-label="译文 #{len(st.session_state.history)-i}（可复制）"]');
                    if (textarea) {{
                        textarea.select();
                        navigator.clipboard.writeText(textarea.value).then(() => {{
                            alert('历史译文已复制到剪贴板！');
                        }}).catch(err => {{
                            console.error('无法复制: ', err);
                            alert('复制失败，请手动复制');
                        }});
                    }}
                }}
                </script>
                <button onclick="copyHist{i}()" style="margin-bottom: 20px; padding: 4px 8px; background-color: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    📋 复制此译文
                </button>
                """
                html(copy_hist_js, height=30)
                
            st.divider()

# 页脚
st.markdown("---")
st.caption("使用 M2M100-1.2B 模型提供翻译服务 | 支持多种语言互译")
    
