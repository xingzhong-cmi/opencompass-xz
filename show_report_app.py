import streamlit as st
import os
import glob
from datetime import datetime
import re
import html

# 主目录设置
BASE_DIR = "/workspace/projects/opencompass/outputs/default/"

# 设置页面配置
st.set_page_config(
    page_title="OpenCompass 测试报告查看器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
        .report-container {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .file-browser {
            background-color: #e8f5e9;
            border-radius: 10px;
            padding: 15px;
            height: 400px;
            overflow-y: auto;
        }
        .sidebar-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2e7d32;
        }
        .btn-generate {
            background-color: #2e7d32;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .btn-generate:hover {
            background-color: #1b5e20;
        }
        .dir-selector {
            margin-bottom: 20px;
        }
        .summary-title {
            color: #2e7d32;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            border-bottom: 2px solid #c8e6c9;
            padding-bottom: 10px;
        }
        /* 为 text_area 定制样式 */
        .stTextArea textarea {
            font-family: monospace;
            font-size: 14px;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# 获取所有子目录并按时间排序
def get_test_dirs():
    if not os.path.exists(BASE_DIR):
        return []
    dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    try:
        dirs_with_time = []
        for d in dirs:
            try:
                dir_time = datetime.strptime(d, "%Y%m%d_%H%M%S")
                dirs_with_time.append((d, dir_time))
            except:
                dir_path = os.path.join(BASE_DIR, d)
                create_time = datetime.fromtimestamp(os.path.getctime(dir_path))
                dirs_with_time.append((d, create_time))
        dirs_with_time.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in dirs_with_time]
    except:
        return dirs

# 获取目录下的所有文件
def get_all_files_in_dir(dir_path):
    all_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), dir_path)
            all_files.append(rel_path)
    return sorted(all_files)

# 读取文件内容
def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败: {str(e)}"

# 解析Markdown表格，仅第五列及以后参与数值比较（标红最大值）
def parse_and_highlight_table(markdown_content):
    table_pattern = r'(\|.*?\|.*?\|\n)(\|.*?\|.*?\|\n)((?:\|.*?\|.*?\|\n)+)'
    matches = re.findall(table_pattern, markdown_content, re.DOTALL)
    if not matches:
        return markdown_content
    for match in matches:
        header_row, separator_row, data_rows_str = match
        data_rows = data_rows_str.strip().split('\n')
        processed_data_rows = []
        for row in data_rows:
            if not row.strip():
                continue
            cells = [cell.strip() for cell in row.split('|')[1:-1]]
            meta_cells, value_cells = cells[:4], cells[4:]
            values = []
            for cell in value_cells:
                try:
                    num_match = re.search(r'(\d+\.?\d*|\.\d+)(%?)', cell)
                    if num_match:
                        val = float(num_match.group(1))
                        if num_match.group(2) == '%':
                            val /= 100
                        values.append(val)
                    else:
                        values.append(None)
                except:
                    values.append(None)
            if values:
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    max_val = max(valid_values)
                    max_index = None
                    for i, (v, cell) in enumerate(zip(values, value_cells)):
                        if v == max_val:
                            num_match = re.search(r'(\d+\.?\d*|\.\d+)(%?)', cell)
                            if num_match:
                                current_val = float(num_match.group(1))
                                if num_match.group(2) == '%':
                                    current_val /= 100
                                if current_val == max_val:
                                    max_index = i
                                    break
                    if max_index is not None:
                        escaped_cell = html.escape(value_cells[max_index])
                        value_cells[max_index] = f'<span style="color:red;">{escaped_cell}</span>'
            new_cells = meta_cells + value_cells
            processed_data_rows.append('| ' + ' | '.join(new_cells) + ' |\n')
        new_table = header_row + separator_row + ''.join(processed_data_rows)
        markdown_content = markdown_content.replace(header_row + separator_row + data_rows_str, new_table)
    return markdown_content

# 初始化会话状态（确保关键变量存在）
def init_session_state():
    if 'selected_dir' not in st.session_state:
        test_dirs = get_test_dirs()
        st.session_state['selected_dir'] = test_dirs[0] if test_dirs else None
    if 'viewing_file' not in st.session_state:
        st.session_state['viewing_file'] = False
    if 'current_file' not in st.session_state:
        st.session_state['current_file'] = ""
    if 'file_content' not in st.session_state:
        st.session_state['file_content'] = ""
    if 'last_selected_file' not in st.session_state:
        st.session_state['last_selected_file'] = ""

# 主界面
def main():
    init_session_state()
    st.title("📊 OpenCompass 测试报告查看器")

    test_dirs = get_test_dirs()
    if not test_dirs:
        st.warning("未找到测试目录，请先运行测试或检查路径配置")
        return

    # 侧边栏：测试会话选择
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>测试会话选择</div>", unsafe_allow_html=True)
        selected_dir = st.selectbox(
            "选择测试会话",
            test_dirs,
            index=0,
            key="dir_selector",
            format_func=lambda x: x
        )
        # 当测试会话变更时，重置文件相关状态
        if selected_dir != st.session_state['selected_dir']:
            st.session_state['selected_dir'] = selected_dir
            st.session_state['viewing_file'] = False
            st.session_state['current_file'] = ""
            st.session_state['file_content'] = ""
            st.session_state['last_selected_file'] = ""

        generate_report = st.button("生成报告", key="generate_btn", type="primary")

        st.markdown("---")
        st.markdown("<div class='sidebar-header'>文件浏览器</div>", unsafe_allow_html=True)

        # 获取当前会话目录下的所有文件
        current_dir = os.path.join(BASE_DIR, st.session_state['selected_dir'])
        all_files = get_all_files_in_dir(current_dir)

        if all_files:
            # 文件选择框：变更时重置查看状态
            selected_file = st.selectbox(
                "选择文件查看",
                all_files,
                key="file_selector",
                # 当文件选择变更时，触发状态重置
                on_change=lambda: st.session_state.update({
                    'viewing_file': False,
                    'current_file': "",
                    'file_content': ""
                })
            )

            # 查看按钮：点击时强制更新会话状态
            if st.button("查看文件内容", key="view_file_btn"):
                # 验证文件路径有效性
                file_path = os.path.join(current_dir, selected_file)
                if os.path.exists(file_path):
                    st.session_state['file_content'] = read_file_content(file_path)
                    st.session_state['current_file'] = selected_file
                    st.session_state['viewing_file'] = True
                    st.session_state['last_selected_file'] = selected_file
                else:
                    st.warning(f"文件不存在：{selected_file}")

    # 主区域：生成报告
    if generate_report:
        st.session_state['viewing_file'] = False  # 生成报告时隐藏文件内容
        current_dir = os.path.join(BASE_DIR, st.session_state['selected_dir'])
        summary_files = glob.glob(os.path.join(current_dir, "summary", "summary_*.md"))

        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.markdown("<div class='summary-title'>测试摘要报告</div>", unsafe_allow_html=True)
        if summary_files:
            summary_file = max(summary_files, key=os.path.getctime)
            summary_content = read_file_content(summary_file)
            processed_content = parse_and_highlight_table(summary_content)
            st.markdown(processed_content, unsafe_allow_html=True)
        else:
            st.warning("未找到测试摘要文件")
        st.markdown("</div>", unsafe_allow_html=True)

    # 主区域：显示文件内容
    if st.session_state['viewing_file'] and st.session_state['current_file']:
        st.markdown("---")
        st.subheader(f"📄 文件内容: {st.session_state['current_file']}")
        # 使用 text_area 安全显示纯文本，支持滚动
        st.text_area(
            label="\u200b",
            value=st.session_state['file_content'],
            height=600,
            key="file_viewer",
            disabled=True  # 设为只读，避免用户编辑
        )

if __name__ == "__main__":
    main()
