import streamlit as st
from snownlp import SnowNLP
import jieba
from collections import Counter
import matplotlib.pyplot as plt

# 设置中文字体（解决matplotlib中文显示问题）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 页面配置
st.set_page_config(
    page_title="文本分析工具",
    page_icon="📝",
    layout="wide"
)

# 标题与说明
st.title("📝 文本分析Web应用")
st.markdown("### 支持功能：字符统计、词频分析、情感分析")

# 文本输入区域
text = st.text_area("请输入需要分析的文本", height=200)

if text:
    # 1. 基础统计
    st.subheader("1. 基础统计")
    total_chars = len(text)  # 总字符数
    total_chars_no_space = len(text.replace(" ", "").replace("\n", ""))  # 去除空格/换行后的字符数
    word_count = len(jieba.lcut(text))  # 分词后的词数

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总字符数", total_chars)
    with col2:
        st.metric("去除空格后字符数", total_chars_no_space)
    with col3:
        st.metric("分词后词数", word_count)

    # 2. 情感分析
    st.subheader("2. 情感分析")
    s = SnowNLP(text)
    sentiment_score = s.sentiments
    sentiment_label = "积极" if sentiment_score > 0.5 else "消极" if sentiment_score < 0.5 else "中性"

    st.slider(
        "情感倾向（0=消极，1=积极）",
        min_value=0.0,
        max_value=1.0,
        value=sentiment_score,
        disabled=True
    )
    st.write(f"情感评分：{sentiment_score:.4f} → 情感倾向：{sentiment_label}")

    # 3. 词频分析（排除停用词）
    st.subheader("3. 词频分析（TOP10）")
    # 简单停用词列表
    stop_words = {"的", "了", "是", "我", "你", "他", "她", "它", "在", "有", "就", "不", "和", "也", "都", "这", "那"}
    # 分词并过滤停用词、空字符
    words = [word for word in jieba.lcut(text) if word not in stop_words and len(word) > 1]
    word_freq = Counter(words).most_common(10)

    # 可视化词频
    if word_freq:
        words_list = [w[0] for w in word_freq]
        freq_list = [w[1] for w in word_freq]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(words_list, freq_list, color="#1f77b4")
        ax.set_xlabel("出现次数")
        ax.set_title("词频TOP10")
        st.pyplot(fig)

        # 显示详细词频表
        st.table(word_freq)
else:
    st.info("请输入文本后查看分析结果")
