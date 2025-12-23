import streamlit as st
import requests
from bs4 import BeautifulSoup
import jieba
from collections import Counter
import re
from pyecharts import options as opts
from pyecharts.charts import WordCloud, Bar, Line, Pie, Radar, Scatter, HeatMap, TreeMap
from streamlit_echarts import st_pyecharts
import numpy as np

# 页面配置
st.set_page_config(page_title="文本词频分析工具", layout="wide")

# 侧边栏配置
st.sidebar.title("可视化筛选")
chart_type = st.sidebar.selectbox(
    "选择图表类型",
    [
        "词云图", "柱状图(前20)", "折线图(前20)", "饼图(前10)",
        "雷达图(前8)", "散点图(前20)", "热力图(前15)", "矩形树图(前15)"
    ]
)
min_freq = st.sidebar.slider("过滤低频词（最小词频）", 1, 20, 2)

# 主页面标题
st.title("URL文本采集与词频分析系统")

# 1. URL输入区域
url = st.text_input("请输入文章URL地址", placeholder="例如：https://www.example.com/article")
submit_btn = st.button("开始分析")

# 定义停用词（基础版）
STOP_WORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也',
    '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
    '他', '她', '它', '我们', '你们', '他们', '这里', '那里', '然后', '但是', '所以', '因为',
    '对于', '关于', '虽然', '如果', '就是', '还有', '什么', '怎么', '为什么', '多少', '几'
])


def fetch_url_content(url):
    """抓取URL文本内容"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')

        # 提取正文（通用规则，适配大多数文章页面）
        text = ''
        # 尝试常见的正文标签
        for tag in ['article', 'div[class*="content"]', 'div[class*="article"]', 'main', 'body']:
            elements = soup.select(tag)
            if elements:
                text = '\n'.join([elem.get_text(strip=True) for elem in elements])
                if text:
                    break

        # 清洗文本
        text = re.sub(r'\s+', ' ', text)  # 去除多余空格
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)  # 只保留中文、字母、数字
        return text
    except Exception as e:
        st.error(f"抓取URL失败：{str(e)}")
        return ""


def analyze_word_freq(text, min_freq):
    """分词并统计词频"""
    if not text:
        return {}

    # 分词
    words = jieba.lcut(text)
    # 过滤停用词和短词，以及低频词
    filtered_words = [
        word for word in words
        if len(word) > 1 and word not in STOP_WORDS
    ]
    # 统计词频
    word_freq = Counter(filtered_words)
    # 过滤低频词
    word_freq = {k: v for k, v in word_freq.items() if v >= min_freq}
    # 按词频排序
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
    return sorted_word_freq


def create_chart(word_freq, chart_type):
    """根据选择创建不同的图表"""
    if not word_freq:
        st.warning("暂无足够的词频数据展示")
        return

    # 取前N个词（根据不同图表调整）
    top_n = {
        "词云图": len(word_freq),
        "柱状图(前20)": 20,
        "折线图(前20)": 20,
        "饼图(前10)": 10,
        "雷达图(前8)": 8,
        "散点图(前20)": 20,
        "热力图(前15)": 15,
        "矩形树图(前15)": 15
    }[chart_type]

    top_words = list(word_freq.items())[:top_n]
    words = [item[0] for item in top_words]
    freqs = [item[1] for item in top_words]

    # 创建不同类型的图表
    if chart_type == "词云图":
        c = (
            WordCloud()
                .add("", top_words, word_size_range=[20, 100])
                .set_global_opts(title_opts=opts.TitleOpts(title="词频词云图"))
        )
        st_pyecharts(c)

    elif chart_type == "柱状图(前20)":
        c = (
            Bar()
                .add_xaxis(words)
                .add_yaxis("词频", freqs)
                .reversal_axis()  # 横向柱状图
                .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名前20柱状图"),
                xaxis_opts=opts.AxisOpts(name="词频"),
                yaxis_opts=opts.AxisOpts(name="词汇")
            )
        )
        st_pyecharts(c)

    elif chart_type == "折线图(前20)":
        c = (
            Line()
                .add_xaxis(words)
                .add_yaxis("词频", freqs, markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]))
                .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名前20折线图"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45))
            )
        )
        st_pyecharts(c)

    elif chart_type == "饼图(前10)":
        c = (
            Pie()
                .add("", top_words)
                .set_global_opts(title_opts=opts.TitleOpts(title="词频排名前10饼图"))
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
        )
        st_pyecharts(c)

    elif chart_type == "雷达图(前8)":
        # 雷达图需要调整数据格式
        max_freq = max(freqs) if freqs else 1
        c = (
            Radar()
                .add_schema(schema=[opts.RadarIndicatorItem(name=word, max_=max_freq) for word in words])
                .add("词频", [freqs])
                .set_global_opts(title_opts=opts.TitleOpts(title="词频排名前8雷达图"))
        )
        st_pyecharts(c)

    elif chart_type == "散点图(前20)":
        c = (
            Scatter()
                .add_xaxis(words)
                .add_yaxis("词频", freqs)
                .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名前20散点图"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
                yaxis_opts=opts.AxisOpts(name="词频")
            )
        )
        st_pyecharts(c)

    elif chart_type == "热力图(前15)":
        # 热力图数据格式：[[行, 列, 值], ...]
        heat_data = []
        for i in range(len(words)):
            heat_data.append([0, i, freqs[i]])  # 单行热力图
        c = (
            HeatMap()
                .add_xaxis(words)
                .add_yaxis("词频", ["频次"], heat_data)
                .set_global_opts(
                title_opts=opts.TitleOpts(title="词频排名前15热力图"),
                visualmap_opts=opts.VisualMapOpts(min_=min(freqs), max_=max(freqs))
            )
        )
        st_pyecharts(c)

    elif chart_type == "矩形树图(前15)":
        # 矩形树图数据格式
        treemap_data = [{"name": word, "value": freq} for word, freq in top_words]
        c = (
            TreeMap()
                .add("", treemap_data)
                .set_global_opts(title_opts=opts.TitleOpts(title="词频排名前15矩形树图"))
        )
        st_pyecharts(c)


# 主逻辑执行
if submit_btn and url:
    with st.spinner("正在抓取URL内容..."):
        text = fetch_url_content(url)

    if text:
        with st.spinner("正在分词并统计词频..."):
            word_freq = analyze_word_freq(text, min_freq)

        # 展示词频排名前20
        st.subheader("词频排名前20词汇")
        top_20 = list(word_freq.items())[:20]
        for idx, (word, freq) in enumerate(top_20, 1):
            st.write(f"{idx}. {word} - 出现次数：{freq}")

        # 展示可视化图表
        st.subheader(f"可视化展示：{chart_type}")
        create_chart(word_freq, chart_type)

        # 展示原始文本预览（可选）
        with st.expander("查看抓取的文本内容（前2000字）"):
            st.text(text[:2000] + "..." if len(text) > 2000 else text)
elif submit_btn:
    st.warning("请输入有效的URL地址")

# 说明文档
st.sidebar.markdown("""
### 使用说明
1. 输入文章URL地址
2. 调整低频词过滤阈值
3. 选择需要展示的图表类型
4. 点击"开始分析"按钮
5. 查看词频排名和可视化图表

### 支持的图表类型
- 词云图：直观展示词汇出现频率
- 柱状图：对比前20词汇的词频
- 折线图：展示前20词汇的词频趋势
- 饼图：展示前10词汇的占比
- 雷达图：多维展示前8词汇的词频
- 散点图：展示前20词汇的词频分布
- 热力图：颜色深浅展示前15词汇词频
- 矩形树图：层级展示前15词汇词频
""")
