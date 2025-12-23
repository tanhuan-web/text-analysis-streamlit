import streamlit as st
import jieba
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

# -------------------------- 基础配置 --------------------------
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 页面配置
st.set_page_config(
    page_title="文本分析工具（支持URL爬取）",
    page_icon="📝",
    layout="wide"
)

# -------------------------- 核心函数 --------------------------
# 1. 网页内容爬取函数
def crawl_webpage(url):
    """爬取指定URL的网页文本内容"""
    try:
        # 请求头（模拟浏览器，避免被反爬）
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # 发送请求（设置超时）
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding  # 自动识别编码
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 提取网页主要文本（优先取正文，过滤导航/广告等）
        # 常见正文标签：p、article、div[class*="content"]、main
        text_elements = soup.find_all(["p", "article", "main"])
        # 补充：过滤class/id包含广告/导航的元素
        text_elements = [elem for elem in text_elements if not any(
            kw in elem.get("class", []) + [elem.get("id", "")] 
            for kw in ["ad", "nav", "menu", "footer", "header"]
        )]
        
        # 提取文本并清洗
        raw_text = "\n".join([elem.get_text().strip() for elem in text_elements])
        if not raw_text:  # 若未提取到，取整个网页文本
            raw_text = soup.get_text().strip()
        
        return raw_text
    
    except requests.exceptions.Timeout:
        st.error("❌ 请求超时：网页响应时间过长")
        return ""
    except requests.exceptions.ConnectionError:
        st.error("❌ 连接失败：无法访问该网址（检查网络/网址是否正确）")
        return ""
    except requests.exceptions.InvalidURL:
        st.error("❌ 无效URL：请输入完整的网址（如 https://www.xxx.com）")
        return ""
    except Exception as e:
        st.error(f"❌ 爬取失败：{str(e)}")
        return ""

# 2. 文本预处理函数
def preprocess_text(text):
    """清洗文本并分词"""
    # 去除标点、数字、空白符、特殊符号
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", text)
    # 分词
    words = jieba.lcut(text)
    # 过滤停用词和单字
    stop_words = {"的", "了", "是", "在", "和", "有", "我", "你", "他", "她", "它", 
                  "就", "都", "而", "及", "与", "之", "于", "也", "还", "这", "那",
                  "为", "以", "可", "将", "对", "能", "会", "要", "把", "被", "所"}
    words = [word for word in words if word not in stop_words and len(word) > 1]
    return words

# -------------------------- 页面交互 --------------------------
st.title("📝 文本分析工具（支持URL爬取）")
st.markdown("### 支持：网页内容爬取、分词、词频统计、关键词提取、词云生成")

# 选择输入方式
input_mode = st.radio("请选择输入方式", ["手动输入文本", "输入URL爬取网页内容"], horizontal=True)

text_source = ""
if input_mode == "手动输入文本":
    # 手动输入文本
    text_source = st.text_area(
        "请输入需要分析的文本（支持中文）",
        height=200,
        placeholder="例如：人工智能是未来科技的核心方向，人工智能正在改变各行各业..."
    )
else:
    # URL输入
    url = st.text_input(
        "请输入网页URL（需包含 https://）",
        placeholder="例如：https://www.baidu.com/news/xxx.html"
    )
    # 爬取按钮
    if st.button("📤 爬取网页内容", type="secondary"):
        if url:
            with st.spinner("正在爬取网页内容..."):
                text_source = crawl_webpage(url)
                # 展示爬取结果（供确认）
                st.subheader("爬取到的内容预览")
                st.text_area("内容预览（前500字）", text_source[:500], height=150)
        else:
            st.warning("⚠️ 请输入有效的URL！")

# 分析按钮
if st.button("🚀 开始分析", type="primary"):
    if not text_source.strip():
        st.warning("⚠️ 请先输入文本/爬取网页内容！")
    else:
        # 预处理
        words = preprocess_text(text_source)
        if not words:
            st.warning("⚠️ 无有效分析内容（请检查文本是否包含中文词汇）！")
        else:
            # 词频统计
            word_count = Counter(words)
            top_20 = word_count.most_common(20)
            
            # 分栏展示结果
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("1. 词频统计（TOP20）")
                df = pd.DataFrame(top_20, columns=["词汇", "出现次数"])
                st.dataframe(df, use_container_width=True)
                
                # 词频柱状图
                st.subheader("2. 词频可视化（TOP10）")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(df["词汇"][:10], df["出现次数"][:10], color="#1f77b4")
                ax.set_xlabel("词汇", fontsize=12)
                ax.set_ylabel("出现次数", fontsize=12)
                ax.set_title("TOP10词汇词频分布", fontsize=14)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                st.subheader("3. 核心关键词（TOP8）")
                keywords = [w[0] for w in top_20[:8]]
                st.markdown(f"**{', '.join(keywords)}**")
                
                # 词云生成
                st.subheader("4. 词云展示")
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="white",
                    font_path="simhei.ttf",  # 兼容云端中文显示
                    max_words=100,
                    colormap="viridis"
                ).generate(" ".join(words))
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.imshow(wordcloud, interpolation="bilinear")
                ax2.axis("off")
                st.pyplot(fig2)

# 页脚
st.divider()
st.caption("✨ 注意：部分网站有反爬机制，可能无法爬取内容 | 基于Streamlit+BeautifulSoup开发")
