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
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="文本分析工具（支持URL爬取）",
    page_icon="📝",
    layout="wide"
)

# -------------------------- 全局变量（解决作用域问题） --------------------------
# 用session_state存储爬取的文本，避免变量丢失
if "crawled_text" not in st.session_state:
    st.session_state.crawled_text = ""

# -------------------------- 核心函数 --------------------------
# 1. 网页内容爬取（适配HTTP协议+强制文本存储）
def crawl_webpage(url):
    """爬取指定URL（兼容HTTP/HTTPS，强制存储到session_state）"""
    try:
        # 补全URL协议（若用户只输入域名）
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
            st.warning(f"自动补全协议：{url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Upgrade-Insecure-Requests": "1"  # 适配HTTP跳转HTTPS
        }
        # 增加重试机制
        for retry in range(2):
            try:
                response = requests.get(
                    url, 
                    headers=headers, 
                    timeout=20,
                    allow_redirects=True,
                    verify=False,
                    stream=False
                )
                break
            except:
                if retry == 1:
                    raise
        
        # 强制编码适配（解决中文乱码）
        encodings = ["utf-8", "gbk", "gb2312", "gb18030", response.apparent_encoding]
        content = ""
        for encoding in encodings:
            try:
                response.encoding = encoding
                content = response.text
                if content:
                    break
            except:
                continue
        
        soup = BeautifulSoup(content, "html.parser")
        for script in soup(["script", "style", "iframe", "noscript"]):
            script.decompose()
        
        # 提取正文（适配目标网站的div/p结构）
        content = ""
        # 优先提取所有可见文本（适配目标网站的无规则文本）
        all_text = soup.get_text(separator=" ", strip=True)
        # 过滤连续空白符
        content = re.sub(r"\s+", " ", all_text)
        
        # 强制存储到session_state
        st.session_state.crawled_text = content
        
        # 预览展示
        if len(content) < 50:
            st.warning("⚠️ 爬取到的内容过短，可能是反爬或网页无有效文本")
        else:
            st.subheader("爬取到的内容预览")
            preview_text = content[:800] + "..." if len(content) > 800 else content
            st.text_area("内容预览", preview_text, height=200, key="preview")
        
        return content
    
    except requests.exceptions.Timeout:
        st.error("❌ 请求超时：网页响应时间超过20秒")
        st.session_state.crawled_text = ""
        return ""
    except requests.exceptions.ConnectionError:
        st.error("❌ 连接失败：无法访问该网址（检查URL是否正确/网站是否可访问）")
        st.session_state.crawled_text = ""
        return ""
    except requests.exceptions.InvalidURL:
        st.error("❌ 无效URL：请输入完整的网址（示例：https://www.baidu.com）")
        st.session_state.crawled_text = ""
        return ""
    except Exception as e:
        st.error(f"❌ 爬取失败：{str(e)[:100]}")
        st.session_state.crawled_text = ""
        return ""

# 2. 文本预处理（适配目标网站的半结构化文本）
def preprocess_text(text):
    """清洗文本并分词（适配含数字/重复内容的半结构化文本）"""
    # 调试：输出session_state中的文本
    st.sidebar.subheader("🔍 调试信息")
    st.sidebar.write(f"Session文本长度：{len(text)} 字符")
    st.sidebar.write(f"Session文本前100字符：{text[:100]}")
    
    if not text or len(text) < 10:
        st.sidebar.warning("预处理：文本过短，返回空")
        return []
    
    # 【关键修改：仅过滤纯数字/日期，保留中文词汇】
    # 步骤1：移除纯数字串（如2025-07、14 2012-12等）
    text = re.sub(r"\d+[-/]\d+[-/]\d+|\d+", "", text)
    # 步骤2：仅保留中文（移除所有非中文字符）
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    # 步骤3：移除连续重复的短文本（适配目标网站的重复内容）
    text = re.sub(r"(.{2,5})\1{3,}", r"\1", text)  # 移除重复3次以上的2-5字短语
    
    st.sidebar.write(f"清洗后文本长度：{len(text)} 字符")
    st.sidebar.write(f"清洗后文本前100字符：{text[:100]}")
    
    if not text:
        st.sidebar.warning("预处理：清洗后无内容，返回空")
        return []
    
    # 分词（适配重复词汇）
    jieba.setLogLevel(20)
    words = jieba.lcut(text)
    
    # 极简停用词表（仅过滤最核心）
    stop_words = {"的", "了", "是", "在", "和", "有", "都", "而", "及", "与", "之", "于", "也", "还", "这", "那"}
    words = [word for word in words if word not in stop_words and len(word) >= 2]
    words = [word for word in words if word.strip()]
    
    st.sidebar.write(f"分词后数量：{len(words)} 个词")
    st.sidebar.write(f"分词后前20个：{words[:20]}")
    
    return words

# 3. 词云生成（兼容云端）
def generate_wordcloud(words):
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            font_path=None,
            max_words=100,
            colormap="viridis",
            random_state=42
        ).generate(" ".join(words))
        return wordcloud
    except Exception as e:
        st.warning(f"⚠️ 词云生成异常：{str(e)[:50]}，使用降级方案")
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=100,
            random_state=42
        ).generate(" ".join(words))
        return wordcloud

# -------------------------- 页面交互 --------------------------
st.title("📝 文本分析工具（支持URL爬取）")
st.markdown("### 支持：网页内容爬取、分词、词频统计、关键词提取、词云生成")

# 选择输入方式
input_mode = st.radio(
    "请选择输入方式", 
    ["手动输入文本", "输入URL爬取网页内容"], 
    horizontal=True,
    key="input_mode"
)

# 手动输入文本逻辑
if input_mode == "手动输入文本":
    st.session_state.crawled_text = st.text_area(
        "请输入需要分析的文本（支持中文）",
        height=200,
        placeholder="例如：人工智能是未来科技的核心方向，人工智能正在改变各行各业...",
        key="text_input"
    )
# URL爬取逻辑
else:
    url = st.text_input(
        "请输入网页URL（支持HTTP/HTTPS）",
        placeholder="例如：http://zpy.cstam.org.cn/",
        key="url_input"
    )
    if st.button("📤 爬取网页内容", type="secondary", key="crawl_btn"):
        if url:
            with st.spinner("正在爬取网页内容...（最多等待20秒）"):
                crawl_webpage(url)
        else:
            st.warning("⚠️ 请输入有效的URL！")

# 分析按钮逻辑（读取session_state中的文本）
analyze_disabled = False if st.session_state.crawled_text.strip() else True
if st.button("🚀 开始分析", type="primary", key="analyze_btn", disabled=analyze_disabled):
    # 从session_state读取文本
    text_source = st.session_state.crawled_text
    words = preprocess_text(text_source)
    
    if not words:
        st.warning("""
        ⚠️ 未提取到可分析的中文词汇！可能原因：
        1. 爬取的文本以英文/数字/特殊符号为主；
        2. 文本中仅包含停用词（如“的、了、是”等）；
        3. 网页内容为图片/视频，无文字信息。
        
        建议：
        - 更换爬取目标（优先选择新闻、博客等纯文本网页）；
        - 手动补充中文文本后再分析。
        """)
    else:
        word_count = Counter(words[:5000])
        top_n = word_count.most_common(min(20, len(word_count)))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"1. 词频统计（TOP{len(top_n)}）")
            df = pd.DataFrame(top_n, columns=["词汇", "出现次数"])
            st.dataframe(df, use_container_width=True)
            
            st.subheader("2. 词频可视化")
            if len(top_n) >= 3:
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_10 = top_n[:min(10, len(top_n))]
                    ax.bar([w[0] for w in top_10], [w[1] for w in top_10], color="#1f77b4")
                    ax.set_xlabel("词汇", fontsize=12)
                    ax.set_ylabel("出现次数", fontsize=12)
                    ax.set_title(f"TOP{len(top_10)}词汇词频分布", fontsize=14)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"⚠️ 图表生成失败：{str(e)[:50]}")
            else:
                st.info("⚠️ 有效词汇不足3个，无法生成柱状图")
        
        with col2:
            st.subheader("3. 核心关键词")
            top_keywords = min(8, len(top_n))
            if top_keywords > 0:
                keywords = [w[0] for w in top_n[:top_keywords]]
                st.markdown(f"**{', '.join(keywords)}**")
            else:
                st.info("⚠️ 无有效关键词")
            
            st.subheader("4. 词云展示")
            try:
                wordcloud = generate_wordcloud(words)
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.imshow(wordcloud, interpolation="bilinear")
                ax2.axis("off")
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"⚠️ 词云生成失败：{str(e)[:50]}")

# 页脚
st.divider()
st.caption("""
✨ 适配说明：针对http://zpy.cstam.org.cn/这类半结构化网页做了特殊适配；
✨ 核心优化：解决HTTP协议爬取、文本传递丢失、重复内容过滤问题；
""")
