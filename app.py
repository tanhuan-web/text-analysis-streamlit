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

# -------------------------- 核心函数 --------------------------
# 1. 网页内容爬取（保留原有逻辑，仅优化返回提示）
def crawl_webpage(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
        response = requests.get(
            url, 
            headers=headers, 
            timeout=15,
            allow_redirects=True,
            verify=False
        )
        try:
            response.encoding = "utf-8"
        except:
            response.encoding = response.apparent_encoding or "gbk"
        
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "iframe", "noscript"]):
            script.decompose()
        
        # 精准提取正文
        content = ""
        content_tags = soup.find_all(
            "div", 
            class_=re.compile(r"content|article|main|text|body|detail", re.I)
        ) or soup.find_all("article") or soup.find_all("main")
        if content_tags:
            content = "\n".join([tag.get_text().strip() for tag in content_tags])
        if not content:
            p_tags = soup.find_all("p")
            content = "\n".join([p.get_text().strip() for p in p_tags])
        if not content:
            content = soup.get_text().strip()
        
        content = re.sub(r"\n+", "\n", content)
        content = re.sub(r"\s+", " ", content)
        
        if len(content) < 50:
            st.warning("⚠️ 爬取到的内容过短，可能是反爬或网页无有效文本")
            return ""
        
        return content
    
    except requests.exceptions.Timeout:
        st.error("❌ 请求超时：网页响应时间超过15秒")
        return ""
    except requests.exceptions.ConnectionError:
        st.error("❌ 连接失败：无法访问该网址（检查URL是否正确/网站是否可访问）")
        return ""
    except requests.exceptions.InvalidURL:
        st.error("❌ 无效URL：请输入完整的网址（示例：https://www.baidu.com）")
        return ""
    except Exception as e:
        st.error(f"❌ 爬取失败：{str(e)[:100]}")
        return ""

# 2. 文本预处理【重点修改：放宽过滤规则+保留日志】
def preprocess_text(text):
    """清洗文本并分词（放宽过滤规则，增加调试信息）"""
    # 调试：输出原始文本长度和前100字符
    st.sidebar.subheader("🔍 调试信息")
    st.sidebar.write(f"原始文本长度：{len(text)} 字符")
    st.sidebar.write(f"原始文本前100字符：{text[:100]}")
    
    if not text or len(text) < 10:
        st.sidebar.warning("预处理：文本过短，返回空")
        return []
    
    # 【修改1：保留中文+中文标点，不再完全移除非中文】
    # 只移除英文、数字、特殊符号，保留中文和中文标点
    text = re.sub(r"[a-zA-Z0-9`~!@#$%^&*()_+-=<>?/:;\"\'\\|{}[\]·~！@#￥%……&*（）——+-=《》？：；“”‘’、|{}【】]", "", text)
    st.sidebar.write(f"清洗后文本长度：{len(text)} 字符")
    st.sidebar.write(f"清洗后文本前100字符：{text[:100]}")
    
    if not text:
        st.sidebar.warning("预处理：清洗后无内容，返回空")
        return []
    
    # 分词
    jieba.setLogLevel(20)
    words = jieba.lcut(text)
    st.sidebar.write(f"分词结果数量：{len(words)} 个词")
    st.sidebar.write(f"分词结果前20个：{words[:20]}")
    
    # 【修改2：缩减停用词表，仅保留最核心停用词】
    stop_words = {
        "的", "了", "是", "在", "和", "有", "我", "你", "他", "都", "而", "及", "与", "之", "于", "也", "还", "这", "那"
    }
    # 【修改3：仅过滤停用词，不再过滤单字（保留短词汇）】
    words = [word for word in words if word not in stop_words]
    # 过滤空字符串
    words = [word for word in words if word.strip()]
    
    st.sidebar.write(f"过滤停用词后数量：{len(words)} 个词")
    st.sidebar.write(f"过滤后前20个：{words[:20]}")
    
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

text_source = ""
if input_mode == "手动输入文本":
    text_source = st.text_area(
        "请输入需要分析的文本（支持中文）",
        height=200,
        placeholder="例如：人工智能是未来科技的核心方向，人工智能正在改变各行各业...",
        key="text_input"
    )
else:
    url = st.text_input(
        "请输入网页URL（需包含 https://）",
        placeholder="例如：https://news.sina.com.cn/c/2025-01-01/doc-xxxx.shtml",
        key="url_input"
    )
    if st.button("📤 爬取网页内容", type="secondary", key="crawl_btn"):
        if url:
            with st.spinner("正在爬取网页内容...（请稍候，最多等待15秒）"):
                text_source = crawl_webpage(url)
                if text_source:
                    st.subheader("爬取到的内容预览")
                    preview_text = text_source[:800] + "..." if len(text_source) > 800 else text_source
                    st.text_area("内容预览", preview_text, height=200, key="preview")
        else:
            st.warning("⚠️ 请输入有效的URL！")

# 分析按钮（优化禁用逻辑）
analyze_disabled = False if text_source.strip() else True
if st.button("🚀 开始分析", type="primary", key="analyze_btn", disabled=analyze_disabled):
    # 预处理
    words = preprocess_text(text_source)
    
    # 【新增：兜底逻辑，无有效词汇时给出引导】
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
        # 词频统计（优化：即使不足20个词也能展示）
        word_count = Counter(words[:5000])
        top_n = word_count.most_common(min(20, len(word_count)))  # 取最小数量，避免空值
        
        # 分栏展示结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"1. 词频统计（TOP{len(top_n)}）")
            df = pd.DataFrame(top_n, columns=["词汇", "出现次数"])
            st.dataframe(df, use_container_width=True)
            
            # 词频柱状图（容错：至少3个词才展示图表）
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
            
            # 词云展示（容错）
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
✨ 调试辅助：侧边栏可查看文本预处理全过程，便于定位分析失败原因；
✨ 推荐测试URL：https://www.ruanyifeng.com/blog/2025/01/weekly-issue-268.html
""")
