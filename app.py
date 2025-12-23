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
# 设置中文字体（兼容云端）
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 页面配置
st.set_page_config(
    page_title="文本分析工具（支持URL爬取）",
    page_icon="📝",
    layout="wide"
)

# -------------------------- 核心函数 --------------------------
# 1. 网页内容爬取+增强清洗
def crawl_webpage(url):
    """爬取指定URL的网页文本内容（增强版）"""
    try:
        # 强化请求头
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
        # 发送请求（允许重定向）
        response = requests.get(
            url, 
            headers=headers, 
            timeout=15,
            allow_redirects=True,
            verify=False  # 忽略SSL证书错误
        )
        # 强制指定编码（优先utf-8，兜底gbk）
        try:
            response.encoding = "utf-8"
        except:
            response.encoding = response.apparent_encoding or "gbk"
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 移除无关标签（脚本、样式、广告）
        for script in soup(["script", "style", "iframe", "noscript"]):
            script.decompose()
        
        # 精准提取正文（多规则匹配）
        content = ""
        # 规则1：匹配常见正文class/id
        content_tags = soup.find_all(
            "div", 
            class_=re.compile(r"content|article|main|text|body|detail", re.I)
        ) or soup.find_all("article") or soup.find_all("main")
        if content_tags:
            content = "\n".join([tag.get_text().strip() for tag in content_tags])
        # 规则2：若未匹配，取所有p标签
        if not content:
            p_tags = soup.find_all("p")
            content = "\n".join([p.get_text().strip() for p in p_tags])
        # 规则3：终极兜底（全文本）
        if not content:
            content = soup.get_text().strip()
        
        # 清洗爬取的文本（去重空行、多余空格）
        content = re.sub(r"\n+", "\n", content)
        content = re.sub(r"\s+", " ", content)
        
        if len(content) < 50:  # 过滤无效内容
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
        st.error(f"❌ 爬取失败：{str(e)[:100]}")  # 截断过长错误信息
        return ""

# 2. 文本预处理+容错
def preprocess_text(text):
    """清洗文本并分词（增强容错）"""
    if not text or len(text) < 10:
        return []
    
    # 保留中文，移除所有非中文字符
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    if not text:
        return []
    
    # 分词（关闭jieba日志）
    jieba.setLogLevel(20)
    words = jieba.lcut(text)
    
    # 扩展停用词表+过滤
    stop_words = {
        "的", "了", "是", "在", "和", "有", "我", "你", "他", "她", "它", 
        "就", "都", "而", "及", "与", "之", "于", "也", "还", "这", "那",
        "为", "以", "可", "将", "对", "能", "会", "要", "把", "被", "所",
        "该", "从", "到", "因", "由", "随", "如", "若", "则", "或", "即",
        "着", "过", "呢", "吗", "吧", "啊", "哦", "嗯", "哈", "哎", "哟",
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万"
    }
    # 过滤停用词、单字、过短词汇
    words = [word for word in words if word not in stop_words and len(word) >= 2]
    
    # 去重后再统计（避免重复词汇干扰）
    words = list(filter(None, words))
    return words

# 3. 词云生成容错函数
def generate_wordcloud(words):
    """生成词云（兼容云端无本地字体）"""
    try:
        # 优先使用系统内置中文字体，避免本地字体依赖
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            font_path=None,  # 云端自动匹配字体
            max_words=100,
            colormap="viridis",
            random_state=42  # 固定随机种子，结果稳定
        ).generate(" ".join(words))
        return wordcloud
    except Exception as e:
        # 降级方案：使用默认字体+提示
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
    key="input_mode"  # 唯一key，避免状态异常
)

text_source = ""
if input_mode == "手动输入文本":
    # 手动输入文本
    text_source = st.text_area(
        "请输入需要分析的文本（支持中文）",
        height=200,
        placeholder="例如：人工智能是未来科技的核心方向，人工智能正在改变各行各业...",
        key="text_input"
    )
else:
    # URL输入
    url = st.text_input(
        "请输入网页URL（需包含 https://）",
        placeholder="例如：https://news.sina.com.cn/c/2025-01-01/doc-xxxx.shtml",
        key="url_input"
    )
    # 爬取按钮
    if st.button("📤 爬取网页内容", type="secondary", key="crawl_btn"):
        if url:
            with st.spinner("正在爬取网页内容...（请稍候，最多等待15秒）"):
                text_source = crawl_webpage(url)
                # 展示爬取结果（供确认）
                if text_source:
                    st.subheader("爬取到的内容预览")
                    preview_text = text_source[:800] + "..." if len(text_source) > 800 else text_source
                    st.text_area("内容预览", preview_text, height=200, key="preview")
        else:
            st.warning("⚠️ 请输入有效的URL！")

# 分析按钮（增加防重复点击）
if st.button("🚀 开始分析", type="primary", key="analyze_btn", disabled=not text_source.strip()):
    # 预处理
    words = preprocess_text(text_source)
    if not words:
        st.warning("⚠️ 无有效分析内容！可能原因：\n1. 文本仅含无意义字符\n2. 爬取内容被反爬过滤\n3. 文本无中文词汇")
    else:
        # 词频统计（限制最大数量，避免卡顿）
        word_count = Counter(words[:5000])  # 仅取前5000个词，防止内存溢出
        top_20 = word_count.most_common(20)
        
        # 分栏展示结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. 词频统计（TOP20）")
            df = pd.DataFrame(top_20, columns=["词汇", "出现次数"])
            st.dataframe(df, use_container_width=True)
            
            # 词频柱状图（容错）
            st.subheader("2. 词频可视化（TOP10）")
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(df["词汇"][:10], df["出现次数"][:10], color="#1f77b4")
                ax.set_xlabel("词汇", fontsize=12)
                ax.set_ylabel("出现次数", fontsize=12)
                ax.set_title("TOP10词汇词频分布", fontsize=14)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"⚠️ 图表生成失败：{str(e)[:50]}")
        
        with col2:
            st.subheader("3. 核心关键词（TOP8）")
            keywords = [w[0] for w in top_20[:8]] if top_20 else []
            if keywords:
                st.markdown(f"**{', '.join(keywords)}**")
            else:
                st.write("暂无有效关键词")
            
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
✨ 注意事项：
1. 部分网站（如知乎、淘宝）有严格反爬机制，可能无法爬取；
2. 建议爬取新闻、博客、百科等静态网页；
3. 若分析失败，请检查文本是否包含足够的中文内容。
""")
