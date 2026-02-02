import arxiv
import google.generativeai as genai
import datetime
import os
import time
import requests
import re
import sys  # [新增] 用于控制输出流
from Bio import Entrez

# ==========================================
# 0. 日志辅助函数 (核心修复)
# ==========================================
def log(msg):
    """
    将日志打印到标准错误流 (stderr)。
    这样在运行 'python main.py > report.md' 时，
    日志会显示在屏幕(控制台)上，而不会污染 report.md 文件。
    """
    print(msg, file=sys.stderr)

# ==========================================
# 1. 基础配置与鉴权
# ==========================================
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# [必须修改] 请填入你的真实邮箱
ENTREZ_EMAIL = "dongwei_li@hotmail.com" 

if not GOOGLE_API_KEY:
    raise ValueError("❌ 未找到 GOOGLE_API_KEY，请检查环境变量设置")

# 强制邮箱检查
if "your_real_email" in ENTREZ_EMAIL or "@" not in ENTREZ_EMAIL:
    # 使用 stderr 打印错误，确保能看到
    log("❌ 错误：请修改 ENTREZ_EMAIL 为真实邮箱！使用默认/假邮箱会导致 IP 被 NCBI 封禁。")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
Entrez.email = ENTREZ_EMAIL

# ==========================================
# 2. 多源检索关键词配置 & 正则预编译
# ==========================================
KEYWORDS_FOCUS = {
    "plant_sc": ["plant single cell", "Arabidopsis scRNA-seq", "rice single cell", "crop spatial transcriptomics"],
    "ai_genomics": ["deep learning genomics", "transformer DNA", "genomic foundation model", "DNA language model"],
    "methodology": ["single cell integration", "batch effect correction", "GRN inference", "trajectory inference"]
}

COMPILED_PATTERNS = []
for cat_list in KEYWORDS_FOCUS.values():
    for k in cat_list:
        COMPILED_PATTERNS.append(re.compile(rf'\b{re.escape(k.lower())}\b'))

# ==========================================
# 3. 核心提示词
# ==========================================
PAPER_PROMPT_TEMPLATE = """
# Role Assignment
你现在是我的**科研参谋（Research Strategist）**。我的背景是：**植物单细胞 + AI育种（水稻/拟南芥）**。
我不需要新闻报道式的总结，我需要**“黑客式”的思路拆解**。

# Task Description
阅读这篇论文（Title: {title}），来源：{source}。
请先进行相关性初筛，若相关，则输出一份**逻辑严密、数据详实**的技术研报。

# Constraints
1. 必须使用中文进行输出，保留必要的英文专业术语（如 Zero-shot, Chain of Thought 等）。
2. 严禁直接翻译原文摘要，必须基于理解进行重述和概括。
3. 语气保持客观、专业，避免使用营销式夸张词汇。
4. "创新点"部分必须具体，指出该论文解决了什么具体痛点，不仅是罗列功能。

# Phase 1: Relevance Check (相关性严查)
请先判断：这篇论文是否对“植物研究”、“单细胞分析”或“AI基因组学”有参考价值？
- 如果完全无关（如纯物理、纯临床药物试验），请只输出一句："❌ [不相关] 本文主要关于...，跳过。"
- 如果相关，请继续执行 Phase 2。

# Phase 2: Output Format (Strict Markdown)
请严格按照以下结构输出：

## 📑 [中文标题]
**原标题**：{title}
**来源**：{source} | **发布时间**：{date}

### 🎯 核心摘要
[在此处撰写 150-200 字的中文摘要。主要描述论文的背景问题、提出的方法论以及最终达成的效果。]

### 🧠 研究思路复盘 (The Logic Chain)
*不要只告诉我他做了什么，要告诉我他是怎么想到的。*
* **🔍 破局点 (The Spark)**：作者是看到了什么痛点，才想出了这个方法的？
* **🛠️ 技术选型逻辑**：为什么他选了 A 方法而不是 B 方法？
* **⛓️ 实验设计闭环**：他是怎么证明自己是对的？

### 💡 核心创新点与贡献
* **[创新点 1 - 技术原理]**：详细解释该创新的技术原理或实现方式，以及它相对于现有 SOTA 方法的优势。
* **[创新点 2 - 实验设计]**：描述该方法在实验设计或数据集构建上的独特之处。
* **[创新点 3 - 量化突破]**：总结该论文在实验结果上的突破（需包含具体的提升数据，如 Accuracy 提升了 x%）。

### 🙋‍♂️ 对我（植物/单细胞）的借鉴 (Actionable Insights)
* **迁移潜力**：
    * *如果是人类/动物研究*：这个思路能直接套用到**水稻/拟南芥**上吗？需要改什么？
    * *如果是AI算法*：这个模型架构适合处理**植物基因组的多倍体/高重复序列**特征吗？

### 📉 避坑指南
* 数据要求高吗？显存占用大吗？代码开源了吗？

---
# Input Data
Title: {title}
Abstract: {abstract}
"""

# ==========================================
# 4. 辅助工具函数
# ==========================================
def parse_pubmed_abstract(article_data):
    """解析 PubMed 摘要"""
    abstract_obj = article_data.get('Abstract', {}).get('AbstractText', [])
    if not abstract_obj:
        return "No Abstract"
    
    parts = []
    items = abstract_obj if isinstance(abstract_obj, list) else [abstract_obj]
    
    for item in items:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            label = item.get('Label', '')
            text = item.get('#text') or item.get('content') or " ".join([str(v) for v in item.values() if isinstance(v, str)])
            if label:
                parts.append(f"**{label}**: {text}")
            else:
                parts.append(text)
        else:
            parts.append(str(item))
            
    return " ".join(parts)

def is_duplicate(seen_set, title, source):
    """大小写不敏感去重"""
    key = (title.lower().strip(), source)
    if key in seen_set:
        return True
    seen_set.add(key)
    return False

def contains_keywords(text):
    """使用预编译正则进行全词匹配"""
    text_lower = text.lower()
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text_lower):
            return True
    return False

# ==========================================
# 5. 各平台抓取函数 (使用 log() 替代 print())
# ==========================================

def fetch_arxiv(seen_set, max_results=3):
    log("📡 [ArXiv] 正在连接...")
    papers = []
    query = ' OR '.join([f'({k})' for cat in KEYWORDS_FOCUS.values() for k in cat])
    
    client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    try:
        for result in client.results(search):
            if is_duplicate(seen_set, result.title, "ArXiv"): continue
            papers.append({
                "title": result.title,
                "abstract": result.summary,
                "url": result.entry_id,
                "date": result.published.strftime("%Y-%m-%d"),
                "source": "ArXiv"
            })
    except Exception as e:
        log(f"⚠️ ArXiv Error: {e}")
    return papers

def fetch_biorxiv(seen_set, limit=4):
    log("📡 [BioRxiv] 正在连接...")
    papers = []
    try:
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=3)
        cursor = "0"
        total_fetched = 0
        
        while True:
            url = f"https://api.biorxiv.org/details/biorxiv/{from_date}/{today}/{cursor}/json"
            resp = requests.get(url).json()
            collection = resp.get('collection', [])
            messages = resp.get('messages', [{}])[0]
            
            if not collection: break
                
            for item in collection:
                if total_fetched >= limit: break
                title = item['title']
                if is_duplicate(seen_set, title, "BioRxiv"): continue
                
                abstract = item['abstract']
                text_to_check = title + " " + abstract
                
                if contains_keywords(text_to_check):
                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "url": f"https://doi.org/{item['doi']}",
                        "date": item['date'],
                        "source": "BioRxiv"
                    })
                    total_fetched += 1
            
            new_cursor = messages.get('next-cursor')
            
            if not new_cursor or str(new_cursor) == str(cursor) or total_fetched >= limit:
                break
                
            cursor = str(new_cursor)
            time.sleep(1)

    except Exception as e:
        log(f"⚠️ BioRxiv Error: {e}")
    return papers

def fetch_pubmed(seen_set, max_results=3):
    log("📡 [PubMed] 正在连接...")
    papers = []
    today_str = datetime.date.today().strftime("%Y/%m/%d")
    past_str = (datetime.date.today() - datetime.timedelta(days=3)).strftime("%Y/%m/%d")
    date_term = f' AND ("{past_str}"[PDAT] : "{today_str}"[PDAT])'
    
    term = ' OR '.join([f'({k})' for cat in KEYWORDS_FOCUS.values() for k in cat]) + date_term

    try:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results, sort="date")
        record = Entrez.read(handle)
        id_list = record["IdList"]
        handle.close()

        if not id_list: return []
        time.sleep(3)

        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        for article in records['PubmedArticle']:
            try:
                article_data = article['MedlineCitation']['Article']
                title = article_data['ArticleTitle']
                if is_duplicate(seen_set, title, "PubMed"): continue

                abstract = parse_pubmed_abstract(article_data)
                pmid = article['MedlineCitation']['PMID']
                
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "date": today_str, 
                    "source": "PubMed"
                })
            except Exception as e:
                log(f"⚠️ Skip PubMed item: {e}")
                continue
    except Exception as e:
        log(f"⚠️ PubMed Error: {e}")
    return papers

# ==========================================
# 6. 主程序
# ==========================================

def process_papers(papers):
    report_content = ""
    for paper in papers:
        # 使用 log() 打印进度，不污染最终报告
        log(f"🤖 正在研读 ({paper['source']}): {paper['title'][:40]}...")
        
        prompt = PAPER_PROMPT_TEMPLATE.format(
            title=paper['title'],
            source=paper['source'],
            date=paper['date'],
            abstract=paper['abstract']
        )
        
        try:
            response = model.generate_content(prompt)
            summary = response.text
            
            if "❌" in summary and "不相关" in summary:
                log(f"   -> ⏭️ 跳过：内容不相关")
                continue
                
            report_content += summary
            report_content += f"\n🔗 **原文直达**: [{paper['source']} Link]({paper['url']})\n"
            report_content += "---\n\n"
            time.sleep(4)
            
        except Exception as e:
            log(f"   -> ❌ 分析失败: {e}")
    return report_content

def main():
    log("🚀 启动 Bio-AI 全网情报抓取 (v6.0 Final)...")
    seen_papers = set()
    all_papers = []
    
    all_papers.extend(fetch_arxiv(seen_papers, max_results=3))
    all_papers.extend(fetch_biorxiv(seen_papers, limit=4))
    all_papers.extend(fetch_pubmed(seen_papers, max_results=3))
    
    log(f"\n📊 共筛选出 {len(all_papers)} 篇高相关论文，开始 AI 深度研读...\n")
    
    if not all_papers:
        log("今日无符合条件的最新文献更新。")
        # 即使没有论文，也打印一个空的提示，或者什么都不打印
        return

    daily_report = f"# 🧠 Bio-AI 每日思路研报 ({datetime.date.today()})\n"
    daily_report += "> 来源：ArXiv (AI/Method) | BioRxiv (Preprint) | PubMed (Published)\n\n"
    daily_report += process_papers(all_papers)

    # ==========================================
    # 唯一的一个 print (输出到 stdout)
    # ==========================================
    print(daily_report)

    log("\n✅ 任务完成，报告已生成。")

if __name__ == "__main__":
    main()
