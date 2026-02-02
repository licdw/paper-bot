import arxiv
import os
import time
import requests
import re
import sys
import datetime
import json
from Bio import Entrez
from google import genai
from google.genai import types

# ==========================================
# 0. 日志与重试机制 (解决 429 报错)
# ==========================================
def log(msg):
    """将日志打印到标准错误流 (stderr)"""
    print(msg, file=sys.stderr)

def safe_generate_content(client, model, contents, config=None, retries=3):
    """
    带重试机制的 API 调用，专门解决 429 Resource Exhausted
    """
    for attempt in range(retries):
        try:
            if config:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            else:
                response = client.models.generate_content(
                    model=model, contents=contents
                )
            return response
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 30 * (attempt + 1) # 第一次等30秒，第二次等60秒
                log(f"⚠️ 触发限流 (429)，休息 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                # 如果是其他错误，直接抛出
                log(f"❌ API 调用错误: {e}")
                return None
    return None

# ==========================================
# 1. 基础配置与鉴权
# ==========================================
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
ENTREZ_EMAIL = "dongwei_li@hotmail.com" # [已保留]

if not GOOGLE_API_KEY:
    raise ValueError("❌ 未找到 GOOGLE_API_KEY")

if "@" not in ENTREZ_EMAIL:
    log("❌ 错误：邮箱格式不正确！")
    sys.exit(1)

Entrez.email = ENTREZ_EMAIL

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# ==========================================
# 2. 搜索策略
# ==========================================
SEARCH_KEYWORDS = [
    "single-cell", "scRNA-seq", "spatial transcriptomics", "chromatin accessibility",
    "foundation model", "transformer", "deep learning genomics",
    "plant", "Arabidopsis", "rice", "maize", "crop breeding"
]
COMPILED_PATTERNS = [re.compile(rf'\b{re.escape(k.lower())}\b') for k in SEARCH_KEYWORDS]

# ==========================================
# 3. Prompt: 阶段一 (裁判 - 评分与分类)
# ==========================================
RELEVANCE_PROMPT_TEMPLATE = """
You are a domain expert in **Plant single-cell biology** and **AI-driven crop breeding**.
Your task is to JUDGE the relevance of this paper.

Title: {title}
Abstract: {abstract}

Step 1: Relevance Scoring (0-3)
- Plant relevance (0: None, 3: Core plant study)
- Single-cell/Omics relevance (0: None, 3: Core single-cell/spatial)
- AI/Modeling relevance (0: None, 3: Deep learning/Foundation model)
- Breeding relevance (0: None, 3: Trait prediction/Improvement)

Step 2: Extract Species
- Extract the main organism/species studied (e.g., "Rice (Oryza sativa)", "Arabidopsis", "Human", "General Model").

Step 3: Decision
- KEEP: Highly relevant.
- DROP: Totally irrelevant.

Step 4: Tagging
- ATLAS, METHOD, APPLICATION, BREEDING

Output JSON format only:
{{
  "plant_score": int,
  "single_cell_score": int,
  "ai_score": int,
  "breeding_score": int,
  "species": "String",
  "decision": "KEEP" | "DROP",
  "tags": ["TAG1", "TAG2"],
  "reason": "Short reason"
}}
"""

# ==========================================
# 4. Prompt: 阶段二 (参谋 - 深度研读)
# ==========================================
DEEP_DIVE_PROMPT_TEMPLATE = """
# Role
你是我（植物单细胞+AI育种博士）的**科研参谋**。
这篇论文已被判定为**高价值**。请进行黑客式拆解。

# Metadata
Title: {title}
Species: {species}
Tags: {tags}

# Output Requirements (Strict Markdown)
## 📑 [中文标题]
**原标题**：{title}
**来源**：{source} | **发布时间**：{date}
**研究物种**：`{species}` | **标签**：`{tags}`

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
# Input Abstract
{abstract}
"""

# ==========================================
# 5. 工具函数
# ==========================================
def parse_pubmed_abstract(article_data):
    abstract_obj = article_data.get('Abstract', {}).get('AbstractText', [])
    if not abstract_obj: return "No Abstract"
    parts = []
    items = abstract_obj if isinstance(abstract_obj, list) else [abstract_obj]
    for item in items:
        if isinstance(item, str): parts.append(item)
        elif isinstance(item, dict):
            text = item.get('#text') or item.get('content') or ""
            label = item.get('Label', '')
            parts.append(f"**{label}**: {text}" if label else text)
    return " ".join(parts)

def is_duplicate(seen_set, title, source):
    key = (title.lower().strip(), source)
    if key in seen_set: return True
    seen_set.add(key)
    return False

# ==========================================
# 6. 核心逻辑：AI 裁判 (Judge)
# ==========================================
def evaluate_paper_relevance(paper):
    """调用 Gemini 判断论文是否值得读，返回 JSON"""
    prompt = RELEVANCE_PROMPT_TEMPLATE.format(
        title=paper['title'],
        abstract=paper['abstract']
    )
    # 使用带重试的安全调用
    response = safe_generate_content(
        client, 
        MODEL_NAME, 
        prompt, 
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    if response and response.text:
        try:
            return json.loads(response.text)
        except:
            return {"decision": "KEEP", "tags": ["PARSE_ERROR"], "species": "Unknown", "reason": "JSON Error"}
    return {"decision": "DROP", "tags": [], "reason": "API Error"}

# ==========================================
# 7. 核心逻辑：AI 参谋 (Analyst)
# ==========================================
def generate_deep_dive(paper, evaluation):
    """对高分论文进行深度解读"""
    transfer_hint = "如果是人类研究，重点分析如何迁移到植物细胞壁/多倍体场景。"
    if "METHOD" in evaluation['tags']:
        transfer_hint += " 重点关注算法是否能处理植物数据的稀疏性。"

    prompt = DEEP_DIVE_PROMPT_TEMPLATE.format(
        title=paper['title'],
        source=paper['source'],
        date=paper['date'],
        tags=",".join(evaluation['tags']),
        species=evaluation.get('species', 'N/A'),
        transfer_hint=transfer_hint,
        abstract=paper['abstract']
    )
    
    # 使用带重试的安全调用
    response = safe_generate_content(client, MODEL_NAME, prompt)
    
    if response and response.text:
        return response.text
    return f"> ❌ 解读失败：API多次重试后无响应。"

# ==========================================
# 8. 抓取函数
# ==========================================
def fetch_arxiv(seen_set, max_results=10):
    log("📡 [ArXiv] 宽范围搜索中...")
    papers = []
    query = ' OR '.join([f'ti:"{k}"' for k in SEARCH_KEYWORDS[:5]]) + \
            ' OR ' + ' OR '.join([f'abs:"{k}"' for k in SEARCH_KEYWORDS[:5]])
    client_arxiv = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    try:
        for result in client_arxiv.results(search):
            if is_duplicate(seen_set, result.title, "ArXiv"): continue
            papers.append({
                "title": result.title, "abstract": result.summary,
                "url": result.entry_id, "date": result.published.strftime("%Y-%m-%d"),
                "source": "ArXiv"
            })
    except Exception as e: log(f"⚠️ ArXiv Error: {e}")
    return papers

def fetch_biorxiv(seen_set, limit=10):
    log("📡 [BioRxiv] 宽范围搜索中...")
    papers = []
    try:
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=7)
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
                text_check = (title + item['abstract']).lower()
                if any(k.lower() in text_check for k in SEARCH_KEYWORDS):
                    papers.append({
                        "title": title, "abstract": item['abstract'],
                        "url": f"https://doi.org/{item['doi']}", "date": item['date'],
                        "source": "BioRxiv"
                    })
                    total_fetched += 1
            new_cursor = messages.get('next-cursor')
            if not new_cursor or str(new_cursor) == str(cursor) or total_fetched >= limit: break
            cursor = str(new_cursor)
            time.sleep(1)
    except Exception as e: log(f"⚠️ BioRxiv Error: {e}")
    return papers

def fetch_pubmed(seen_set, max_results=5):
    log("📡 [PubMed] 宽范围搜索中...")
    papers = []
    today_str = datetime.date.today().strftime("%Y/%m/%d")
    past_str = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y/%m/%d")
    date_term = f' AND ("{past_str}"[PDAT] : "{today_str}"[PDAT])'
    term = ' OR '.join([f'({k})' for k in SEARCH_KEYWORDS]) + date_term
    try:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results, sort="date")
        record = Entrez.read(handle)
        id_list = record["IdList"]
        handle.close()
        if not id_list: return []
        time.sleep(2)
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        for article in records['PubmedArticle']:
            try:
                article_data = article['MedlineCitation']['Article']
                title = article_data['ArticleTitle']
                if is_duplicate(seen_set, title, "PubMed"): continue
                papers.append({
                    "title": title, "abstract": parse_pubmed_abstract(article_data),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{article['MedlineCitation']['PMID']}/",
                    "date": today_str, "source": "PubMed"
                })
            except: continue
    except Exception as e: log(f"⚠️ PubMed Error: {e}")
    return papers

# ==========================================
# 9. 主流程 (逻辑更新：先评分，再排序，后研读)
# ==========================================
def process_papers(papers):
    
    # 1. 评分阶段 (Phase 1: Judging)
    log(f"⚖️ 开始第一轮筛选 (共 {len(papers)} 篇)...")
    kept_papers = []
    
    for paper in papers:
        eval_result = evaluate_paper_relevance(paper)
        
        # 调试输出
        decision = eval_result.get('decision', 'DROP')
        species = eval_result.get('species', 'N/A')
        log(f"   -> {paper['title'][:20]}... | {decision} | {species}")
        
        if decision == "KEEP":
            paper['eval'] = eval_result # 把评分结果存进去
            kept_papers.append(paper)
        
        # 即使是评分，也加一点延迟防止 429
        time.sleep(2)

    if not kept_papers:
        return "", 0

    # 2. 排序阶段 (Phase 2: Sorting)
    # 排序逻辑：
    # Group 1: Plant Score >= 2 (植物相关，放最前)
    # Group 2: AI Score >= 2 (方法相关，放中间)
    # Group 3: Others (其他迁移，放最后)
    log("🔄 正在智能排序...")
    
    def sort_key(p):
        plant_score = p['eval'].get('plant_score', 0)
        ai_score = p['eval'].get('ai_score', 0)
        
        # 返回一个元组，Python会按顺序比较
        # 负号是因为要降序排列 (分数高的在前)
        if plant_score >= 2:
            return (0, -plant_score, -ai_score) # 优先级 0 (最高)
        elif ai_score >= 2:
            return (1, -ai_score, -plant_score) # 优先级 1
        else:
            return (2, -ai_score, -plant_score) # 优先级 2
            
    kept_papers.sort(key=sort_key)

    # 3. 研读阶段 (Phase 3: Deep Dive)
    log(f"🧠 开始深度研读 (入选 {len(kept_papers)} 篇)...")
    report_content = ""
    
    for paper in kept_papers:
        summary = generate_deep_dive(paper, paper['eval'])
        
        report_content += summary
        report_content += f"\n🔗 **原文直达**: [{paper['source']} Link]({paper['url']})\n"
        
        # 添加底部状态栏
        tags = paper['eval'].get('tags', [])
        plant_score = paper['eval'].get('plant_score', 0)
        ai_score = paper['eval'].get('ai_score', 0)
        report_content += f"> 🏷️ **自动标签**: `{', '.join(tags)}` | 📊 **评分**: Plant({plant_score}) AI({ai_score})\n"
        report_content += "---\n\n"
        
        # 研读后必须sleep，防止 Deep Dive 触发限流
        time.sleep(5) 

    return report_content, len(kept_papers)

def main():
    log(f"🚀 启动 Bio-AI 情报 Agent (v12.0 Sorted & Retry)...")
    seen_papers = set()
    all_papers = []
    
    all_papers.extend(fetch_arxiv(seen_papers, max_results=10))
    all_papers.extend(fetch_biorxiv(seen_papers, limit=10))
    all_papers.extend(fetch_pubmed(seen_papers, max_results=5))
    
    log(f"\n📊 宽召回阶段：共获取 {len(all_papers)} 篇候选论文...\n")
    
    if not all_papers:
        log("未获取到任何论文。")
        return

    # 处理流程 (包含评分、排序、研读)
    report_body, kept_count = process_papers(all_papers)

    # 生成报告头
    daily_report = f"# 🧠 Bio-AI 每日情报决策 ({datetime.date.today()})\n"
    daily_report += f"> 📊 今日大盘：召回 {len(all_papers)} 篇 -> AI 严选 {kept_count} 篇\n"
    daily_report += "> 🤖 排序策略：植物研究 > 核心算法 > 迁移借鉴\n\n"
    
    if kept_count == 0:
        daily_report += "### 今日无高价值论文入选\n建议明天继续关注。\n"
    else:
        daily_report += report_body

    print(daily_report)
    log("\n✅ 任务完成。")

if __name__ == "__main__":
    main()
