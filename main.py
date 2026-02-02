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
# 0. 日志辅助函数
# ==========================================
def log(msg):
    """将日志打印到标准错误流 (stderr)"""
    print(msg, file=sys.stderr)

# ==========================================
# 1. 基础配置与鉴权
# ==========================================
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# [必须修改] 填入你的真实邮箱
ENTREZ_EMAIL = "dongwei_li@hotmail.com"

if not GOOGLE_API_KEY:
    raise ValueError("❌ 未找到 GOOGLE_API_KEY")

if "@" not in ENTREZ_EMAIL:
    log("❌ 错误：邮箱格式不正确！")
    sys.exit(1)

Entrez.email = ENTREZ_EMAIL

client = genai.Client(api_key=GOOGLE_API_KEY)
# 使用性价比最高的稳定版
MODEL_NAME = "gemini-2.5-flash"

# ==========================================
# 2. 搜索策略：宽召回 (Broad Recall)
# ==========================================
# 我们不再在搜索阶段做极其严格的过滤，而是先把相关的都抓回来，让 LLM 去判断
SEARCH_KEYWORDS = [
    # Layer 1: 核心技术 (只要沾边就抓)
    "single-cell", "scRNA-seq", "spatial transcriptomics", "chromatin accessibility",
    "foundation model", "transformer", "deep learning genomics",
    
    # Layer 2: 植物/作物 (用于组合查询)
    "plant", "Arabidopsis", "rice", "maize", "crop breeding"
]

# 预编译去重正则
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

Step 2: Decision
- KEEP: Highly relevant to "Plant Single-Cell" OR "AI Genomics Method".
- MAYBE: Potentially useful method transferable to plants.
- DROP: Pure clinical/cancer study with no transferable method.

Step 3: Tagging (Select all that apply)
- ATLAS (Cell atlas/Reference map)
- METHOD (New computational/experimental method)
- APPLICATION (Biological discovery)
- BREEDING (Trait prediction/Crop improvement)

Output JSON format only:
{{
  "plant_score": int,
  "single_cell_score": int,
  "ai_score": int,
  "breeding_score": int,
  "decision": "KEEP" | "MAYBE" | "DROP",
  "tags": ["TAG1", "TAG2"],
  "reason": "Short reason why"
}}
"""

# ==========================================
# 4. Prompt: 阶段二 (参谋 - 深度研读)
# ==========================================
DEEP_DIVE_PROMPT_TEMPLATE = """
# Role
你是我（植物单细胞+AI育种博士）的**科研参谋**。
这篇论文已被判定为**高价值 ({tags})**。请进行黑客式拆解。

# Metadata
Title: {title}
Tags: {tags}
Relevance Reason: {reason}

# Output Requirements (Strict Markdown)
## 📑 [中文标题]
**原标题**：{title}
**来源**：{source} | **发布时间**：{date}
**标签**：`{tags}`

### 🎯 核心摘要
[150字左右，背景-方法-结果]

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
    """解析 PubMed 摘要"""
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
    try:
        # 强制输出 JSON
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        log(f"⚠️ 评分失败: {e}")
        # 默认放行，防止漏掉，标记为 MAYBE
        return {"decision": "MAYBE", "tags": ["ERROR"], "reason": "JSON parse error"}

# ==========================================
# 7. 核心逻辑：AI 参谋 (Analyst)
# ==========================================
def generate_deep_dive(paper, evaluation):
    """对高分论文进行深度解读"""
    # 动态调整 Prompt：如果是纯 AI 方法，强调迁移性
    transfer_hint = "如果是人类研究，重点分析如何迁移到植物细胞壁/多倍体场景。"
    if "METHOD" in evaluation['tags']:
        transfer_hint += " 重点关注算法是否能处理植物数据的稀疏性。"

    prompt = DEEP_DIVE_PROMPT_TEMPLATE.format(
        title=paper['title'],
        source=paper['source'],
        date=paper['date'],
        tags=",".join(evaluation['tags']),
        reason=evaluation['reason'],
        transfer_hint=transfer_hint,
        abstract=paper['abstract']
    )
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        log(f"❌ 深度解读失败: {e}")
        return f"> ❌ 解读出错: {e}"

# ==========================================
# 8. 抓取函数 (宽搜索)
# ==========================================
def fetch_arxiv(seen_set, max_results=10): # 抓多点，让 AI 筛
    log("📡 [ArXiv] 宽范围搜索中...")
    papers = []
    # 构造更宽的查询：(Single Cell OR AI) AND (Plant OR Deep Learning)
    # 这里我们稍微放宽，只要包含核心词即可
    query = ' OR '.join([f'ti:"{k}"' for k in SEARCH_KEYWORDS[:5]]) + \
            ' OR ' + ' OR '.join([f'abs:"{k}"' for k in SEARCH_KEYWORDS[:5]])
    
    client_arxiv = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    try:
        for result in client_arxiv.results(search):
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

def fetch_biorxiv(seen_set, limit=10): # 抓多点
    log("📡 [BioRxiv] 宽范围搜索中...")
    papers = []
    try:
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=7) # 7天
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
                
                # 本地简单关键词初筛 (Layer 0)，避免送给 LLM 太多垃圾
                text_check = (title + item['abstract']).lower()
                if any(k.lower() in text_check for k in SEARCH_KEYWORDS):
                    papers.append({
                        "title": title,
                        "abstract": item['abstract'],
                        "url": f"https://doi.org/{item['doi']}",
                        "date": item['date'],
                        "source": "BioRxiv"
                    })
                    total_fetched += 1
            
            new_cursor = messages.get('next-cursor')
            if not new_cursor or str(new_cursor) == str(cursor) or total_fetched >= limit: break
            cursor = str(new_cursor)
            time.sleep(1)
    except Exception as e:
        log(f"⚠️ BioRxiv Error: {e}")
    return papers

def fetch_pubmed(seen_set, max_results=5):
    log("📡 [PubMed] 宽范围搜索中...")
    papers = []
    today_str = datetime.date.today().strftime("%Y/%m/%d")
    past_str = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y/%m/%d")
    date_term = f' AND ("{past_str}"[PDAT] : "{today_str}"[PDAT])'
    
    # 构造查询：(plant OR single cell OR AI)
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
                    "title": title,
                    "abstract": parse_pubmed_abstract(article_data),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{article['MedlineCitation']['PMID']}/",
                    "date": today_str,
                    "source": "PubMed"
                })
            except: continue
    except Exception as e:
        log(f"⚠️ PubMed Error: {e}")
    return papers

# ==========================================
# 9. 主流程 (Pipeline)
# ==========================================
def process_papers(papers):
    report_content = ""
    kept_count = 0
    
    for paper in papers:
        log(f"🤖 [裁判] 正在评审: {paper['title'][:30]}...")
        
        # Step 1: 裁判打分
        eval_result = evaluate_paper_relevance(paper)
        decision = eval_result.get('decision', 'DROP')
        tags = eval_result.get('tags', [])
        
        # 调试日志
        log(f"   -> 结果: {decision} | 标签: {tags}")
        
        # Step 2: 过滤
        if decision == "DROP":
            continue
            
        kept_count += 1
        log(f"🧠 [参谋] 正在深度研读...")
        
        # Step 3: 深度研读
        summary = generate_deep_dive(paper, eval_result)
        
        # 结果拼接
        report_content += summary
        report_content += f"\n🔗 **原文直达**: [{paper['source']} Link]({paper['url']})\n"
        report_content += f"> 🏷️ **自动标签**: `{', '.join(tags)}` | 📊 **AI评分**: Plant({eval_result.get('plant_score')}) AI({eval_result.get('ai_score')})\n"
        report_content += "---\n\n"
        
        time.sleep(2)

    return report_content, kept_count

def main():
    log(f"🚀 启动 Bio-AI 情报 Agent (v11.0 Architect)...")
    seen_papers = set()
    all_papers = []
    
    # 1. 宽范围抓取 (数量设大一点，让 AI 筛)
    all_papers.extend(fetch_arxiv(seen_papers, max_results=10))
    all_papers.extend(fetch_biorxiv(seen_papers, limit=10))
    all_papers.extend(fetch_pubmed(seen_papers, max_results=5))
    
    log(f"\n📊 宽召回阶段：共获取 {len(all_papers)} 篇候选论文，开始 AI 评审...\n")
    
    if not all_papers:
        log("未获取到任何论文。")
        return

    # 2. 智能评审与研读
    report_body, kept_count = process_papers(all_papers)

    # 3. 生成报告头
    daily_report = f"# 🧠 Bio-AI 每日情报决策 ({datetime.date.today()})\n"
    daily_report += f"> 📊 今日大盘：召回 {len(all_papers)} 篇 -> AI 严选 {kept_count} 篇\n"
    daily_report += "> 🤖 架构：Broad Recall -> Relevance Scoring -> Deep Dive\n\n"
    
    if kept_count == 0:
        daily_report += "### 今日无高价值论文入选\n"
        daily_report += "虽然抓取了候选论文，但经 AI 裁判评审，均未达到 KEEP 标准（相关性不足）。建议明天继续关注。\n"
    else:
        daily_report += report_body

    print(daily_report)
    log("\n✅ 任务完成。")

if __name__ == "__main__":
    main()
