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
# 0. 日志与配置
# ==========================================
def log(msg):
    """将日志打印到标准错误流 (stderr)"""
    print(msg, file=sys.stderr)

# 获取 Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY') # [新增] 从环境变量获取
ENTREZ_EMAIL = "dongwei_li@hotmail.com"

# 检查配置
if not GOOGLE_API_KEY:
    raise ValueError("❌ 未找到 GOOGLE_API_KEY")
if not ZHIPU_API_KEY:
    log("⚠️ 未找到 ZHIPU_API_KEY，将仅使用 Gemini 模式")

if "@" not in ENTREZ_EMAIL:
    log("❌ 错误：邮箱格式不正确！")
    sys.exit(1)

Entrez.email = ENTREZ_EMAIL

# 初始化 Gemini 客户端
client_gemini = genai.Client(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

# ==========================================
# 1. 双模型底层封装 (核心升级)
# ==========================================

def call_gemini(prompt, is_json=False):
    """调用 Google Gemini"""
    try:
        config = types.GenerateContentConfig(response_mime_type="application/json") if is_json else None
        response = client_gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        # 抛出异常让上层捕获，以便切换模型
        raise e

def call_zhipu(prompt, is_json=False):
    """调用智谱 GLM-4 (使用你提供的 requests 方式)"""
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    
    # 构造 GLM 的 prompt 格式
    messages = [
        {"role": "system", "content": "你是一个专业的生物信息学科研助手。请直接输出结果，不要废话。"},
        {"role": "user", "content": prompt}
    ]
    
    # 如果需要 JSON，我们在 System Prompt 里强调一下（GLM-4-Flash 对 JSON mode 支持视版本而定，这里通过 Prompt 约束）
    if is_json:
        messages[0]["content"] += " 请务必输出严格的 JSON 格式，不要包含 Markdown 代码块标记。"

    payload = {
        "model": "glm-4-flash", # 使用性价比高的 Flash 版本
        "messages": messages,
        "stream": False,
        "temperature": 0.5, # 科研任务稍微降低创造性
        "thinking": { "type": "disabled" } # 暂时关掉 thinking 以免干扰 JSON 解析，除非你需要思维链
    }
    
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60) # 设置60秒超时
        resp_json = resp.json()
        
        # 解析响应
        if "choices" in resp_json:
            content = resp_json['choices'][0]['message']['content']
            #以此清理可能的 markdown 标记
            content = content.replace("```json", "").replace("```", "").strip()
            return content
        elif "error" in resp_json:
            raise Exception(f"Zhipu API Error: {resp_json['error']}")
        else:
            raise Exception(f"Unknown Zhipu Response: {resp.text}")
            
    except Exception as e:
        raise e

def hybrid_generate_content(prompt, is_json=False):
    """
    [智能混合调用]
    策略：优先 Gemini -> 失败/限流 -> 切换 GLM-4 -> 再失败 -> 休息重试
    """
    # 1. 尝试 Gemini (主力)
    try:
        # log("   ⚡ [Gemini] 正在思考...")
        return call_gemini(prompt, is_json)
    except Exception as e:
        error_str = str(e)
        
        # 2. 如果 Gemini 挂了 (429 限流 或 500 错误)，且配置了智谱 Key
        if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str) and ZHIPU_API_KEY:
            log(f"   ⚠️ Gemini 限流/报错，自动切换至 [智谱GLM-4] 接力...")
            try:
                return call_zhipu(prompt, is_json)
            except Exception as e_zhipu:
                log(f"   ❌ 智谱也挂了: {e_zhipu}")
                # 两个都挂了，只能休息等待了
                time.sleep(30)
                return None
        
        # 如果没配置智谱 Key，只能硬等
        elif "429" in error_str:
             log(f"   ⚠️ Gemini 限流，无备用模型，等待 30秒...")
             time.sleep(30)
             return None
        else:
            log(f"   ❌ API 未知错误: {e}")
            return None

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
    """调用混合模型判断论文"""
    prompt = RELEVANCE_PROMPT_TEMPLATE.format(
        title=paper['title'],
        abstract=paper['abstract']
    )
    
    # 使用混合调用接口
    response_text = hybrid_generate_content(prompt, is_json=True)
    
    if response_text:
        try:
            return json.loads(response_text)
        except:
            # 简单的 JSON 修复尝试
            try:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                return json.loads(response_text[start:end])
            except:
                return {"decision": "KEEP", "tags": ["PARSE_ERROR"], "species": "Unknown", "reason": "JSON Error"}
    return {"decision": "DROP", "tags": [], "reason": "API Error"}

# ==========================================
# 7. 核心逻辑：AI 参谋 (Analyst)
# ==========================================
def generate_deep_dive(paper, evaluation):
    """深度解读"""
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
    
    # 使用混合调用接口
    response_text = hybrid_generate_content(prompt, is_json=False)
    
    if response_text:
        return response_text
    return f"> ❌ 解读失败：所有模型均无响应。"

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
# 9. 主流程
# ==========================================
def process_papers(papers):
    
    # 1. 评分阶段
    log(f"⚖️ 开始第一轮筛选 (共 {len(papers)} 篇)...")
    kept_papers = []
    
    for paper in papers:
        eval_result = evaluate_paper_relevance(paper)
        
        decision = eval_result.get('decision', 'DROP')
        species = eval_result.get('species', 'N/A')
        log(f"   -> {paper['title'][:20]}... | {decision} | {species}")
        
        if decision == "KEEP":
            paper['eval'] = eval_result
            kept_papers.append(paper)
        
        time.sleep(1) # 稍微减少等待，因为有双模型切换保障

    if not kept_papers:
        return "", 0

    # 2. 排序阶段
    log("🔄 正在智能排序...")
    def sort_key(p):
        plant_score = p['eval'].get('plant_score', 0)
        ai_score = p['eval'].get('ai_score', 0)
        if plant_score >= 2: return (0, -plant_score, -ai_score)
        elif ai_score >= 2: return (1, -ai_score, -plant_score)
        else: return (2, -ai_score, -plant_score)
            
    kept_papers.sort(key=sort_key)

    # 3. 研读阶段
    log(f"🧠 开始深度研读 (入选 {len(kept_papers)} 篇)...")
    report_content = ""
    
    for paper in kept_papers:
        summary = generate_deep_dive(paper, paper['eval'])
        
        report_content += summary
        report_content += f"\n🔗 **原文直达**: [{paper['source']} Link]({paper['url']})\n"
        tags = paper['eval'].get('tags', [])
        plant_score = paper['eval'].get('plant_score', 0)
        ai_score = paper['eval'].get('ai_score', 0)
        report_content += f"> 🏷️ **自动标签**: `{', '.join(tags)}` | 📊 **评分**: Plant({plant_score}) AI({ai_score})\n"
        report_content += "---\n\n"
        
        time.sleep(2) 

    return report_content, len(kept_papers)

def main():
    log(f"🚀 启动 Bio-AI 情报 Agent (v13.0 Hybrid: Gemini + Zhipu)...")
    seen_papers = set()
    all_papers = []
    
    all_papers.extend(fetch_arxiv(seen_papers, max_results=10))
    all_papers.extend(fetch_biorxiv(seen_papers, limit=10))
    all_papers.extend(fetch_pubmed(seen_papers, max_results=5))
    
    log(f"\n📊 宽召回阶段：共获取 {len(all_papers)} 篇候选论文...\n")
    
    if not all_papers:
        log("未获取到任何论文。")
        return

    report_body, kept_count = process_papers(all_papers)

    daily_report = f"# 🧠 Bio-AI 每日情报决策 ({datetime.date.today()})\n"
    daily_report += f"> 📊 今日大盘：召回 {len(all_papers)} 篇 -> AI 严选 {kept_count} 篇\n"
    daily_report += "> 🤖 引擎策略：Gemini 2.5 Flash (Main) + Zhipu GLM-4 (Backup)\n\n"
    
    if kept_count == 0:
        daily_report += "### 今日无高价值论文入选\n建议明天继续关注。\n"
    else:
        daily_report += report_body

    print(daily_report)
    log("\n✅ 任务完成。")

if __name__ == "__main__":
    main()
