import arxiv
import os
import time
import requests
import re
import sys
import datetime
import json
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from Bio import Entrez
from google import genai
from google.genai import types

# ==========================================
# 0. 配置与日志
# ==========================================
def log(msg):
    print(msg, file=sys.stderr)

# API Keys & Config
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')
EMAIL_USER = "dongwei_li@hotmail.com" 
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_TO = "dongwei_li@hotmail.com"

# [核心修改] 严格的冷却时间配置 (秒)
GEMINI_COOLDOWN = 300  # Gemini 休息 5 分钟
ZHIPU_COOLDOWN = 180   # 智谱 休息 3 分钟

if not GOOGLE_API_KEY: raise ValueError("❌ 未找到 GOOGLE_API_KEY")
Entrez.email = EMAIL_USER

# 初始化
client_gemini = genai.Client(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

# ==========================================
# 1. 邮件发送模块
# ==========================================
def send_email(subject, body_markdown):
    if not EMAIL_PASSWORD:
        log("⚠️ 未配置 EMAIL_PASSWORD，跳过邮件发送。")
        return
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_TO
    msg['Subject'] = subject
    msg.attach(MIMEText(body_markdown, 'plain', 'utf-8'))
    try:
        server = smtplib.SMTP('smtp.office365.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())
        server.quit()
        log(f"✅ 邮件已成功发送至 {EMAIL_TO}")
    except Exception as e:
        log(f"❌ 邮件发送失败: {e}")

# ==========================================
# 2. 智能生成模块 (主动交替 + 严格限流)
# ==========================================
def call_gemini(prompt, is_json=False):
    """底层：调用 Gemini"""
    try:
        config = types.GenerateContentConfig(response_mime_type="application/json") if is_json else None
        response = client_gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
        return response.text
    except Exception as e: raise e

def call_zhipu(prompt, is_json=False):
    """底层：调用智谱 GLM-4"""
    if not ZHIPU_API_KEY: raise Exception("No Zhipu Key Configured")
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    messages = [{"role": "system", "content": "你是一个生物信息学科研助手。"}, {"role": "user", "content": prompt}]
    if is_json: messages[0]["content"] += " 请输出严格JSON。"
    payload = {"model": "glm-4-flash", "messages": messages, "stream": False, "temperature": 0.5}
    headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        return resp.json()['choices'][0]['message']['content'].replace("```json", "").replace("```", "").strip()
    except Exception as e: raise e

def generate_with_strategy(prompt, preferred_engine="gemini", is_json=False):
    """
    [核心逻辑] 根据指定的首选引擎尝试生成
    返回: (content, used_engine)
    """
    # 1. 尝试首选引擎
    if preferred_engine == "gemini":
        try:
            return call_gemini(prompt, is_json), "gemini"
        except Exception as e:
            log(f"   ⚠️ Gemini 失败 ({e})，尝试切换智谱...")
            # 失败则回退到智谱
            try:
                if ZHIPU_API_KEY: return call_zhipu(prompt, is_json), "zhipu"
            except: pass
            
    elif preferred_engine == "zhipu":
        try:
            if ZHIPU_API_KEY: return call_zhipu(prompt, is_json), "zhipu"
            else: raise Exception("No Zhipu Key")
        except Exception as e:
            log(f"   ⚠️ 智谱 失败 ({e})，尝试切换 Gemini...")
            # 失败则回退到 Gemini
            try:
                return call_gemini(prompt, is_json), "gemini"
            except: pass

    return None, "none"

# ==========================================
# 2. 搜索策略：宽召回 (Broad Recall)
# ==========================================
# 策略：用最少的词，覆盖最大的面。不要太细，太细了会漏。
SEARCH_KEYWORDS = [
    # --- 方向1: 植物单细胞 & 图谱 ---
    "plant single-cell", "scRNA-seq", "spatial transcriptomics", "cell atlas",
    
    # --- 方向2: 数据整合 & 多组学 ---
    "data integration", "multi-omics", "reference mapping",
    
    # --- 方向3: AI育种 & 基础模型 ---
    "foundation model", "deep learning genomics", "AI breeding", "trait prediction",
    
    # --- 核心物种限制 (辅助) ---
    "plant", "Arabidopsis", "rice", "maize" 
]

# 预编译正则 (保持不变)
COMPILED_PATTERNS = [re.compile(rf'\b{re.escape(k.lower())}\b') for k in SEARCH_KEYWORDS]

# ==========================================
# 3. Prompt: 阶段一 (裁判 - 评分与分类)
# ==========================================
RELEVANCE_PROMPT_TEMPLATE = """
You are a domain expert in **Plant Single-Cell**, **Data Integration**, and **AI Breeding**.
Your task is to JUDGE the relevance of this paper based on the user's specific research interests.

User's Core Interests:
1. **Plant Single-Cell**: scRNA-seq atlas, spatial transcriptomics, developmental trajectory.
2. **Data Integration**: Cross-species/dataset integration, batch correction, reference mapping, foundation models for representation learning.
3. **Plant AI Breeding**: Genotype-to-phenotype prediction, regulatory variant effect, crop improvement using AI.

Paper Metadata:
Title: {title}
Abstract: {abstract}

Step 1: Relevance Scoring (0-3) for EACH dimension:
- **Plant/Crop Relevance**: (0=None, 1=General Bio, 2=Plant Related, 3=Core Plant/Crop Study)
- **Single-Cell/Omics Relevance**: (0=None, 1=Bulk, 2=Single-Cell/Spatial/Multi-omics, 3=Atlas/Integration Level)
- **AI/Algorithm Relevance**: (0=None, 1=Stats, 2=ML/DL Application, 3=Foundation Model/New Algorithm)
- **Breeding/Function Relevance**: (0=None, 1=Basic Bio, 2=Functional study, 3=Breeding/Trait Prediction)

Step 2: Extract Species
- Extract the main organism (e.g., "Rice", "Maize", "Arabidopsis", "General Method").

Step 3: Decision Logic (Strict)
- **KEEP**: If the paper matches AT LEAST ONE of the User's Core Interests strongly (Score >= 2 in relevant dimensions).
    - Example: A generic AI method for single-cell integration is KEEP (transferable).
    - Example: A pure clinical human study is DROP.
- **DROP**: If strictly irrelevant (e.g., human cancer drug trials, pure math without bio application).

Step 4: Auto-Tagging
- Select tags: [ATLAS], [INTEGRATION], [AI_BREEDING], [METHOD], [SPATIAL], [MULTI_OMICS]

Output JSON format only:
{{
  "plant_score": int,
  "single_cell_score": int,
  "ai_score": int,
  "breeding_score": int,
  "species": "String",
  "decision": "KEEP" | "DROP",
  "tags": ["TAG1", "TAG2"],
  "reason": "One short sentence explaining why it matches the user's interests."
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
    try:
        abstract_obj = article_data.get('Abstract', {}).get('AbstractText', [])
        if not abstract_obj: return "No Abstract"
        parts = []
        items = abstract_obj if isinstance(abstract_obj, list) else [abstract_obj]
        for item in items:
            if isinstance(item, str): parts.append(item)
            elif isinstance(item, dict): parts.append(item.get('#text') or "")
        return " ".join(parts)
    except: return "No Abstract"

def is_duplicate(seen_set, title, source):
    key = (title.lower().strip(), source)
    if key in seen_set: return True
    seen_set.add(key)
    return False

# ==========================================
# 6. 核心流程 (严格慢速交替)
# ==========================================
def process_papers(papers):
    log(f"⚖️ 开始筛选 {len(papers)} 篇论文 (慢速交替模式)...")
    kept_papers = []
    
    # 引擎切换开关: 0=Gemini, 1=Zhipu
    engine_toggle = 0 
    
    # --- Phase 1: 评分筛选 ---
    for i, paper in enumerate(papers):
        # 决定当前用哪个引擎
        current_engine = "gemini" if (engine_toggle % 2 == 0) else "zhipu"
        
        log(f"   [{i+1}/{len(papers)}] 正在评分 (引擎: {current_engine})...")
        
        prompt = RELEVANCE_PROMPT_TEMPLATE.format(title=paper['title'], abstract=paper['abstract'])
        
        # 执行调用
        resp, used_engine = generate_with_strategy(prompt, preferred_engine=current_engine, is_json=True)
        
        # 处理结果
        try:
            eval_result = json.loads(resp)
        except:
            eval_result = {"decision": "KEEP", "tags": ["ERROR"], "species": "N/A"} # 容错
            
        if eval_result.get('decision') == "KEEP":
            paper['eval'] = eval_result
            kept_papers.append(paper)
            log(f"     -> ✅ KEEP")
        else:
            log(f"     -> ⏭️ DROP")
            
        # [关键] 根据实际使用的引擎，执行严格冷却
        if used_engine == "gemini":
            log(f"     ⏳ Gemini 完成，休息 {GEMINI_COOLDOWN} 秒...")
            time.sleep(GEMINI_COOLDOWN)
        elif used_engine == "zhipu":
            log(f"     ⏳ 智谱 完成，休息 {ZHIPU_COOLDOWN} 秒...")
            time.sleep(ZHIPU_COOLDOWN)
        else:
            time.sleep(10) # 失败时的默认短休息

        # 切换开关，下次换另一个
        engine_toggle += 1

    if not kept_papers: return "", 0

    # 2. 排序 (植物优先)
    kept_papers.sort(key=lambda p: (
        -p['eval'].get('plant_score', 0), 
        -p['eval'].get('ai_score', 0)
    ))

    # --- Phase 2: 深度研读 ---
    log(f"\n🧠 开始精读 {len(kept_papers)} 篇 (继续慢速交替)...")
    report_content = ""
    
    # 继续使用之前的开关状态，保持交替
    for i, paper in enumerate(kept_papers):
        current_engine = "gemini" if (engine_toggle % 2 == 0) else "zhipu"
        log(f"   [{i+1}/{len(kept_papers)}] 深度研读 (首选: {current_engine})...")

        hint = "重点分析迁移到植物研究的潜力。"
        if "METHOD" in paper['eval'].get('tags', []): hint += " 关注算法对稀疏数据的鲁棒性。"
        
        prompt = DEEP_DIVE_PROMPT_TEMPLATE.format(
            title=paper['title'], source=paper['source'], date=paper['date'],
            tags=",".join(paper['eval'].get('tags', [])), species=paper['eval'].get('species', 'N/A'),
            transfer_hint=hint, abstract=paper['abstract']
        )
        
        summary, used_engine = generate_with_strategy(prompt, preferred_engine=current_engine, is_json=False)
        
        if summary:
            report_content += summary + f"\n🔗 **Link**: {paper['url']}\n---\n\n"
        else:
            report_content += f"> ❌ {paper['title']} 解读失败\n---\n\n"

        # [关键] 再次执行严格冷却
        if used_engine == "gemini":
            log(f"     ⏳ Gemini 完成，休息 {GEMINI_COOLDOWN} 秒...")
            time.sleep(GEMINI_COOLDOWN)
        elif used_engine == "zhipu":
            log(f"     ⏳ 智谱 完成，休息 {ZHIPU_COOLDOWN} 秒...")
            time.sleep(ZHIPU_COOLDOWN)
            
        engine_toggle += 1

    return report_content, len(kept_papers)

# 抓取函数 (保持)
def fetch_arxiv(seen, limit=10):
    log("📡 [ArXiv] Searching...")
    papers = []
    query = ' OR '.join([f'ti:"{k}"' for k in SEARCH_KEYWORDS[:6]]) 
    try:
        client = arxiv.Client(page_size=limit, delay_seconds=3, num_retries=3)
        search = arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.SubmittedDate)
        for r in client.results(search):
            if not is_duplicate(seen, r.title, "ArXiv"):
                papers.append({"title": r.title, "abstract": r.summary, "url": r.entry_id, "date": r.published.strftime("%Y-%m-%d"), "source": "ArXiv"})
    except Exception as e: log(f"ArXiv Error: {e}")
    return papers

def fetch_biorxiv(seen, limit=10):
    log("📡 [BioRxiv] Searching...")
    papers = []
    try:
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=5)
        url = f"https://api.biorxiv.org/details/biorxiv/{from_date}/{today}/0/json"
        resp = requests.get(url).json()
        for item in resp.get('collection', [])[:limit*2]: 
            if len(papers) >= limit: break
            if not is_duplicate(seen, item['title'], "BioRxiv"):
                if any(k in (item['title']+item['abstract']).lower() for k in ["single-cell", "plant", "genomics", "deep learning"]):
                    papers.append({"title": item['title'], "abstract": item['abstract'], "url": f"https://doi.org/{item['doi']}", "date": item['date'], "source": "BioRxiv"})
    except Exception as e: log(f"BioRxiv Error: {e}")
    return papers

def fetch_pubmed(seen, limit=5):
    log("📡 [PubMed] Searching...")
    papers = []
    today = datetime.date.today().strftime("%Y/%m/%d")
    past = (datetime.date.today() - datetime.timedelta(days=5)).strftime("%Y/%m/%d")
    term = ' OR '.join([f'({k})' for k in SEARCH_KEYWORDS[:8]]) + f' AND ("{past}"[PDAT] : "{today}"[PDAT])'
    try:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=limit)
        id_list = Entrez.read(handle)["IdList"]
        if not id_list: return []
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        for art in records['PubmedArticle']:
            try:
                data = art['MedlineCitation']['Article']
                title = data['ArticleTitle']
                if not is_duplicate(seen, title, "PubMed"):
                    papers.append({"title": title, "abstract": parse_pubmed_abstract(data), "url": f"https://pubmed.ncbi.nlm.nih.gov/{art['MedlineCitation']['PMID']}/", "date": today, "source": "PubMed"})
            except: pass
    except: pass
    return papers

def main():
    log(f"🚀 Bio-AI Agent v15.0 (Strict Slow-Switch Mode)...")
    seen = set()
    all_p = []
    all_p.extend(fetch_arxiv(seen, 15))
    all_p.extend(fetch_biorxiv(seen, 15))
    all_p.extend(fetch_pubmed(seen, 10))
    
    if not all_p:
        log("No papers found.")
        return

    body, count = process_papers(all_p)
    
    report = f"# 🧠 Bio-AI Daily ({datetime.date.today()})\n"
    report += f"> 📊 Scanned: {len(all_p)} | Selected: {count}\n"
    report += f"> ⏳ Strategy: Gemini(5m) <-> Zhipu(3m)\n\n"
    if count == 0: report += "No relevant papers today.\n"
    else: report += body

    print(report)
    log("📧 正在发送邮件...")
    send_email(f"Bio-AI Report {datetime.date.today()}", report)

if __name__ == "__main__":
    main()
