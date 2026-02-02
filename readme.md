这是一份基于你提供的文章内容的详细实施指南。

这份指南将教你如何利用 **Gemini 3.0 API**、**ArXiv** 和 **GitHub Actions**，搭建一个**零成本、无需服务器**的每日论文自动摘要机器人。

---

### 📋 项目概览与准备

* **目标**：每天早上 8:00 自动抓取最新 AI 论文，用 Gemini 3.0 生成中文摘要和创新点分析。
* **核心工具**：
* **Python**：处理逻辑。
* **Google Gemini API**：核心大脑，负责阅读和总结。
* **GitHub Actions**：自动化运维，负责每天定时运行。


* **准备工作**：
1. **GitHub 账号**：用于托管代码和运行自动任务。
2. **Google AI Studio API Key**：[点击申请](https://aistudio.google.com/)。



---

### 📂 第一步：创建项目文件结构

在你的电脑上新建一个文件夹（例如 `paper-bot`），并严格按照以下结构创建文件和文件夹：

```bash
mkdir -p /data/lidongwei/ai/paper-bot
cd /data/lidongwei/ai/paper-bot
# 1. 创建嵌套目录（.github/workflows），-p自动创建父目录
mkdir -p .github/workflows

# 2. 创建空的主程序、依赖清单、自动化流程文件
touch main.py requirements.txt .github/workflows/daily_paper_bot.yml
```



```text
paper-bot/               <-- 项目根目录
├── .github/
│   └── workflows/
│       └── daily_paper_bot.yml   <-- 自动化流程文件
├── main.py              <-- 主程序代码
└── requirements.txt     <-- 环境依赖清单

```

---

### 📝 第二步：编写代码文件

#### 1. 环境依赖清单 (`requirements.txt`)

在根目录下创建此文件，填入以下内容。这是告诉服务器需要安装哪些库。

```text
google-generativeai
arxiv
biopython
requests
```

#### 2. 主程序代码 (`main.py`)

在根目录下创建此文件。
**注意**：代码已根据文中提示进行了优化，加入了 `os` 模块以读取环境变量中的 Key，确保安全。

```python
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
ENTREZ_EMAIL = "your_real_email@gmail.com" 

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

```

#### 3. 自动化流程文件 (`.github/workflows/daily_paper_bot.yml`)

在 `.github/workflows/` 文件夹下创建此文件。
**注意**：`cron: '0 0 * * *'` 表示 UTC 时间 0 点，即北京时间早上 8 点。

```yaml
name: Daily AI Paper Digest

on:
  schedule:
    # 北京时间每天早上 08:00 (UTC 00:00)
    - cron: '0 0 * * *'
  # 允许手动触发测试
  workflow_dispatch:

jobs:
  build-and-run:
    runs-on: ubuntu-latest
    
    permissions:
      contents: read
      
    steps:
    - name: Check out repository code
      uses: actions/checkout@v4  # 升级到 v4

    - name: Set up Python
      uses: actions/setup-python@v5  # 升级到 v5
      with:
        python-version: '3.12'  # 升级到 3.12

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run paper summarizer & Save output
      env:
        GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
      # Python 脚本中的 log() 会显示在 Actions 控制台
      # Python 脚本中的 print() 会被写入 report.md
      run: python main.py > report.md

    - name: Upload Daily Report
      uses: actions/upload-artifact@v4
      with:
        name: Daily-Paper-Report
        path: report.md
        retention-days: 30

```

---

* 太棒了！你的目录结构（`tree -a .` 的结果）非常完美，**标准的 GitHub Actions 项目结构**。

  现在万事俱备，只欠“上传部署”。因为你是第一次处理，我将把接下来的步骤拆解得非常细，你只需要**像复制粘贴一样**跟着操作即可。

  我们将分为三个阶段：

  1. **准备阶段**：最后检查。
  2. **上传阶段**：把代码推送到 GitHub。
  3. **配置阶段**：填入密钥并启动。

  ------

  ### 第一阶段：最后检查 (本地)

  在上传之前，请务必确认 `main.py` 中的一行代码：

  1. 打开 `main.py`。
  2. 找到 `ENTREZ_EMAIL = "..."` 这一行。
  3. **确认里面填的是你真实的邮箱**（如 `xxxx@qq.com` 或 `xxxx@hotmail.com`）。
     - *原因：如果是默认的 `your_email`，程序运行会被报错终止。*

  ------

  ### 第二阶段：创建仓库并上传代码 (云端)

  我们需要在 GitHub 上给你的代码安个家。

  #### 1. 在 GitHub 网页上创建仓库

  1. 登录 [GitHub.com](https://github.com/)。
  2. 点击右上角的 **+** 号 -> **New repository**。
  3. **Repository name** 填入：`paper-bot` (名字最好和文件夹一致)。
  4. **Public/Private**：建议选 **Public** (私有仓库有时限制 GitHub Actions 的免费额度，但个人用 Private 也可以，推荐先 Public 方便调试)。
  5. **不要勾选** "Add a README file" (因为你本地已经有了)。
  6. 点击 **Create repository** 绿色按钮。

  #### 2. 将本地代码推送到 GitHub

  创建完成后，GitHub 会跳出一个页面，显示一堆代码。你不需要看懂，只需要在你的 Linux 终端（就是你运行 `tree` 的那个界面）依次执行以下命令：

  *(请将下面的 `你的GitHub用户名` 替换为你实际的用户名)*

  Bash

  ```
  # 1. 初始化 Git 仓库
  git init
  
  # 2. 将所有文件加入暂存区 (. 表示当前目录所有文件)
  git add .
  
  # 3. 提交文件，备注为 "Initial commit"
  git commit -m "Initial commit"
  
  # 4. 切换到主分支 (现在的标准是 main)
  git branch -M main
  
  # 5. 关联远程仓库 (这一步最重要，把 URL 换成你刚才创建的仓库地址)
  # 注意：地址通常是 https://github.com/你的用户名/paper-bot.git
  git remote add origin https://github.com/你的GitHub用户名/paper-bot.git
  
  # 6. 推送代码到云端
  git push -u origin main
  ```

  > **新手避坑提示**：
  >
  > 执行第 6 步 `git push` 时，如果它让你输入密码，现在的 GitHub **不再支持输入登录密码**，而是需要输入 **Personal Access Token (PAT)**。
  >
  > **如果你觉得命令行太麻烦，或者是第一次搞不定 Git 鉴权，可以用最简单的“网页上传法”：**
  >
  > 1. 在刚才创建好的 GitHub 仓库页面，点击 **"uploading an existing file"** 链接。
  > 2. 把你本地文件夹里的 `main.py`, `requirements.txt`, `readme.md` 直接拖进去。
  > 3. **注意**：网页上传很难处理 `.github/workflows` 文件夹。
  >    - **建议**：先拖那三个文件上传。
  >    - 然后点击 **Add file** -> **Create new file**。
  >    - 文件名输入：`.github/workflows/daily_paper_bot.yml` (注意斜杠会自动创建文件夹)。
  >    - 把你的 `daily_paper_bot.yml` 内容复制进去，点 **Commit changes**。

  ------

  ### 第三阶段：配置密钥 (至关重要)

  代码传上去后，GitHub Actions 虽然能看到代码，但它不知道你的 API Key。**绝对不能把 Key 写在代码里上传**，我们要用“保险箱”。

  1. 打开你刚才上传好的 GitHub 仓库页面。
  2. 点击顶部的 **Settings** (设置) 选项卡。
  3. 在左侧菜单栏，向下找，点击 **Secrets and variables**，然后点击展开项里的 **Actions**。
  4. 点击右侧绿色的 **New repository secret** 按钮。
  5. 填写信息：
     - **Name** (必须完全一致): `GOOGLE_API_KEY`
     - **Secret** (粘贴内容): `AIzaSy...` (这里粘贴你申请到的那一长串 Gemini API Key)。
  6. 点击 **Add secret** 保存。

  ------

  ### 第四阶段：启动并见证奇迹

  配置好密钥后，我们可以手动触发一次，看看能不能跑通。

  1. 点击仓库顶部的 **Actions** 选项卡。
  2. 在左侧你应该能看到 **Daily AI Paper Digest** (这是我们在 yaml 里写的名字)。点击它。
  3. 在右侧，你会看到一个灰色的条，或者一个 **Run workflow** 的按钮（在右边偏上位置）。
     - 点击 **Run workflow** -> 再次点击绿色的 **Run workflow**。
  4. **观察运行**：
     - 页面上会出现一个黄色的圆圈在转，表示正在运行。
     - 点击那个正在转的项目（通常叫 "Daily AI Paper Digest" 或 "Initial commit"），你可以实时看到它在安装 Python、安装依赖。
  5. **查看结果**：
     - 等待约 1-3 分钟，当黄色圆圈变成 **绿色对勾 ✅**，说明运行成功！
     - 点击这个运行记录进入详情页。
     - 划到页面最底部，找到 **Artifacts** 区域。
     - 你会看到一个叫 **Daily-Paper-Report** 的文件。
     - 点击下载，解压，打开里面的 `.md` 文件。

  ### 总结下一步操作清单

  1. **改邮箱**：确保 `main.py` 里是真邮箱。
  2. **建仓库**：GitHub 建个新库。
  3. **推代码**：`git init` ... `git push`。
  4. **填密钥**：Settings -> Secrets -> `GOOGLE_API_KEY`。
  5. **点运行**：Actions -> Run workflow。

  快去试试吧！如果哪一步报错（比如 git push 失败，或者 Actions 亮红灯），把报错截图或文字发给我，我帮你修！



### 💡 进阶提示 (原文提及)

如果你希望内容推送到**微信**而不是只在 GitHub 日志里看，你需要注册一个 **PushPlus** 账号，获取 Token，然后在 `main.py` 的最后部分，用 Python 的 `requests` 库将 `daily_report` 变量的内容发送到 PushPlus 提供的 URL 接口即可。
