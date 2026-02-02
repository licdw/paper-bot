# **🧬 Bio-AI Paper Bot: 植物单细胞与AI育种情报 Agent**

**Bio-AI Paper Bot** 是一个全自动化的科研情报参谋。它每天凌晨自动运行，利用 **Google Gemini** 和 **智谱 GLM-4** 双引擎，为你深度阅读最新的 **ArXiv, BioRxiv, PubMed** 文献，并筛选出对 **植物单细胞**、**多组学数据整合** 及 **AI 育种** 最具价值的研究。

## **✨ 核心特性 (v15.0)**

### **🧠 1\. 双引擎混合动力 (Hybrid Intelligence)**

* **Gemini 2.5 Flash (主力)**：Google 最新推理模型，速度快，负责主要的论文初筛和精读。  
* **Zhipu GLM-4 (协同)**：智谱 AI 模型。与 Gemini 组成双发引擎，**主动交替工作**，互为备份，确保任务 100% 完成。

### **🛡️ 仿生慢速阅读模式 (Anti-Ban Strategy)**

为了彻底解决 API 限流（429）和封号风险，本系统采用了**严格的慢速轮替策略**：

* **主动交替**：处理一篇论文后，自动切换引擎（Gemini \<-\> GLM-4）。  
* **强制冷却**：  
  * Gemini 处理完 \-\> 🛌 **强制休息 5 分钟**  
  * 智谱 GLM-4 处理完 \-\> ☕ **强制休息 3 分钟**  
* **结果**：任务分散在凌晨 04:00 \- 08:00 执行，极其安全稳定。

### **📊 智能决策与排序**

Agent 像审稿人一样工作，而非简单的摘要工具：

1. **裁判打分**：基于你的研究方向，对每篇论文进行 0-3 分的相关性打分。  
2. **智能排序**：优先展示 **植物/作物相关** 的高分论文，其次是 **核心算法**，最后是其他参考。  
3. **自动标签**：自动打上 ATLAS (图谱), METHOD (方法), BREEDING (育种) 等标签。

### **📧 每日晨报直达**

* **无感交付**：每天早上 8:00 前，一份排版精美的 Markdown 研报会自动发送到你的 **邮箱**。  
* **云端备份**：同时作为 Artifact 永久保存在 GitHub Actions 历史记录中。

## **🛠️ 部署指南 (Step-by-Step)**

本项目基于 **GitHub Actions** 运行，无需购买服务器，完全免费。请按照以下步骤操作：

### **第一步：准备密钥 (Keys)**

在开始部署前，你需要准备好以下 3 个凭证。请先申请好并记在记事本上。

#### **1\. Google Gemini API Key**

* **用途**: 主力 AI 引擎。  
* **获取地址**: [Google AI Studio](https://aistudio.google.com/app/apikey)  
* **操作**: 登录后点击左上角 **"Get API key"** \-\> **"Create API key"**。  
* **复制**: 复制以 AIza 开头的那一长串字符。

#### **2\. 智谱 AI API Key (GLM-4)**

* **用途**: 备用/协同 AI 引擎。  
* **获取地址**: [智谱大模型开放平台](https://open.bigmodel.cn/usercenter/apikeys)  
* **操作**: 注册并实名认证后，进入 **"API Key管理"**，创建一个新的 API Key。  
* **复制**: 复制生成的 Key。

#### **3\. 邮箱应用专用密码 (Email App Password)**

⚠️ **重要提示**：为了让代码能发邮件，**不能使用你的邮箱登录密码**，必须生成“应用密码”。

**以 Outlook / Hotmail 为例：**

1. 登录 [Microsoft 账户安全页](https://www.google.com/search?q=https://account.live.com/proofs/manage/additional)。  
2. 找到 **“高级安全选项”**。  
3. **开启双重验证 (2FA)**：如果未开启，通常无法生成应用密码（这是微软的安全规定）。  
4. 向下滚动找到 **“应用密码”** 区域。  
5. 点击 **“创建新的应用密码”**。  
6. **复制**: 复制生成的那个随机字符串密码（这就是我们要填的 EMAIL\_PASSWORD）。

### **第二步：Fork 本仓库**

1. 打开本项目的 GitHub 页面。  
2. 点击右上角的 **Fork** 按钮。  
3. 点击 **Create fork**，将项目复制到你自己的 GitHub 账号下。

### **第三步：配置保险箱 (GitHub Secrets)**

这一步是将你刚才申请的密钥告诉 GitHub，但不对外公开。

1. 进入你 Fork 后的仓库页面。  
2. 点击顶部菜单栏的 **Settings** (设置)。  
3. 在左侧菜单栏，向下找到 **Security** 区域。  
4. 点击 **Secrets and variables**，然后点击展开项里的 **Actions**。  
5. 点击右上角的绿色按钮 **New repository secret**。  
6. **依次添加以下 3 个密钥**（Name 必须完全一致，Value 填你刚才获取的内容）：

| Secret Name (变量名) | Secret Value (填入内容) |
| :---- | :---- |
| GOOGLE\_API\_KEY | 你的 Gemini API Key (AIza...) |
| ZHIPU\_API\_KEY | 你的智谱 API Key |
| EMAIL\_PASSWORD | 你的邮箱应用专用密码 (不是登录密码) |

### **第四步：个性化配置 (修改代码)**

你需要告诉机器人你的邮箱地址和感兴趣的研究方向。

1. 在仓库文件列表中，点击 main.py。  
2. 点击右侧的铅笔图标 ✏️ (Edit file)。  
3. **修改邮箱配置** (大约第 25 行)：  
   \# 修改为你自己的邮箱  
   EMAIL\_USER \= "your\_email@hotmail.com"   
   EMAIL\_TO \= "your\_email@hotmail.com" 

4. **修改研究关键词** (大约第 125 行)：  
   \# 修改为你关注的领域  
   SEARCH\_KEYWORDS \= \[  
       "plant single-cell", "scRNA-seq", "spatial transcriptomics",   
       "deep learning genomics", "AI breeding", "trait prediction",  
       "rice", "maize", "Arabidopsis"  
   \]

5. 点击右上角的 **Commit changes** 按钮保存修改。

### **第五步：启动运行**

配置全部完成！你可以选择自动运行或手动测试。

* **自动运行**：系统会在每天 **北京时间 04:00** (UTC 20:00) 自动唤醒运行。  
* **手动测试 (立即查看效果)**：  
  1. 点击仓库顶部的 **Actions** 选项卡。  
  2. 在左侧列表点击 **Daily AI Paper Digest**。  
  3. 点击右侧的 **Run workflow** 按钮 \-\> 再次点击绿色的 **Run workflow**。

## **📂 项目结构说明**

paper-bot/  
├── .github/workflows/  
│   └── daily\_paper\_bot.yml  \# 自动化调度文件 (定义了每天 4 点运行及环境变量注入)  
├── main.py                  \# 核心代码 (包含双引擎逻辑、邮件发送、评分系统)  
├── requirements.txt         \# Python 依赖包 (google-genai, arxiv, biopython等)  
└── README.md                \# 本说明文档

## **⚠️ 常见问题 (FAQ)**

**Q: 为什么运行时间这么长？**

A: 这是正常的。因为启用了 **防封号慢速模式**。为了保护你的 API Key，Gemini 每处理一篇论文会强制休息 **5分钟**，智谱会休息 **3分钟**。如果你一次抓取了 20 篇论文，运行时间可能超过 1 小时。请耐心等待邮件。

**Q: 为什么收不到邮件？**

A: 请检查：

1. EMAIL\_PASSWORD 是否填的是**应用专用密码**（绝大多数发信失败都是因为填了登录密码）。  
2. GitHub Secrets 的名字是否拼写正确（全大写）。  
3. 检查垃圾邮件箱（Spam）。

**Q: 我想修改运行时间怎么办？**

A: 修改 .github/workflows/daily\_paper\_bot.yml 文件中的 cron 字段。

* 例如：cron: '0 20 \* \* \*' 代表 UTC 20:00 (北京时间 04:00)。

## **📜 License**

MIT License. Feel free to modify and use for your own research\!