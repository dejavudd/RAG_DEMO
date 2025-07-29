# RAG (Retrieval-Augmented Generation) 示例项目

<div align="center">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/OpenAI-GPT--4-orange" alt="OpenAI">
</div>

## 📖 项目简介

这是一个演示检索增强生成（RAG）技术的 Python 项目。该项目展示了如何结合文本检索和大语言模型来构建一个智能问答系统，支持多模态输入（文本和图像），可以处理 PDF 文档并进行智能问答。

### 🌟 主要特性

- 📑 支持 PDF 文档的文本提取和处理
- 🖼️ PDF 页面转图像及表格自动识别
- 📊 向量数据库存储和检索
- 🤖 集成 OpenAI GPT 模型
- 🔍 支持文本和图像的智能问答
- 📏 多种相似度计算方法

## 🛠️ 技术架构

### 1. PDF 处理模块

#### 文本提取
```python
def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    '''从 PDF 文件中提取文字内容
    - filename: PDF 文件路径
    - page_numbers: 指定页码范围
    - min_line_length: 最小行长度
    '''
```

#### 图像转换
```python
def pdf2images(pdf_file):
    '''将 PDF 转换为图像
    - pdf_file: PDF 文件路径
    - 输出: 每页对应一张 PNG 图像
    '''
```

#### 表格检测
```python
def detect_and_crop_save_table(file_path):
    '''检测并提取表格
    - file_path: 图像文件路径
    - 输出: 保存提取的表格图像
    '''
```

### 2. 文本处理模块

#### 文本分块
```python
def split_text(paragraphs, chunk_size=300, overlap_size=100):
    '''文本分块处理
    - paragraphs: 文本段落
    - chunk_size: 块大小
    - overlap_size: 重叠大小
    '''
```

#### 相似度计算
```python
def cos_sim(a, b):
    '''余弦相似度计算'''

def l2(a, b):
    '''欧氏距离计算'''
```

### 3. 向量数据库

#### 基础版本
```python
class MyVectorDBConnector:
    '''向量数据库连接器
    功能:
    - 文档添加
    - 向量存储
    - 相似度检索
    '''
```

#### 增强版本
```python
class NewVectorDBConnector:
    '''支持图像的向量数据库连接器
    增加功能:
    - 图像内容提取
    - 图像向量化
    - 元数据存储
    '''
```

### 4. RAG 问答系统

#### 核心组件
```python
class RAG_Bot:
    '''RAG 问答机器人
    工作流程:
    1. 文档检索
    2. 提示词构建
    3. LLM 生成
    '''
```

## 🔧 环境配置

### 系统要求
- Python 3.8+
- CUDA 支持（推荐，用于图像处理）

### 依赖安装
```bash
pip install -r requirements.txt
```

### 主要依赖包
| 包名 | 版本 | 用途 |
|------|------|------|
| pdfminer | 最新版 | PDF 文本提取 |
| PyMuPDF | 最新版 | PDF 处理 |
| chromadb | 最新版 | 向量数据库 |
| openai | 最新版 | OpenAI API |
| torch | 最新版 | 深度学习支持 |
| transformers | 最新版 | 模型加载 |
| sentence_transformers | 最新版 | 文本向量化 |
| Pillow | 最新版 | 图像处理 |

## 📝 使用示例

### 1. 环境准备
```python
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()
```

### 2. 文本问答
```python
# 初始化向量数据库
vector_db = MyVectorDBConnector("demo", get_embeddings)

# 添加文档
vector_db.add_documents(paragraphs)

# 创建问答机器人
bot = RAG_Bot(vector_db, llm_api=get_completion)

# 进行问答
response = bot.chat("你的问题")
```

### 3. 图像问答
```python
# 初始化支持图像的向量数据库
new_db_connector = NewVectorDBConnector("demo", get_embeddings)

# 添加图像
new_db_connector.add_images(image_paths)

# 检索和问答
results = new_db_connector.search(query, 1)
response = image_qa(query, results["metadatas"][0][0]["image"])
```

## 📂 项目结构

```
RAG/
├── README.md               # 项目文档
├── day1.ipynb             # 主要代码和示例
├── requirements.txt       # 依赖包列表
├── .env                  # 环境变量配置
├── llama2.pdf            # 示例 PDF 文件
└── llama2_page8/         # 处理后的文件
    ├── page_1.png        # PDF 页面图像
    └── table_images/     # 提取的表格
        ├── page_1_0.png
        └── page_1_1.png
```

## ⚠️ 注意事项

1. **API 密钥配置**
   - 在 `.env` 文件中配置 OpenAI API 密钥
   - 注意保护密钥安全，不要提交到代码仓库

2. **资源需求**
   - 图像处理需要较大内存
   - 建议使用 GPU 加速（针对表格检测）

3. **性能优化**
   - 根据实际需求调整文本分块参数
   - 可以缓存向量计算结果
   - 考虑批量处理大量文档

4. **数据安全**
   - 向量数据库默认为演示模式，每次重启会重置
   - 生产环境需要配置持久化存储

## 🔄 更新日志

### v1.0.0 (2024-03)
- 🎉 首次发布
- 📑 支持基础的 PDF 处理
- 🤖 集成 OpenAI API
- 🔍 实现向量检索
- 📊 添加表格识别功能

## 📜 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。

1. Fork 本仓库
2. 创建特性分支
3. 提交改动
4. 发起 Pull Request

## 📚 参考资料

- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [Sentence Transformers 文档](https://www.sbert.net/)


---

<div align="center">
    <p>如果这个项目对你有帮助，欢迎 star ⭐️</p>
</div>