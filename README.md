# rag-sdk

一个最小可运行的 RAG 检索 SDK：

```
上传 PDF → 解析 (pypdf) → 切 Chunk → Embedding → 存入 Qdrant → 问题检索 Top-5
```

可以当作 **Python 库** 直接嵌入另一个项目，也可以作为 **HTTP 服务** 运行，并通过 SDK 内置的远程客户端调用。

## 项目结构

```
src/rag_sdk/
  pdf_parser.py    # 2. 解析 PDF，按页提取文本
  chunker.py       # 3. 递归字符切分（中英文标点），带 overlap，保留页码
  embeddings.py    # 4. Embedding：FastEmbed(本地) / OpenAI 兼容接口 / Hash(离线测试)
  vector_store.py  # 5. Qdrant 写入、删除、Top-K 检索
  client.py        # RAGClient：把整条流水线串起来（SDK 主入口）
  server.py        # 1. FastAPI：上传 PDF / 检索
  remote.py        # RAGRemoteClient：调用 HTTP 服务的 SDK 客户端
  cli.py           # 命令行 rag-sdk ingest / search / serve
examples/
  sample.pdf       # 示例 PDF
  consumer_app/    # 一个独立的调用方项目
tests/
```

## 安装

```bash
pip install -e ".[all]"          # 本地开发：SDK + fastembed + HTTP 服务
```

```bash
docker compose up -d             # 启动 Qdrant (http://localhost:6333)
```

不想启动 Qdrant 也可以：不设置 `QDRANT_URL` 时使用内存模式，设置 `QDRANT_PATH=./qdrant_data` 则使用本地磁盘模式。

## 作为 SDK 使用

```python
from rag_sdk import RAGClient

rag = RAGClient(qdrant_url="http://localhost:6333")   # 默认本地模型 BAAI/bge-small-zh-v1.5
info = rag.ingest_pdf("paper.pdf")
print(info.num_pages, info.num_chunks, info.doc_id)

for hit in rag.search("这篇文章的主要贡献是什么？", top_k=5):
    print(f"{hit.score:.3f}  p.{hit.page_number}  {hit.text[:80]}")
```

更换 Embedding 模型：

```python
from rag_sdk import RAGClient, FastEmbedEmbedder, OpenAIEmbedder

RAGClient(embedder=FastEmbedEmbedder("BAAI/bge-small-en-v1.5"))
RAGClient(embedder=OpenAIEmbedder(model="text-embedding-3-small"))           # 读取 OPENAI_API_KEY
RAGClient(embedder=OpenAIEmbedder(model="text-embedding-v3",                 # 任何 OpenAI 兼容接口
                                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                  api_key="sk-..."))
```

也可以继承 `rag_sdk.Embedder`，实现 `dim` 和 `embed_documents` 即可接入自己的模型。

其他参数：`collection_name`、`chunk_size`（默认 500 字符）、`chunk_overlap`（默认 50）、`qdrant_path`、`qdrant_api_key`。
同一文件重复导入会先删除旧数据再写入（`doc_id` 为文件内容的 SHA-256），`search(..., doc_id=...)` 可只在某一篇文档中检索。

## 作为 HTTP 服务使用

```bash
QDRANT_URL=http://localhost:6333 rag-sdk serve --port 8000      # Swagger: http://localhost:8000/docs
```

```bash
curl -F "file=@examples/sample.pdf" http://localhost:8000/documents
```

```bash
curl -X POST http://localhost:8000/search -H 'Content-Type: application/json' -d '{"query": "Which database is written in Rust?", "top_k": 5}'
```

在 Python 里调用该服务：

```python
from rag_sdk import RAGRemoteClient

with RAGRemoteClient("http://localhost:8000") as rag:
    rag.upload_pdf("paper.pdf")
    hits = rag.search("问题", top_k=5)
```

## 命令行

```bash
rag-sdk ingest examples/sample.pdf
```

```bash
rag-sdk search "Which database is written in Rust?" -k 5
```

## 环境变量（`RAGClient.from_env()`、CLI 和服务使用）

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `RAG_EMBEDDER` | `fastembed` / `openai` / `hash` | `fastembed` |
| `RAG_EMBEDDING_MODEL` | 模型名 | `BAAI/bge-small-zh-v1.5` / `text-embedding-3-small` |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | OpenAI 兼容接口 | |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant 服务 | |
| `QDRANT_PATH` | 本地磁盘模式（未设置 `QDRANT_URL` 时生效） | 内存模式 |
| `RAG_COLLECTION` | Collection 名 | `rag_chunks` |
| `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP` | 切分参数 | `500`, `50` |

> `hash` 是不依赖模型的哈希向量，只用于测试和离线演示，不具备语义能力。
> 注意：同一个 collection 必须使用相同维度的模型，换模型时请换 `RAG_COLLECTION`。

## 在另一个独立项目中调用

见 [`examples/consumer_app`](examples/consumer_app)：

```bash
pip install "rag-sdk[fastembed] @ git+https://github.com/fsy2020/rag-sdk.git"
```

```bash
python examples/consumer_app/main.py local examples/sample.pdf "Which database is written in Rust?"
```

```bash
python examples/consumer_app/main.py remote examples/sample.pdf "Which database is written in Rust?" --url http://localhost:8000
```

## 测试

```bash
pip install -e ".[dev]" && pytest
```

测试使用内存版 / 磁盘版 Qdrant 和 `HashEmbedder`，无需网络和 Qdrant 服务。

## 局限

- 扫描版 PDF（无文本层）需要先做 OCR，否则提取不到文字。
- 只做检索（返回 Top-K Chunk），不包含调用 LLM 生成答案的步骤。
