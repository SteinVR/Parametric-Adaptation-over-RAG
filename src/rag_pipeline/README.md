# PDF RAG Pipeline

Автономный пакет для построения RAG-системы поверх PDF-документов.
Включает полный цикл: **извлечение текста из PDF -> многоуровневое чанкирование -> гибридная индексация (Dense + Sparse) -> гибридный поиск с RRF-фьюжном -> cross-encoder reranking -> evidence compression -> page grounding**.

## Установка

```bash
pip install PyMuPDF>=1.24 qdrant-client>=1.9 sentence-transformers>=3.0 transformers>=4.40 pydantic>=2.0 huggingface_hub>=0.23
```

Или из `requirements.txt`:
```bash
pip install -r src/rag_pipeline/requirements.txt
```

Для работы Dense-эмбеддингов нужна скачанная модель `Qwen/Qwen3-Embedding-0.6B` (HuggingFace).
Для Reranker'а — `Qwen/Qwen3-Reranker-0.6B`.

---

## Архитектура

```
PDF файлы
    │
    ▼
┌─────────────────────────────────┐
│  1. INGESTION (ingestion/)      │
│  PyMuPDF → text + tables        │
│  → ParsedDocument               │
│  → CanonicalPageRecord (corpus) │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  2. INDEXING (indexing/)         │
│  Chunking: 5 уровней            │
│  Dense: Qwen3-Embedding-0.6B   │
│  Sparse: BM25 Okapi            │
│  → Qdrant (hybrid collection)  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  3. RETRIEVAL (retrieval/)      │
│  Hybrid Search (Dense + Sparse) │
│  Reciprocal Rank Fusion (RRF)  │
│  Cross-Encoder Reranking       │
│  Evidence Compression          │
│  Page Grounding                │
│  → RetrievalResult             │
└─────────────────────────────────┘
```

---

## 1. Ingestion — извлечение из PDF

### Что происходит

1. **PyMuPDF** открывает PDF и для каждой страницы:
   - Извлекает полный текст (`page.get_text("text")`)
   - Извлекает текстовые блоки/параграфы (`page.get_text("blocks")`)
   - Находит таблицы (`page.find_tables()`) и конвертирует в Markdown/HTML
2. Первая страница используется для извлечения **заголовка** и **классификации документа**:
   - `law_or_regulation` — если содержит "LAW NO." / "DIFC LAW"
   - `court_or_arbitration_decision` — если есть "COURT OF", "JUDGMENT" и т.д.
   - `other_legal_document` — по умолчанию
3. Таблицы **сериализуются** в текстовые блоки для поиска:
   - Каждая строка таблицы становится отдельным блоком
   - К блоку добавляется контекст: заголовки столбцов, caption, сноски

### Ключевые структуры данных

- `ParsedPage` — текст, блоки, таблицы одной страницы
- `ParsedDocument` — весь документ (список страниц + метаданные)
- `CanonicalPageRecord` — финальная запись страницы для индексации (текст + блоки контента)

### Пример

```python
from pdf_rag_pipeline import parse_pdf, serialize_document_tables, build_corpus

# 1. Парсинг PDF
doc = parse_pdf("contract.pdf")
print(f"Страниц: {doc.page_count}, Тип: {doc.document_family}")

# 2. Сериализация таблиц
table_blocks = serialize_document_tables(doc)

# 3. Сборка корпуса
corpus_records = build_corpus(doc, table_blocks)
```

---

## 2. Chunking — многоуровневое разбиение на чанки

Текст разбивается на **5 типов чанков**, каждый оптимизирован под свой сценарий поиска.

### Иерархия чанков

```
Документ
  │
  ├── Section    ─── группа страниц под одним заголовком
  │   (например, "Part 2 — Termination", страницы 5-8)
  │
  ├── Page       ─── одна страница целиком
  │
  ├── Clause     ─── один нумерованный пункт
  │   (например, "(a) The employee shall...")
  │   │
  │   └── Microchunk ── предложения внутри clause
  │         (бюджет: 300 токенов, overlap: 50 токенов)
  │
  └── Table      ─── строки сериализованных таблиц
```

### Как работает каждый уровень

#### Page chunks
Весь текст страницы = один чанк. ID: `{doc_id}-p{N}-page`.

#### Section chunks
Несколько страниц, объединённых одним заголовком. Заголовки (Part, Article, Section) извлекаются регулярными выражениями и **прокидываются вперёд** (heading carry-forward): если на странице 2 нет заголовка, она наследует его со страницы 1.

#### Clause chunks
Юридически-осведомлённая сегментация. Ищутся нумерованные пункты: `(a)`, `(1)`, `10A.` и т.д. Каждый пункт — отдельный чанк. Если перед пунктами есть вводный текст — он становится `lead-in` чанком. Если нумерации нет — текст делится на параграфы.

#### Microchunks
Clause дробится на предложения с бюджетом **300 токенов**. Алгоритм:
1. Разбить текст на предложения по `.!?`
2. Накапливать предложения, пока не превышен бюджет
3. Создать чанк, откатить на **50 токенов overlap** для следующего

Токенизатор: Qwen3 (или regex-фоллбэк `[a-z0-9]+`).

#### Table chunks
Каждая строка таблицы с контекстом столбцов и caption.

### Метаданные чанка (`IndexChunk`)

Каждый чанк несёт:

| Поле | Описание |
|---|---|
| `chunk_id` | Уникальный ID, например `doc-p1-clause-3-micro-1` |
| `chunk_type` | `page` / `section` / `clause` / `microchunk` / `table` |
| `text` | Текст чанка |
| `page_span` | Какие страницы покрывает |
| `section` | Заголовок раздела |
| `clause` | Метка пункта или `lead-in` |
| `neighboring_headings` | Иерархия: `["Part 2", "Section 4"]` |
| `entities` | Именованные сущности (regex, до 8) |
| `dates` | Даты (regex, до 6) |
| `bm25_terms` | Топ-32 термина для sparse search |
| `token_count` | Количество токенов |

### Пример

```python
from pdf_rag_pipeline import build_index_chunks

chunks = build_index_chunks(
    corpus_records,
    enabled_chunk_families={"page", "section", "clause", "microchunk", "table"},
    token_chunk_size=300,
    token_chunk_overlap=50,
)
print(f"Всего чанков: {len(chunks)}")
for c in chunks[:3]:
    print(f"  {c.chunk_type}: {c.chunk_id} ({c.token_count} tok)")
```

---

## 3. Indexing — Dense + Sparse → Qdrant

Каждый чанк индексируется **двумя способами** одновременно.

### Dense-эмбеддинги (семантический поиск)
- **Модель:** `Qwen/Qwen3-Embedding-0.6B` (SentenceTransformer)
- Для документов: `prompt_name="document"`
- Для запросов: `prompt_name="query"`
- Нормализация L2, косинусное расстояние

### Sparse-эмбеддинги (лексический поиск, BM25)
- **Алгоритм:** BM25 Okapi (k1=1.5, b=0.75)
- Токенизация: `[a-z0-9]+` (lowercase)
- Вычисляет IDF для каждого термина
- Формула: `score = IDF * (tf * (k1+1)) / (tf + k1 * (1 - b + b * dl/avgdl))`

### Хранилище: Qdrant
Локальная embedded БД. Каждая точка содержит:
- Именованный вектор `dense` (cosine)
- Именованный вектор `sparse` (BM25)
- Payload: полные метаданные чанка

### Пример

```python
from pdf_rag_pipeline import (
    Qwen3DenseEmbedder, BM25SparseEncoder, build_and_persist_index
)
from pathlib import Path

embedder = Qwen3DenseEmbedder()
dense_vectors = embedder.encode([c.text for c in chunks])
sparse_encoder = BM25SparseEncoder([c.text for c in chunks])

build_and_persist_index(
    chunks=chunks,
    dense_vectors=dense_vectors,
    sparse_encoder=sparse_encoder,
    qdrant_dir=Path("output/index/qdrant"),
    collection_name="document_index",
)
```

---

## 4. Retrieval — полный пайплайн поиска

Retrieval состоит из 4 стадий, которые можно использовать по отдельности или через единый оркестратор `RetrievalService`.

### 4.1 Hybrid Search + RRF

1. Запрос кодируется **dense** (Qwen3, prompt="query") и **sparse** (BM25)
2. Из Qdrant достаются кандидаты по обоим каналам (3x top_k каждый)
3. **Reciprocal Rank Fusion (RRF)** объединяет ранги:
   ```
   score(chunk) = dense_weight/(rrf_k + rank_dense) + sparse_weight/(rrf_k + rank_sparse)
   ```
   Где `rrf_k=60` по умолчанию.
4. Результаты сортируются по combined score

### 4.2 Reranking — cross-encoder переранжирование

После RRF-фьюжна топ-кандидаты переранжируются cross-encoder моделью `Qwen/Qwen3-Reranker-0.6B`.

**Как работает:**
1. Для каждого кандидата формируется prompt: `<Instruct>: ... <Query>: запрос <Document>: текст чанка`
2. Модель (causal LM) генерирует logits для следующего токена
3. Из вероятностей "yes" vs "no" вычисляется скор релевантности: `softmax(logit_yes, logit_no)[1]`
4. Кандидаты пересортируются по этому скору

**Fallback:** если модель недоступна (не скачана, нет GPU), автоматически используется `LexicalFallbackReranker` — лёгкий ранкер на основе пересечения токенов запроса и документа.

### 4.3 Evidence Compression

`EvidenceCompressor` отбирает финальный набор evidence-чанков из переранжированных кандидатов с учётом **page diversity**:

1. Чанки группируются по страницам (через `PageLifter`)
2. Страницы ранжируются по лучшему скору чанка на странице
3. С каждой страницы берётся не более `max_chunks_per_page` чанков (по умолчанию 1)
4. Итерация по страницам продолжается до заполнения `evidence_budget`

Это гарантирует, что в финальном наборе evidence будут чанки с **разных страниц**, а не 3 чанка с одной и той же страницы.

### 4.4 Page Grounding

`PageLifter` агрегирует чанки в дедуплицированные `PageReference` объекты:

```python
PageReference(doc_id="contract", page_numbers=[5, 7, 8])
```

Это позволяет downstream-системе точно указать, из каких страниц каких документов получен ответ.

---

## 5. RetrievalService — единый оркестратор

`RetrievalService` объединяет все стадии в один вызов `retrieve(query)`:

```
query (str)
  │
  ▼
HybridSearchEngine.search()     candidate_budget=10 чанков из Qdrant
  │
  ▼
Reranker.rerank()                rerank_budget=8 лучших переранжируются
  │
  ▼
min_rerank_score фильтр          отбрасывает слабых кандидатов
  │
  ▼
EvidenceCompressor.compress()    evidence_budget=3, page diversity
  │
  ▼
PageLifter.to_page_references()  дедупликация страниц
  │
  ▼
RetrievalResult
  ├── evidence_chunks: list[RetrievedChunk]
  ├── page_references: list[PageReference]
  ├── candidate_count, reranked_count
  └── is_unanswerable: bool
```

### Пример

```python
from pdf_rag_pipeline import (
    RetrievalService, LexicalFallbackReranker,
    HybridSearchEngine, build_query_embedder,
)

engine = HybridSearchEngine(
    qdrant_dir=config.qdrant_dir,
    dense_embedder=build_query_embedder(),
    sparse_encoder=sparse_encoder,
    collection_name="document_index",
)

service = RetrievalService(
    search_backend=engine,
    reranker=LexicalFallbackReranker(),
    candidate_budget=10,
    rerank_budget=8,
    evidence_budget=3,
)

result = service.retrieve("What are the requirements for forming a partnership?")

print(f"Кандидатов: {result.candidate_count}")
print(f"Переранжировано: {result.reranked_count}")
print(f"Evidence: {len(result.evidence_chunks)}")
print(f"Страницы: {[(pr.doc_id, pr.page_numbers) for pr in result.page_references]}")

for chunk in result.evidence_chunks:
    print(f"  [{chunk.rerank_score:.4f}] {chunk.doc_id} p{chunk.page_span}: {chunk.text[:80]}...")

engine.close()
```

---

## Полный E2E пример

```python
from pathlib import Path
from pdf_rag_pipeline import (
    PipelineConfig,
    parse_pdf,
    serialize_document_tables,
    build_corpus,
    build_index_chunks,
    Qwen3DenseEmbedder,
    BM25SparseEncoder,
    build_and_persist_index,
    build_query_embedder,
    HybridSearchEngine,
    RetrievalService,
    LexicalFallbackReranker,
)

# ── Конфигурация ──
config = PipelineConfig(
    documents_dir=Path("data/pdfs"),
    output_dir=Path("data/output"),
)
config.ensure_directories()

# ── 1. Ingestion ──
all_records = []
for pdf_path in config.documents_dir.glob("*.pdf"):
    doc = parse_pdf(pdf_path)
    table_blocks = serialize_document_tables(doc)
    records = build_corpus(doc, table_blocks)
    all_records.extend(records)

print(f"Корпус: {len(all_records)} страниц")

# ── 2. Chunking ──
chunks = build_index_chunks(
    all_records,
    enabled_chunk_families=set(config.enabled_chunk_families),
    token_chunk_size=config.token_chunk_size,
    token_chunk_overlap=config.token_chunk_overlap,
)
print(f"Чанков: {len(chunks)}")

# ── 3. Indexing ──
embedder = Qwen3DenseEmbedder(model_name=config.embedding_model)
dense_vectors = embedder.encode([c.text for c in chunks])
sparse_encoder = BM25SparseEncoder([c.text for c in chunks])

build_and_persist_index(
    chunks=chunks,
    dense_vectors=dense_vectors,
    sparse_encoder=sparse_encoder,
    qdrant_dir=config.qdrant_dir,
    collection_name=config.qdrant_collection,
)

# ── 4. Retrieval ──
engine = HybridSearchEngine(
    qdrant_dir=config.qdrant_dir,
    dense_embedder=build_query_embedder(),
    sparse_encoder=sparse_encoder,
    collection_name=config.qdrant_collection,
)

service = RetrievalService(
    search_backend=engine,
    reranker=LexicalFallbackReranker(),
    candidate_budget=config.candidate_budget,
    rerank_budget=8,
    evidence_budget=3,
)

result = service.retrieve("What is the governing law?")

for chunk in result.evidence_chunks:
    print(f"[{chunk.rerank_score:.3f}] {chunk.doc_id} p{chunk.page_span} | {chunk.text[:120]}...")
print(f"Страницы: {[(pr.doc_id, pr.page_numbers) for pr in result.page_references]}")

engine.close()
```

---

## Конфигурация

Все параметры — в `PipelineConfig`:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `documents_dir` | — | Папка с PDF |
| `output_dir` | — | Куда писать артефакты |
| `token_chunk_size` | 300 | Бюджет токенов на microchunk |
| `token_chunk_overlap` | 50 | Overlap между microchunks |
| `enabled_chunk_families` | все 5 | Какие типы чанков генерировать |
| `embedding_model` | Qwen3-0.6B | Dense-модель |
| `embedding_batch_size` | 16 | Батч для эмбеддинга |
| `qdrant_collection` | document_index | Имя коллекции в Qdrant |
| `candidate_budget` | 10 | Сколько чанков достать из Qdrant |
| `candidate_multiplier` | 3 | Множитель prefetch для каждого канала |
| `dense_weight` | 1.0 | Вес dense-канала в RRF |
| `sparse_weight` | 1.0 | Вес sparse-канала в RRF |
| `rrf_k` | 60 | Параметр k в формуле RRF |

Бюджеты `RetrievalService`:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `candidate_budget` | 10 | Чанков из hybrid search |
| `rerank_budget` | 5 | Из них передаётся reranker'у |
| `evidence_budget` | 3 | Финальных evidence чанков |
| `min_rerank_score` | 0.0 | Порог отсечения после reranking'а |

---

## Структура пакета

```
pdf_rag_pipeline/
├── __init__.py              # Публичный API
├── config.py                # PipelineConfig
├── schemas.py               # Pydantic-модели данных
├── requirements.txt         # Зависимости
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py        # PyMuPDF: PDF → ParsedDocument
│   ├── table_serializer.py  # Таблицы → текстовые блоки
│   └── corpus_builder.py    # Сборка корпуса
├── indexing/
│   ├── __init__.py
│   ├── chunking.py          # 5 уровней чанкирования
│   ├── embeddings.py        # Qwen3 Dense + BM25 Sparse
│   └── qdrant_store.py      # Запись/чтение Qdrant
└── retrieval/
    ├── __init__.py
    ├── hybrid_search.py     # Hybrid Search + RRF
    ├── reranker.py          # Cross-encoder Qwen3 + lexical fallback
    ├── evidence_compressor.py  # Page-diverse evidence selection
    ├── page_lifter.py       # Chunk→page агрегация и grounding
    └── service.py           # RetrievalService — единый оркестратор
```
