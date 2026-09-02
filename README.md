# RAG Observability with LangSmith

A hands-on **RAG observability lab** for understanding what happens inside a Retrieval-Augmented Generation pipeline when an answer is correct, ambiguous, or wrong.

Instead of treating the final LLM response as a black box, this project traces the retrieval and generation path so you can inspect **retrieved documents, similarity scores, LLM inputs/outputs, latency, failures, and execution flow** using LangSmith.

> **Learning-only project:** All policies, credentials, operational values, and examples in this repository are synthetic and created for experimentation. They do not represent a real organization or production security policy.

---

## Why I Built This

A RAG application can return the wrong answer for very different reasons:

- **Retrieval failure** — the right context was never retrieved.
- **Generation failure** — relevant context was retrieved, but the model did not use it correctly.

From the outside, both failures can look identical: **a wrong answer**.

This project explores how observability helps narrow down which layer actually failed instead of blindly tuning embeddings, prompts, or retrieval parameters.

---

## Architecture

```text
User Question
     |
     v
Vector Search
     |
     v
Retrieved Chunks + Similarity Scores
     |
     v
Prompt / Context Construction
     |
     v
LLM Generation
     |
     v
Final Answer

        +----------------------+
        |   LangSmith Tracing  |
        |----------------------|
        | Retriever spans      |
        | Chain spans          |
        | LLM calls            |
        | Inputs / Outputs     |
        | Latency / Failures   |
        +----------------------+
```

---

## What This Project Covers

### 1. LangSmith tracing fundamentals

The notebook starts with small custom spans using `@traceable` to make the execution hierarchy visible:

```python
@traceable(run_type="retriever", name="Mock Document Retriever")
def search_docs(question):
    ...

@traceable(run_type="llm", name="Mock LLM Generator")
def LLM_Fake(question, docs):
    ...

@traceable(run_type="chain", name="Mock Pipeline")
def pipeline(question):
    ...
```

This makes it easier to understand how retrievers, LLM calls, tools, and chains appear inside LangSmith before moving to the full RAG pipeline.

### 2. Enterprise Security RAG experiment

A synthetic enterprise AI security policy PDF is processed through:

```text
PDF
 -> RecursiveCharacterTextSplitter
 -> OpenAI Embeddings
 -> InMemoryVectorStore
 -> Retriever
 -> Prompt
 -> ChatOpenAI
 -> Answer
```

The pipeline uses:

- `PyPDFLoader`
- `RecursiveCharacterTextSplitter`
- `OpenAIEmbeddings`
- `InMemoryVectorStore`
- `ChatPromptTemplate`
- `ChatOpenAI`
- LangChain LCEL
- LangSmith tracing

The RAG chain is invoked with custom LangSmith **run names, tags, and metadata** so individual executions are easier to inspect.

### 3. Programmatic trace inspection

The project also uses the LangSmith `Client` to inspect recent runs programmatically, including:

- run name
- run type
- status
- latency
- token usage when available

### 4. Synthetic IT Helpdesk RAG experiment

A second, smaller knowledge base makes retrieval behavior easier to inspect.

Example synthetic policies include:

- lost or stolen device procedure
- VPN troubleshooting
- password reset
- software installation
- laptop replacement
- account lockout

The experiment is built around two traced functions.

#### `retrieve_it_policy()`

```python
@traceable(run_type="retriever", name="retrieve_it_policy")
def retrieve_it_policy(question: str, k: int = 2) -> list[dict]:
    results = helpdesk_vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": round(float(score), 4),
        }
        for doc, score in results
    ]
```

This exposes the retrieved chunks **and their similarity scores** instead of only returning documents.

#### `it_helpdesk_rag()`

```python
@traceable(run_type="chain", name="it_helpdesk_rag")
def it_helpdesk_rag(question: str) -> dict:
    retrieved = retrieve_it_policy(question, k=2)

    context = "\n\n".join(
        item["content"] for item in retrieved
    )

    messages = helpdesk_prompt.format_messages(
        context=context,
        question=question,
    )

    answer = llm.invoke(messages).content

    return {
        "question": question,
        "answer": answer,
        "retrieved": retrieved,
    }
```

This creates a clean trace from **retrieval -> context construction -> LLM generation -> final output**.

---

## Experiments

The helpdesk RAG system is tested with three types of questions.

### Test 1 — Clear answer exists

```text
My company laptop was stolen. What should I do,
and how quickly should I report it?
```

Goal: verify that the relevant lost-device policy is retrieved and the model answers from it.

### Test 2 — Ambiguous question

```text
I cannot access company systems from home.
What should I check?
```

This produced the most interesting debugging case during the experiment.

The retriever surfaced the **VPN troubleshooting policy**, but the model still returned that it could not find the answer in the policy.

That is exactly the kind of behavior observability is useful for: instead of immediately blaming retrieval, the trace shows that relevant context reached the pipeline and shifts the investigation toward **prompt construction, context quality, or generation behavior**.

### Test 3 — Missing knowledge

```text
How many paid vacation days do employees receive each year?
```

The knowledge base contains no vacation policy.

Goal: verify that the model stays grounded and refuses to invent an answer.

---

## What to Inspect in LangSmith

For each `it_helpdesk_rag` run, inspect:

1. **Retriever input** — what question was sent to retrieval?
2. **Retrieved documents** — which policies were returned?
3. **Similarity scores** — how strongly did the query match each document?
4. **LLM input** — what context actually reached the model?
5. **LLM output** — did the answer use the retrieved evidence?
6. **Latency** — which part of the pipeline took the most time?
7. **Failures** — which span failed and what caused it?

The central idea is simple:

> **Do not debug only the final answer. Debug the path that produced it.**

---

## Project Structure

```text
RAG_Observability/
|
|-- observability.ipynb
|-- enterprise_ai_security_policy.pdf
|-- .env                  # local only — do not commit
|-- .gitignore
|-- README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/RAG_Observability.git
cd RAG_Observability
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install \
  jupyter \
  python-dotenv \
  langchain \
  langchain-core \
  langchain-community \
  langchain-openai \
  langchain-text-splitters \
  langsmith \
  pypdf
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-observability-lab
```

Never commit `.env` or real API keys.

### 5. Run the notebook

```bash
jupyter notebook observability.ipynb
```

Run the notebook cells in order, then open your LangSmith project to inspect the generated traces.

---

## Recommended `.gitignore`

```gitignore
.env
.venv/
venv/
__pycache__/
.ipynb_checkpoints/
.DS_Store
```

---

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| RAG framework | LangChain |
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | LangChain `InMemoryVectorStore` |
| Observability | LangSmith |
| Document loading | PyPDFLoader |
| Notebook | Jupyter |

---

## What I Learned

This project reinforced a few important ideas about RAG engineering:

- A wrong answer does **not** automatically mean retrieval failed.
- Retrieval quality should be inspected independently from generation quality.
- Similarity scores provide useful debugging evidence, but they are not a complete quality metric by themselves.
- Tracing custom functions makes execution boundaries much easier to reason about.
- Observability turns an opaque LLM pipeline into a sequence of inspectable decisions.
- Missing-knowledge tests are important for checking whether a RAG system stays grounded instead of hallucinating.

---

## Current Scope and Limitations

This is intentionally a **learning lab**, not a production RAG service.

Current limitations include:

- in-memory vector storage
- small synthetic datasets
- no persistent database
- no authentication layer
- no automated retrieval benchmark yet
- no production deployment
- no systematic RAG evaluation suite yet

These constraints are deliberate so the project stays focused on **understanding and debugging observability**.

---

## Roadmap

The plan is to evolve the same pipeline instead of creating disconnected framework demos:

- [x] RAG pipeline
- [x] LangSmith observability
- [x] custom retriever and chain traces
- [x] similarity-score inspection
- [x] clear / ambiguous / missing-knowledge experiments
- [ ] LLM Gateway with Portkey AI
- [ ] LLM evaluation with DeepEval
- [ ] AI guardrails
- [ ] RAG evaluation with RAGAS
- [ ] compare retrieval configurations with measurable evaluation results

---

## Repository Goal

This repository is part of an ongoing effort to understand how GenAI systems behave beyond the happy path — not just how to make a RAG pipeline return an answer, but how to **observe, evaluate, debug, and eventually harden it**.

---

## License

This project is intended for educational and portfolio use. Add a license such as MIT if you plan to make the repository openly reusable.
