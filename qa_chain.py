from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
from datetime import datetime

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore_index", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 25, "lambda_mult": 0.5},
)

def build_context(docs, max_chars=1800):
    buffer, total = [], 0
    for d in docs:
        text = d.page_content.strip()
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                buffer.append(text[:remaining])
            break
        buffer.append(text)
        total += len(text)
    return "\n\n---\n\n".join(buffer)

repo_id = "google/flan-t5-base"
hf_model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
hf_tokenizer = AutoTokenizer.from_pretrained(repo_id)
hf_pipeline = pipeline(
    "text2text-generation",
    model=hf_model,
    tokenizer=hf_tokenizer,
    max_new_tokens=64,
    do_sample=False,
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
examples = [
    {
        "context": "Revenue in 2023 was $61B. No data about 2024.",
        "question": "What was the revenue in 2024?",
        "answer": "I don't know.",
    },
    {
        "context": "Key risks include supply chain disruption and cybersecurity.",
        "question": "What are the main risks?",
        "answer": "Supply chain disruption and cybersecurity.",
    },
    {
        "context": "Operating income was $8.2B; net income was $6.9B.",
        "question": "What was the net income?",
        "answer": "$6.9B.",
    },
]

example_prompt = PromptTemplate.from_template(
    "Context:\n{context}\nQuestion:\n{question}\nAnswer:\n{answer}\n"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=(
        "You are a precise financial analyst. Base answers ONLY on the context. "
        "If the answer is not explicitly present, reply exactly: I don't know.\n\n"
        "Follow the concise style shown in the examples.\n\nExamples:\n"
    ),
    suffix="Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
    input_variables=["context", "question"],
)

chain = (
    {"context": retriever | (lambda docs: build_context(docs, max_chars=1800)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

queries = [
    "What was Accenture's total revenue in 2024?",
    "What is the net income reported by Accenture?",
    "How much did the company spend on research and development in 2024?",
    "List the key financial highlights for 2024.",
    "What are the main risks described?",
    "What regulatory or compliance challenges are mentioned?",
    "Which cybersecurity risks does Accenture highlight?",
    "In which markets does Accenture operate?",
    "What are the strategic priorities for the next year?",
    "How is the company's business segmented?",
    "Which innovation initiatives were developed this year?",
    "What sustainability or ESG actions did Accenture take?",
    "What programs are in place for diversity and inclusion?",
    "How many employees does Accenture have?",
    "Who are Accenture's main clients?",
    "What strategic partnerships are referenced?",
]

for i, q in enumerate(queries, 1):
    print("\n" + "="*80)
    print(f"[{i}] Q: {q}")
    docs = retriever.invoke(q)
    for j, d in enumerate(docs, 1):
        src = d.metadata.get("source", "")
        page = d.metadata.get("page", "?")
        print(f"[SOURCE {j}] {src} p.{page}")
    context = build_context(docs)
    print("--- Context used ---")
    print(context)
    ans = chain.invoke(q)
    print("Answer:", ans)

    try:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_dir = os.path.join("docs", "reports", ts)
        os.makedirs(base_dir, exist_ok=True)

        safe = "".join([c if c.isalnum() else "-" for c in q])
        safe = "-".join([seg for seg in safe.split("-") if seg])[:80]
        fname = f"{i:02d}-{safe}.md"
        fpath = os.path.join(base_dir, fname)

        sources_lines = []
        for j, d in enumerate(docs, 1):
            src = d.metadata.get("source", "")
            page = d.metadata.get("page", "?")
            sources_lines.append(f"- SOURCE {j}: {src} p.{page}")

        md = []
        md.append(f"# Q{i}: {q}\n")
        md.append("## Sources\n")
        md.append("\n".join(sources_lines) + "\n")
        md.append("## Context Used\n")
        md.append("```\n" + context + "\n```\n")
        md.append("## Answer\n")
        md.append(ans + "\n")

        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md))

        index_path = os.path.join(base_dir, "index.md")
        with open(index_path, "a", encoding="utf-8") as idx:
            idx.write(f"- [Q{i}] {q} -> {fname}\n")

        print(f"Report saved: {fpath}")
    except Exception as e:
        print(f"Warning: Could not save report for Q{i}: {e}")