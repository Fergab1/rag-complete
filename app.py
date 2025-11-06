import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

st.set_page_config(
    page_title="Financial Reports Q&A",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource(show_spinner=False)
def load_index():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "vectorstore_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 25, "lambda_mult": 0.5},
    )
    return retriever

retriever = load_index()

def build_context(docs, max_chars=1600):
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

@st.cache_resource(show_spinner=False)
def load_llm(repo_id: str = "google/flan-t5-small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    pipeline_obj = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=48,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipeline_obj)

examples = [
    {"context": "Revenue in 2023 was $61B. No data about 2024.", "question": "What was the revenue in 2024?", "answer": "I don't know."},
    {"context": "Key risks include supply chain disruption and cybersecurity.", "question": "What are the main risks?", "answer": "Supply chain disruption and cybersecurity."},
    {"context": "Operating income was $8.2B; net income was $6.9B.", "question": "What was the net income?", "answer": "$6.9B."},
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

def get_docs(query: str, k: int = 3):
    try:
        return retriever.vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=25,
            lambda_mult=0.5,
        )
    except Exception:
        return retriever.vectorstore.similarity_search(query, k=k)

st.title("RAG – Financial Reports (Accenture)")
st.caption("FAISS + HuggingFace + Flan‑T5 + Few‑Shot")

user_q = st.text_input(
    "Sua pergunta",
    placeholder="What was Accenture's total revenue in 2024?",
    key="user_q",
)

k = st.slider("Top‑k chunks", min_value=2, max_value=8, value=3, step=1, key="k_slider")

model_choice = st.selectbox(
    "Modelo",
    options=["google/flan-t5-small", "google/flan-t5-base"],
    index=1,
    help="Use base para melhor qualidade; small para velocidade.",
    key="model_choice",
)

llm = load_llm(model_choice)

if st.button("Perguntar", key="ask_btn"):
    if not user_q.strip():
        st.warning("Por favor, digite uma pergunta.")
    else:
        import time
        t0 = time.time()
        try:
            with st.spinner("Buscando e gerando resposta..."):
                docs = get_docs(user_q, k=k)
                context = build_context(docs, max_chars=1600)
                final_prompt = prompt.format(context=context, question=user_q)
                answer = llm.invoke(final_prompt)
        except Exception as e:
            st.exception(e)
        else:
            elapsed = time.time() - t0
            st.subheader("Resposta")
            st.write(answer)
            st.caption(f"Tempo: {elapsed:.2f}s | k={k}")

            st.subheader("Fontes utilizadas")
            for i, d in enumerate(docs[:k], 1):
                src = d.metadata.get("source", "")
                page = d.metadata.get("page", "?")
                with st.expander(f"Fonte {i}: {src} p.{page}"):
                    st.write(d.page_content)
