from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore_index", embeddings, allow_dangerous_deserialization=True)

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

k = 5

for question in queries:
    print("\n" + "="*70)
    print(f"Query: {question}")
    print("="*70)
    results = vectorstore.similarity_search(question, k=k)
    for i, doc in enumerate(results, 1):
        print(f"\n[Chunk {i}] {'-'*40}")
        print(doc.page_content[:1200], "\n...")
    print("\n" + "-"*70)

print('\nBusca de todas as queries concluída!')