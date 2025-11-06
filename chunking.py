from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = 'Data/accenture-fiscal-2024-annual-report.pdf'
loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(docs)

print(f"Total de chunks criados: {len(chunks)}")
print("Exemplo de chunk:")
print(chunks[0].page_content)