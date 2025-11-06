from langchain_community.document_loaders import PyPDFLoader

pdf_path = 'Data/accenture-fiscal-2024-annual-report.pdf'
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Total de páginas lidas: {len(docs)}")
print("Primeira página:")
print(docs[0].page_content[:1000])