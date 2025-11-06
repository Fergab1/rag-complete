import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

script_dir = Path(__file__).parent
pdf_path = script_dir / 'Data' / 'accenture-fiscal-2024-annual-report.pdf'
pdf_path_str = str(pdf_path)

if not pdf_path.exists():
    print("❌ ERRO: Arquivo PDF não encontrado!")
    print(f"   Caminho esperado: {pdf_path_str}")
    print(f"   Diretório atual: {os.getcwd()}")
    print(f"   Diretório do script: {script_dir}")
    
    data_dir = script_dir / 'Data'
    if data_dir.exists():
        print(f"\n📁 Arquivos encontrados na pasta Data/:")
        files = list(data_dir.glob('*.pdf'))
        if files:
            for f in files:
                print(f"   - {f.name}")
        else:
            print("   (nenhum arquivo PDF encontrado)")
    else:
        print(f"\n⚠️  A pasta Data/ não existe em: {data_dir}")
        print("   Criando pasta Data/...")
        data_dir.mkdir(exist_ok=True)
    
    print("\n📋 Para resolver:")
    print("   1. Baixe o PDF da Accenture em: https://www.accenture.com/us-en/about/company/annual-report")
    print("   2. Renomeie para: accenture-fiscal-2024-annual-report.pdf")
    print(f"   3. Coloque na pasta: {data_dir}")
    print("\n   Ou veja Data/README.md para mais instruções.")
    exit(1)

loader = PyPDFLoader(pdf_path_str)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(docs)

vstore = FAISS.from_documents(chunks, embeddings)
vstore.save_local("vectorstore_index")

print("Embeddings criados e index salvo!")
print(f"Banco contém {vstore.index.ntotal} vetores.")