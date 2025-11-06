🚀 RAG – Financial Reports QA (LangChain + HuggingFace + FAISS)

Sistema de inteligência artificial para consulta factual de relatórios anuais corporativos usando Retrieval-Augmented Generation (RAG).

> 💡 **Novo no projeto?** Veja a seção [Instalação Rápida](#-instalação-rápida-primeira-vez) abaixo para começar em minutos!

## 🎯 Objetivo de Negócio

**Problema:** Consultas manuais em relatórios extensos (100+ páginas) são lentas e sujeitas a erros.

**Solução:** Pipeline RAG que indexa PDFs, recupera os trechos mais relevantes e gera respostas objetivas e auditáveis com base no contexto recuperado.

## 📊 Destaques

- **Revenue 2024** → $64.9 billion
- **R&D spend 2024** → $1.15 billion
- **Markets** → North America, EMEA (Europe, Middle East and Africa) and Growth Markets
- **Risk taxonomy** → Business, Financial, Operational, Legal & Regulatory

## 🧠 Arquitetura Técnica

### Pipeline de Dados
```
PDFs → Ingestão → Chunking → Embeddings → FAISS → Retriever → LLM (Few-Shot) → Resposta
```

### Componentes

1. **Ingestão** (`ingestao_pdf.py`): Extração de texto de PDFs usando PyPDF via LangChain Community
2. **Chunking** (`chunking.py`): Divisão de texto em chunks com RecursiveCharacterTextSplitter (1000 caracteres, overlap 200)
3. **Embeddings** (`embedding.py`): Geração de embeddings com HuggingFace all-MiniLM-L6-v2 e armazenamento em FAISS
4. **Busca Manual** (`busca_manual.py`): Consultas diretas ao índice vetorial para testes
5. **QA Chain** (`qa_chain.py`): Pipeline completo de retrieval e generation com MMR e Few-Shot prompting
6. **Interface Streamlit** (`app_user.py`): Aplicação web interativa com scoring customizado, highlighting e post-processing

## 🛠️ Stack Tecnológica

- **Python** 3.13+
- **LangChain** 0.2+ (LCEL, retrievers, vectorstores)
- **Embeddings:** HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store:** FAISS (CPU)
- **LLM:** Google Flan-T5-base (offline, via transformers)
- **UI:** Streamlit 1.38+

## 🏗️ Estrutura do Projeto

```
rag/
├── Data/                      # PDFs de entrada
├── ingestao_pdf.py            # Extração de texto
├── chunking.py                # Divisão em chunks
├── embedding.py               # Geração de embeddings e índice FAISS
├── busca_manual.py            # Consultas manuais ao índice
├── qa_chain.py               # Pipeline RAG completo (linha de comando)
├── app_user.py               # Interface Streamlit (aplicação principal)
├── app.py                    # Interface Streamlit alternativa
├── requirements.txt          # Dependências do projeto
├── vectorstore_index/        # Índice FAISS persistido
├── docs/reports/             # Relatórios Markdown gerados automaticamente
└── logs/                     # Logs de execução
```

## ⚙️ Configurações Técnicas

### Chunking
- `chunk_size=1000` caracteres
- `chunk_overlap=200` caracteres

### Retrieval
- **MMR** (qa_chain.py): `k=3`, `fetch_k=25`, `lambda_mult=0.5`
- **Similarity Search** (app_user.py): Scoring customizado baseado em keywords, números e sinônimos

### Context Window
- Streamlit: `max_chars=2200`
- Linha de comando: `max_chars=1800`

### Few-Shot Prompting
- Streamlit: 15 exemplos cobrindo múltiplos cenários (revenue, income, R&D, risks, markets, employees, ESG)
- Linha de comando: 3 exemplos básicos

## 🚀 Como Executar

### 📥 Instalação Rápida (Primeira Vez)

**1. Clonar o repositório:**
```bash
git clone https://github.com/Fergab1/rag-complete.git
cd rag-complete
```

**2. Criar e ativar ambiente virtual:**
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**3. Instalar dependências:**
```bash
pip install -r requirements.txt
```
⚠️ **IMPORTANTE:** Certifique-se de que o ambiente virtual está ativado antes de instalar. Se der erro de módulo não encontrado, verifique se o `(venv)` aparece no início do prompt do terminal.

**4. Adicionar PDF na pasta Data (OBRIGATÓRIO):**
- ⚠️ **CRÍTICO:** O PDF deve estar na pasta `Data/` ANTES de executar o próximo passo!
- A pasta `Data/` está vazia por padrão (não commitamos PDFs grandes)
- **Opção A - Usar o PDF da Accenture (recomendado para teste):**
  - Baixe o "Accenture Fiscal 2024 Annual Report" em: https://www.accenture.com/us-en/about/company/annual-report
  - Renomeie para `accenture-fiscal-2024-annual-report.pdf` (nome exato!)
  - Coloque na pasta `Data/` (crie a pasta se não existir)
  - Verifique se o arquivo está lá: `Data/accenture-fiscal-2024-annual-report.pdf`
- **Opção B - Usar seu próprio PDF:**
  - Coloque seu PDF na pasta `Data/`
  - Atualize o caminho em `embedding.py` (linha 12) se necessário
- 📁 **Estrutura esperada:** `rag-complete/Data/accenture-fiscal-2024-annual-report.pdf`

**5. Criar o índice FAISS:**
```bash
python embedding.py
```
⏱️ Isso pode levar alguns minutos na primeira vez (baixa o modelo de embeddings)

**6. Executar a aplicação:**
```bash
streamlit run app_user.py
```

O navegador abrirá automaticamente em `http://localhost:8501`

---

### 📋 Passo a Passo Detalhado

#### 1. Configuração do Ambiente

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows (PowerShell)
.\venv\Scripts\Activate
# Linux/Mac
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

#### 2. Preparação dos Dados

```bash
# Adicionar PDF(s) na pasta Data/
# Exemplo: Data/accenture-fiscal-2024-annual-report.pdf
```

#### 3. Criação do Índice

```bash
# Gerar embeddings e criar índice FAISS
python embedding.py
```

#### 4. Execução

**Opção A: Interface Streamlit (Recomendado)**
```bash
streamlit run app_user.py
```

**Opção B: Linha de Comando**
```bash
# Busca manual (testes)
python busca_manual.py

# Pipeline completo (gera relatórios em docs/reports/)
python qa_chain.py
```

## ✨ Features do Streamlit

- **Scoring Customizado:** Sistema de pontuação para melhorar a relevância dos chunks recuperados
- **Highlighting Visual:** Destaque automático de números, termos financeiros e palavras-chave nas sources
- **Post-processing:** Tratamento inteligente de respostas para melhorar formatação e precisão
- **Filtragem de Sources:** Seleção automática das fontes mais relevantes para cada resposta
- **Interface Interativa:** Consultas em tempo real com feedback visual

## 📄 Relatórios e Evidências

Após executar `qa_chain.py`, são gerados automaticamente:
- Arquivos Markdown em `docs/reports/<timestamp>/` com fontes, contexto e resposta para cada query
- Índice consolidado em `index.md`
- Logs completos em `logs/qa_latest.txt`

## 🔎 Exemplos de Queries

- What was Accenture's total revenue in 2024?
- How much did the company spend on research and development in 2024?
- In which markets does Accenture operate?
- What are the main risks described?
- How many employees does Accenture have?
- What sustainability or ESG actions did Accenture take?

## 🧪 Prompting Strategy

O sistema utiliza **Few-Shot Prompting** para guiar o LLM:
- Respostas concisas e factuais
- Extração de números quando disponíveis
- Resposta "I don't know" quando a informação não está no contexto
- Formatação consistente de valores monetários

Implementado com `FewShotPromptTemplate` do LangChain, com exemplos específicos para diferentes tipos de consultas.

## 📌 Boas Práticas

- Consultas no mesmo idioma dos documentos (inglês recomendado)
- Ajustar `k` (número de chunks) conforme o tamanho do documento
- Usar MMR para reduzir redundância nos trechos recuperados
- Validar respostas consultando as sources fornecidas

## ⚠️ Limitações Conhecidas

- Modelos de tamanho médio (Flan-T5-base) podem perder detalhes finos em respostas muito complexas
- FAISS local é adequado para projetos de pequena/média escala; para grande escala, considerar Qdrant, Weaviate ou PGVector
- Fontes são exibidas de forma expandida no Streamlit para transparência total

## 📚 Requisitos do Sistema

- Python 3.13+
- 4-8GB RAM recomendado (dependendo do modelo LLM escolhido)
- Espaço em disco: ~2GB para dependências e modelos

## 📬 Sobre o Projeto

Projeto desenvolvido para portfólio com foco em aplicações corporativas e análise de relatórios financeiros. Demonstra implementação completa de um pipeline RAG do zero, incluindo ingestão, processamento, indexação e geração de respostas.
