# 📄 Dados do Projeto

Esta pasta deve conter os PDFs que serão processados pelo sistema RAG.

## 📥 Como obter o PDF de exemplo

Para testar o sistema com o relatório da Accenture usado no projeto:

1. **Baixar o relatório anual da Accenture 2024:**
   - Acesse: https://www.accenture.com/us-en/about/company/annual-report
   - Baixe o "Fiscal 2024 Annual Report" (Form 10-K)
   - Ou busque por: "Accenture Fiscal 2024 Annual Report PDF"

2. **Renomear o arquivo:**
   - Renomeie para: `accenture-fiscal-2024-annual-report.pdf`
   - Coloque o arquivo nesta pasta (`Data/`)

## 📋 Estrutura esperada

```
Data/
├── README.md (este arquivo)
└── accenture-fiscal-2024-annual-report.pdf
```

## ⚠️ Importante

- O PDF deve estar nesta pasta **antes** de executar `python embedding.py`
- O nome do arquivo deve corresponder ao que está configurado em `embedding.py` e `ingestao_pdf.py`
- Você pode usar qualquer PDF, mas precisará atualizar o caminho nos scripts se usar outro nome

## 🔄 Usando seu próprio PDF

Se quiser usar um PDF diferente:

1. Coloque o PDF nesta pasta
2. Atualize o caminho em:
   - `ingestao_pdf.py` (linha 3)
   - `chunking.py` (linha 4)
   - `embedding.py` (linha 12)

