import streamlit as st

st.set_page_config(
    page_title="Financial Reports Q&A",
    page_icon="📊",
    layout="wide"
)

import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

@st.cache_resource(show_spinner=False)
def load_index():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "vectorstore_index", embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore

@st.cache_resource(show_spinner=False)
def load_llm():
    repo_id = "google/flan-t5-base"
    try:
        hf_model = AutoModelForSeq2SeqLM.from_pretrained(
            repo_id,
            low_cpu_mem_usage=True,
            torch_dtype="float32"
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(repo_id)
        hf_pipeline = pipeline(
            "text2text-generation",
            model=hf_model,
            tokenizer=hf_tokenizer,
            max_new_tokens=128,
            do_sample=False,
        )
        return HuggingFacePipeline(pipeline=hf_pipeline)
    except OSError as e:
        if "1455" in str(e) or "paginação" in str(e).lower() or "paging" in str(e).lower():
            return "MEMORY_ERROR"
        raise
    except Exception as e:
        return f"ERROR: {str(e)}"

def translate_query_to_english(query):
    pt_to_en = {
        'quantos': 'how many',
        'quantas': 'how many',
        'funcionarios': 'employees',
        'funcionários': 'employees',
        'funcionario': 'employee',
        'funcionário': 'employee',
        'tem': 'has',
        'receita': 'revenue',
        'receitas': 'revenues',
        'lucro': 'income',
        'lucro líquido': 'net income',
        'renda': 'income',
        'faturamento': 'revenue',
        'vendas': 'sales',
        'mercados': 'markets',
        'mercado': 'market',
        'opera': 'operates',
        'opera em': 'operates in',
        'riscos': 'risks',
        'risco': 'risk',
        'clientes': 'clients',
        'cliente': 'client',
        'parceiros': 'partners',
        'parceiro': 'partner',
        'parcerias': 'partnerships',
        'parceria': 'partnership',
        'segmentos': 'segments',
        'segmento': 'segment',
        'gastou': 'spent',
        'gastou em': 'spent on',
        'pesquisa': 'research',
        'desenvolvimento': 'development',
        'pesquisa e desenvolvimento': 'research and development',
        'sustentabilidade': 'sustainability',
        'esg': 'esg',
        'diversidade': 'diversity',
        'inclusao': 'inclusion',
        'inclusão': 'inclusion',
        'inovações': 'innovations',
        'inovação': 'innovation',
        'iniciativas': 'initiatives',
        'iniciativa': 'initiative',
        'programas': 'programs',
        'programa': 'program',
        'desafios': 'challenges',
        'desafio': 'challenge',
        'regulatórios': 'regulatory',
        'regulatório': 'regulatory',
        'conformidade': 'compliance',
        'cibersegurança': 'cybersecurity',
        'ciberseguranca': 'cybersecurity',
        'foi': 'was',
        'foram': 'were',
        'é': 'is',
        'são': 'are',
        'qual': 'what',
        'quais': 'which',
        'onde': 'where',
        'quando': 'when',
        'quem': 'who',
        'como': 'how',
        'quanto': 'how much',
        'a accenture': 'accenture',
        'da accenture': 'accenture',
        'da empresa': 'company',
        'a empresa': 'company',
    }
    
    query_lower = query.lower()
    translated = query
    
    for pt_term, en_term in sorted(pt_to_en.items(), key=lambda x: len(x[0]), reverse=True):
        if pt_term in query_lower:
            translated = translated.replace(pt_term, en_term)
            translated = translated.replace(pt_term.capitalize(), en_term.capitalize())
    
    return translated.strip()

def extract_answer_parts(answer):
    parts = []
    answer_clean = answer.strip()
    
    if not answer_clean or len(answer_clean) < 10:
        return parts
    
    answer_lower = answer_clean.lower()
    
    if 'i don\'t know' in answer_lower or 'não sei' in answer_lower:
        return parts
    
    skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'o', 'a', 'os', 'as', 'de', 'da', 'do', 'das', 'dos', 'em', 'na', 'no', 'nas', 'nos', 'e', 'ou', 'we', 'our', 'they', 'their', 'this', 'that', 'these', 'those', 'which', 'what', 'who', 'where', 'when', 'how', 'nós', 'nosso', 'nossa', 'eles', 'elas', 'este', 'esta', 'esse', 'essa'}
    
    if ',' in answer_clean or ' and ' in answer_lower or ' e ' in answer_lower:
        separators = [',', ' and ', ' e ', ';']
        for sep in separators:
            if sep in answer_clean:
                split_parts = answer_clean.split(sep)
                for part in split_parts:
                    part = part.strip()
                    part = re.sub(r'^[^\w\s]*', '', part)
                    part = re.sub(r'[^\w\s]*$', '', part)
                    if 8 <= len(part) <= 80 and not part.lower().startswith('i don'):
                        parts.append(part)
                if parts:
                    break
    
    if not parts:
        sentences = re.split(r'[.!?]\s+', answer_clean)
        for sentence in sentences:
            sentence = sentence.strip()
            if 15 <= len(sentence) <= 150 and not sentence.lower().startswith('i don'):
                words = sentence.split()
                if len(words) >= 3:
                    key_phrases = []
                    for i in range(len(words) - 2):
                        phrase = ' '.join(words[i:i+4])
                        if len(phrase) >= 8 and len(phrase) <= 60:
                            key_phrases.append(phrase)
                    if key_phrases:
                        parts.extend(key_phrases[:3])
    
    if not parts:
        words = answer_clean.split()
        important_words = []
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word)
            if len(word_clean) > 5 and word_clean.lower() not in skip_words:
                important_words.append(word_clean)
        if important_words:
            parts.extend(important_words[:8])
    
    return parts[:12]

def build_context(docs, max_chars=3000):
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

examples = [
    {
        "context": "We delivered revenues of $65 billion for the year, representing 2% growth. Total Revenues $ 64.9 billion for fiscal 2024.",
        "question": "What was Accenture's total revenue in 2024?",
        "answer": "$64.9 billion (or $65 billion)."
    },
    {
        "context": "NET INCOME ATTRIBUTABLE TO ACCENTURE PLC $ 7,264,787 $ 6,871,557 $ 6,877,169",
        "question": "What is the net income reported by Accenture?",
        "answer": "$7.3 billion for fiscal 2024."
    },
    {
        "context": "Research and development costs were $1,150,430 thousand in fiscal 2024.",
        "question": "How much did the company spend on research and development in 2024?",
        "answer": "$1.15 billion (or $1,150,430 thousand)."
    },
    {
        "context": "Key risks include supply chain disruption, cybersecurity threats, and regulatory compliance challenges.",
        "question": "What are the main risks described?",
        "answer": "Supply chain disruption, cybersecurity threats, and regulatory compliance challenges."
    },
    {
        "context": "We serve clients in three geographic markets: North America, EMEA (Europe, Middle East and Africa), and Growth Markets.",
        "question": "In which markets does Accenture operate?",
        "answer": "North America, EMEA (Europe, Middle East and Africa), and Growth Markets."
    },
    {
        "context": "Accenture is a leading global professional services company, providing services across Strategy & Consulting, Technology, Operations, Industry X and Song.",
        "question": "How is the company's business segmented?",
        "answer": "Strategy & Consulting, Technology, Operations, Industry X and Song."
    },
    {
        "context": "Accenture has approximately 774,000 people serving clients in more than 120 countries.",
        "question": "How many employees does Accenture have?",
        "answer": "Approximately 774,000 employees."
    },
    {
        "context": "Accenture committed to achieving net-zero emissions by 2025. We invested $3.4 billion in sustainability initiatives and reduced our carbon footprint by 40%.",
        "question": "What sustainability or ESG actions did Accenture take?",
        "answer": "Committed to achieving net-zero emissions by 2025, invested $3.4 billion in sustainability initiatives, and reduced carbon footprint by 40%."
    },
    {
        "context": "Our diversity and inclusion programs include the Women's Leadership Program, the Black Leadership Program, and the Pride Network for LGBTQ+ employees.",
        "question": "What programs are in place for diversity and inclusion?",
        "answer": "Women's Leadership Program, Black Leadership Program, and Pride Network for LGBTQ+ employees."
    },
    {
        "context": "Accenture serves Fortune 500 companies across industries including financial services, healthcare, technology, and retail. Our top clients include major global corporations.",
        "question": "Who are Accenture's main clients?",
        "answer": "Fortune 500 companies across industries including financial services, healthcare, technology, and retail."
    },
    {
        "context": "Accenture has strategic partnerships with Microsoft, Salesforce, SAP, and AWS to deliver cloud transformation services.",
        "question": "What strategic partnerships are referenced?",
        "answer": "Microsoft, Salesforce, SAP, and AWS."
    },
    {
        "context": "This year, Accenture launched the AI Innovation Lab, the Quantum Computing Center, and the Metaverse Continuum business group.",
        "question": "Which innovation initiatives were developed this year?",
        "answer": "AI Innovation Lab, Quantum Computing Center, and Metaverse Continuum business group."
    },
    {
        "context": "Regulatory challenges include GDPR compliance in Europe, data privacy laws in various jurisdictions, and financial services regulations.",
        "question": "What regulatory or compliance challenges are mentioned?",
        "answer": "GDPR compliance in Europe, data privacy laws in various jurisdictions, and financial services regulations."
    },
    {
        "context": "Cybersecurity risks include data breaches, ransomware attacks, phishing attempts, and supply chain vulnerabilities.",
        "question": "Which cybersecurity risks does Accenture highlight?",
        "answer": "Data breaches, ransomware attacks, phishing attempts, and supply chain vulnerabilities."
    },
    {
        "context": "Revenue in 2023 was $61B. No data about 2024.",
        "question": "What was the revenue in 2024?",
        "answer": "I don't know."
    },
    {
        "context": "Accenture purchased Class A ordinary shares for $605 million. This was for share repurchases.",
        "question": "What was Accenture's total revenue in 2024?",
        "answer": "I don't know. The context mentions share purchases, not revenue."
    },
    {
        "context": "Accenture has approximately 774,000 people serving clients in more than 120 countries.",
        "question": "Quantos funcionários a Accenture tem?",
        "answer": "Aproximadamente 774.000 funcionários."
    },
    {
        "context": "Total Revenues $ 64.9 billion for fiscal 2024.",
        "question": "Qual foi a receita total da Accenture em 2024?",
        "answer": "$64.9 bilhões em 2024."
    },
    {
        "context": "Research and development costs were $1,150,430 thousand in fiscal 2024.",
        "question": "Quanto a empresa gastou em pesquisa e desenvolvimento em 2024?",
        "answer": "$1.15 bilhões (ou $1,150,430 mil) em 2024."
    },
    {
        "context": "We serve clients in three geographic markets: North America, EMEA (Europe, Middle East and Africa), and Growth Markets.",
        "question": "Em quais mercados a Accenture opera?",
        "answer": "América do Norte, EMEA (Europa, Oriente Médio e África) e Mercados de Crescimento."
    },
    {
        "context": "Accenture serves Fortune 500 companies across industries including financial services, healthcare, technology, and retail. Our top clients include major global corporations.",
        "question": "Quais são os clientes da Accenture?",
        "answer": "Empresas Fortune 500 em diversos setores, incluindo serviços financeiros, saúde, tecnologia e varejo."
    },
]
example_prompt = PromptTemplate.from_template(
    "Context:\n{context}\nQuestion:\n{question}\nAnswer:\n{answer}\n"
)
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
        prefix=(
            "You are a precise analyst answering questions about Accenture based ONLY on the context provided. "
            "IMPORTANT: Answer in the SAME LANGUAGE as the question. If the question is in Portuguese, answer in Portuguese. If in English, answer in English. "
            "Instructions: "
            "1. Extract ONLY the specific information requested. Do NOT copy entire tables, raw text, or provide definitions. "
            "2. Do NOT copy raw table data, sequences of numbers, or unformatted text. Always provide a clear, readable answer. "
            "3. If asked for a NUMBER, provide the NUMBER in a readable format (e.g., '$64.9 billion' or '$64.9 bilhões', not '64.9 64.11 10.8'). "
            "4. If asked for a LIST, provide the LIST as a comma-separated list or clear enumeration. Use the EXACT terms from the context when possible (e.g., if context says 'supply chain disruption', use 'supply chain disruption', not 'supply chain problems'). "
            "5. If asked for a DESCRIPTION, provide the DESCRIPTION in natural language. Use key terms and phrases from the context to maintain accuracy. "
            "6. For numbers: Convert appropriately (e.g., 7,264,787 thousand = $7.3 billion, 64,896 thousand = $64.9 billion). "
            "7. When multiple years are shown, use the most recent year (usually the first number in a sequence). "
            "8. Your answer must be a single, clear sentence or short paragraph. Do NOT include table formatting, raw numbers, percentages, or symbols. "
            "9. If the context contains ANY relevant information, extract and use it. Do NOT say 'I don't know' if the context has related information. "
            "10. If the context mentions the topic but doesn't have exact numbers, provide the information that IS available. "
            "11. Only reply 'I don't know' (or 'Não sei' in Portuguese) if the context is completely unrelated to the question. "
            "12. Format numbers with currency and appropriate units (e.g., '$64.9 billion' or '$64.9 bilhões'). "
            "13. IMPORTANT: When listing items or providing descriptions, use the EXACT terminology from the context when possible. This helps maintain precision and traceability. "
            "14. CRITICAL: If asked 'Who are...' or 'Quais são...' or 'What are...', provide a DIRECT, SPECIFIC answer. Do NOT give generic descriptions about the company's approach or philosophy. Extract the SPECIFIC entities, names, or categories mentioned in the context.\n\nExamples:\n"
        ),
    suffix="Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
    input_variables=["context", "question"],
)

st.title("📊 Accenture Q&A")
st.markdown("Ask any question about Accenture based on the 2024 Annual Report.")

st.divider()
user_q = st.text_input(
    "Your question",
    placeholder="Ask any question about Accenture...",
    key="user_q",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 6])
with col1:
    ask_btn = st.button("Ask", type="primary", key="ask_btn", use_container_width=True)

if ask_btn:
    if not user_q.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching and analyzing..."):
                if 'vectorstore_instance' not in st.session_state:
                    st.session_state.vectorstore_instance = load_index()
                if 'llm_instance' not in st.session_state:
                    with st.spinner("Loading AI model (first time only, please wait)..."):
                        st.session_state.llm_instance = load_llm()
                
                if st.session_state.llm_instance == "MEMORY_ERROR" or (isinstance(st.session_state.llm_instance, str) and st.session_state.llm_instance.startswith("ERROR")):
                    st.error("Erro de memória ao carregar o modelo")
                    st.warning("""
                    Soluções possíveis:
                    1. Feche outros programas pesados
                    2. Aumente o arquivo de paginação do Windows
                    3. Use um computador com mais RAM
                    """)
                    st.stop()
                
                query_lower = user_q.lower()
                all_docs = []
                seen = set()
                
                translated_query = translate_query_to_english(user_q)
                search_query = translated_query if translated_query != user_q else user_q
                
                docs1 = st.session_state.vectorstore_instance.similarity_search(search_query, k=20)
                for doc in docs1:
                    doc_id = id(doc.page_content)
                    if doc_id not in seen:
                        seen.add(doc_id)
                        all_docs.append(doc)
                
                stop_words = {'what', 'was', 'were', 'when', 'where', 'which', 'who', 'how', 'the', 'a', 'an', 
                             'is', 'are', 'does', 'do', 'did', 'accenture', 'accenture\'s', 'that', 'this', 'for', 'in', 'on', 'at', 'many', 'much', 'list', 'are',
                             'quantos', 'quantas', 'qual', 'quais', 'onde', 'quando', 'quem', 'como', 'quanto', 'a', 'o', 'os', 'as', 'da', 'de', 'do', 'das', 'dos', 'em', 'na', 'no', 'nas', 'nos'}
                key_words = [w.lower() for w in translated_query.split() if w.lower() not in stop_words and len(w) > 2]
                
                variations = []
                if len(key_words) >= 3:
                    variations.extend([
                        ' '.join(key_words[:3]),
                        ' '.join(key_words[:2]),
                        ' '.join(key_words[:4]) if len(key_words) >= 4 else ' '.join(key_words[:3]),
                    ])
                elif len(key_words) == 2:
                    variations.extend([
                        ' '.join(key_words),
                        key_words[0] + ' accenture',
                    ])
                elif len(key_words) == 1:
                    variations.append(key_words[0] + ' accenture')
                
                if '2024' in user_q:
                    for kw in key_words[:2]:
                        variations.append(f"{kw} 2024")
                
                for variation in variations[:6]:
                    docs_var = st.session_state.vectorstore_instance.similarity_search(variation, k=5)
                    for doc in docs_var:
                        doc_id = id(doc.page_content)
                        if doc_id not in seen:
                            seen.add(doc_id)
                            all_docs.append(doc)
                
                scoring_query = translated_query if translated_query != user_q else user_q
                query_words = set([w.lower() for w in scoring_query.split() if len(w) > 3])
                query_words_small = set([w.lower() for w in scoring_query.split() if len(w) > 2 and w.lower() not in stop_words])
                scored_docs = []
                
                for doc in all_docs:
                    content_lower = doc.page_content.lower()
                    content = doc.page_content
                    
                    score = 0
                    
                    keyword_matches = sum(1 for word in query_words if word in content_lower)
                    score += keyword_matches * 5
                    
                    small_keyword_matches = sum(1 for word in query_words_small if word in content_lower)
                    score += small_keyword_matches * 2
                    
                    important_terms = [w for w in query_words_small if w not in stop_words]
                    for term in important_terms:
                        if term in content_lower:
                            score += 4
                    
                    if any(char.isdigit() for char in user_q):
                        numbers = re.findall(r'\d{1,3}(?:,\d{3})*', content)
                        if numbers:
                            score += 5
                            large_numbers = [n for n in numbers if int(n.replace(',', '')) > 1000]
                            if large_numbers:
                                score += 3
                    
                    is_list_question = any(word in query_lower for word in ['list', 'what are', 'which', 'what programs', 'quais', 'quais são', 'who are', 'quem são'])
                    if is_list_question:
                        if ',' in content or ' and ' in content_lower or ';' in content:
                            score += 4
                    
                    if any(word in query_lower for word in ['client', 'cliente', 'customer', 'clients', 'clientes', 'customers']):
                        if any(term in content_lower for term in ['fortune', '500', 'companies', 'corporations', 'industries', 'financial services', 'healthcare', 'technology', 'retail', 'setores', 'empresas']):
                            score += 10
                    
                    if "2024" in user_q and "2024" in content:
                        score += 5
                    
                    synonym_groups = {
                        'employee': ['people', 'workforce', 'staff', 'personnel', 'funcionarios', 'funcionários', 'funcionario', 'funcionário'],
                        'revenue': ['sales', 'income', 'earnings', 'receita', 'receitas', 'faturamento'],
                        'risk': ['threat', 'challenge', 'concern', 'risco', 'riscos'],
                        'market': ['region', 'geography', 'country', 'mercado', 'mercados'],
                        'client': ['customer', 'customer base', 'cliente', 'clientes'],
                        'segment': ['business', 'service', 'division', 'segmento', 'segmentos'],
                        'funcionarios': ['employees', 'people', 'workforce', 'staff', 'personnel'],
                        'funcionários': ['employees', 'people', 'workforce', 'staff', 'personnel'],
                        'receita': ['revenue', 'sales', 'income'],
                        'receitas': ['revenues', 'sales', 'income'],
                        'mercado': ['market', 'region', 'geography'],
                        'mercados': ['markets', 'regions', 'geographies'],
                        'cliente': ['client', 'customer'],
                        'clientes': ['clients', 'customers'],
                        'risco': ['risk', 'threat', 'challenge'],
                        'riscos': ['risks', 'threats', 'challenges'],
                    }
                    for main_term, synonyms in synonym_groups.items():
                        if main_term in query_lower:
                            for synonym in synonyms:
                                if synonym in content_lower:
                                    score += 3
                    
                    content_terms = set([w for w in content_lower.split() if len(w) > 4])
                    query_terms = set([w for w in query_lower.split() if len(w) > 4])
                    if len(content_terms) > 0 and len(query_terms) > 0:
                        overlap = len(content_terms.intersection(query_terms))
                        overlap_ratio = overlap / len(query_terms) if len(query_terms) > 0 else 0
                        if overlap_ratio < 0.2:
                            score -= 3
                    
                    if len(content) < 100:
                        score -= 5
                    
                    if 200 <= len(content) <= 1500:
                        score += 2
                    
                    scored_docs.append((score, doc))
                
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                filtered = [(score, doc) for score, doc in scored_docs if score > 0]
                if filtered:
                    docs = [doc for _, doc in filtered[:7]]
                else:
                    docs = [doc for _, doc in scored_docs[:7]]
                context = build_context(docs, max_chars=2200)
                
                query_lower = user_q.lower()
                context_lower = context.lower()
                has_relevant_info = False
                
                query_keywords = [w for w in query_lower.split() if len(w) > 2 and w not in ['what', 'was', 'were', 'when', 'where', 'which', 'who', 'how', 'accenture', 'the', 'a', 'an', 'is', 'are', 'does', 'do', 'did', 'quantos', 'quantas', 'qual', 'quais', 'onde', 'quando', 'quem', 'como', 'quanto']]
                keyword_matches = sum(1 for keyword in query_keywords if keyword in context_lower)
                
                has_numbers = bool(re.search(r'\d{1,3}(?:,\d{3})*', context))
                
                has_synonyms = False
                synonym_groups = {
                        'employee': ['people', 'workforce', 'staff', 'personnel', 'funcionarios', 'funcionários'],
                        'revenue': ['sales', 'income', 'earnings', 'receita', 'receitas'],
                        'risk': ['threat', 'challenge', 'concern', 'risco', 'riscos'],
                        'market': ['region', 'geography', 'country', 'mercado', 'mercados'],
                        'client': ['customer', 'customer base', 'cliente', 'clientes'],
                        'segment': ['business', 'service', 'division', 'segmento', 'segmentos'],
                        'funcionarios': ['employees', 'people', 'workforce'],
                        'funcionários': ['employees', 'people', 'workforce'],
                        'receita': ['revenue', 'sales', 'income'],
                        'mercado': ['market', 'region'],
                        'cliente': ['client', 'customer'],
                        'risco': ['risk', 'threat'],
                    }
                for main_term, synonyms in synonym_groups.items():
                    if main_term in query_lower:
                        for synonym in synonyms:
                            if synonym in context_lower:
                                has_synonyms = True
                                break
                
                if keyword_matches >= 1 or (has_numbers and any(char.isdigit() for char in user_q)) or has_synonyms:
                    has_relevant_info = True
                
                final_prompt = prompt.format(context=context, question=user_q)
                
                if has_relevant_info:
                    extra_instruction = "\n\nThe context below contains relevant information. Extract and provide it directly. Do NOT say 'I don't know' if information exists in the context.\n\n"
                    final_prompt = final_prompt.replace(
                        "Context:\n",
                        f"Context:\n{extra_instruction}"
                    )
                
                if any(word in user_q.lower() for word in ['who are', 'quais são', 'quem são', 'what are', 'who is', 'quem é']):
                    specific_instruction = "\n\n⚠️ CRITICAL: This is a 'WHO/WHAT ARE' question. Provide a DIRECT, SPECIFIC answer with names, entities, or categories. Do NOT give generic descriptions or company philosophy. Extract the SPECIFIC information requested.\n\n"
                    final_prompt = final_prompt.replace(
                        "Context:\n",
                        f"Context:\n{specific_instruction}"
                    )
                
                answer = st.session_state.llm_instance.invoke(final_prompt)
                
                answer_clean = answer.strip()
                answer_lower = answer_clean.lower()
                
                says_dont_know = False
                
                words = answer_clean.split()
                meaningful_words = [w for w in words if len(re.sub(r'[^\w]', '', w)) > 2 and not w.replace('.', '').replace(',', '').replace('$', '').replace('%', '').isdigit()]
                numbers_and_symbols = [w for w in words if re.match(r'^[\d\s\$\%\(\)\.\,]+$', w) or w.replace('.', '').replace(',', '').replace('$', '').replace('%', '').isdigit()]
                
                is_table_copy = len(numbers_and_symbols) > len(meaningful_words) * 2 and len(meaningful_words) < 3
                
                has_repetitive_numbers = len(re.findall(r'\d+\.?\d*', answer_clean)) > 5
                has_table_pattern = bool(re.search(r'\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*', answer_clean))
                
                if is_table_copy or (has_repetitive_numbers and has_table_pattern):
                    answer_clean = ""
                    answer = ""
                    says_dont_know = True
                
                instruction_phrases = [
                    "the context contains relevant information",
                    "extract and provide it",
                    "do not say i don't know",
                    "important:"
                ]
                for phrase in instruction_phrases:
                    if phrase in answer_lower:
                        if len(answer_clean.split()) < 15:
                            answer_clean = ""
                            answer = ""
                            says_dont_know = True
                
                if "i don't know" in answer_lower or "don't know" in answer_lower:
                    says_dont_know = True
                
                if says_dont_know and len(answer_clean) > 50:
                    parts = re.split(r'i don\'t know[.,]?\s*', answer_clean, flags=re.IGNORECASE)
                    if len(parts) > 1 and len(parts[1].strip()) > 20:
                        answer_clean = parts[1].strip()
                        answer = answer_clean
                        says_dont_know = False
                
                if not answer_clean or len(answer_clean) < 10:
                    says_dont_know = True
                
                answer_has_small_number = re.search(r'\$\s*(\d{1,3}(?:,\d{3})*)\s*(?:thousand|hundred)', answer_clean, re.IGNORECASE)
                answer_is_generic = any(word in answer_lower for word in ["represents", "is defined", "refers to", "means", "is the", "is a"])
                
                if says_dont_know or answer_has_small_number or answer_is_generic:
                    stop_words = {'what', 'was', 'were', 'when', 'where', 'which', 'who', 'how', 'accenture', 
                                 'the', 'that', 'this', 'is', 'are', 'does', 'do', 'did', 'for', 'in', 'on', 'at'}
                    key_words = [w for w in query_lower.split() if w not in stop_words and len(w) > 3]
                    
                    patterns_to_try = []
                    
                    if key_words:
                        key_pattern = '|'.join(key_words[:3])
                        patterns_to_try.append(
                            rf'({key_pattern}.*?\$?\s*(\d{{1,3}}(?:,\d{{3}})*))'
                        )
                        patterns_to_try.append(
                            rf'({key_pattern}.*?(\d{{1,3}}(?:,\d{{3}})*\s*(?:billion|million|thousand)))'
                        )
                        patterns_to_try.append(
                            rf'((?:TOTAL|NET|GROSS|OPERATING)\s+{key_pattern}.*?\$?\s*(\d{{1,3}}(?:,\d{{3}})*))'
                        )
                    
                    if 'revenue' in query_lower:
                        patterns_to_try.extend([
                            r'(revenue[s]?.*?\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million|thousand))',
                            r'(total\s+revenue[s]?.*?\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?))',
                            r'(\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million).*?revenue)',
                        ])
                    
                    if any(char.isdigit() for char in user_q):
                        patterns_to_try.append(
                            r'(\b\w+\s+\$?\s*(\d{1,3}(?:,\d{3})*)\s*(?:billion|million|thousand))'
                        )
                    
                    found_answer = False
                    
                    for pattern in patterns_to_try:
                        matches = re.finditer(pattern, context, re.IGNORECASE)
                        for match in matches:
                            numbers_in_match = re.findall(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', match.group(0))
                            if numbers_in_match:
                                nums = []
                                for n in numbers_in_match:
                                    try:
                                        clean_num = n.replace(',', '')
                                        nums.append(float(clean_num))
                                    except (ValueError, AttributeError):
                                        continue
                                
                                if nums:
                                    candidate_nums = [n for n in nums if n > 10]
                                    if candidate_nums:
                                        num = max(candidate_nums)
                                        
                                        if 10 <= num <= 1000 and 'revenue' in query_lower:
                                            answer = f"${num:.1f} billion"
                                            found_answer = True
                                            break
                                        elif num > 1000000:
                                            billions = num / 1000000.0
                                            answer = f"${billions:.1f} billion"
                                            found_answer = True
                                            break
                                        elif num > 1000:
                                            billions = num / 1000.0
                                            if billions > 10:
                                                answer = f"${billions:.1f} billion"
                                            else:
                                                answer = f"${billions:.1f} million"
                                            found_answer = True
                                            break
                        
                        if found_answer:
                            break
                    
                    if says_dont_know and not found_answer:
                        sentences = re.split(r'[.!?]\s+', context)
                        relevant_sentences = []
                        query_words_set = set([w.lower() for w in user_q.split() if len(w) > 3])
                        
                        for sentence in sentences:
                            sentence_lower = sentence.lower()
                            matches = sum(1 for word in query_words_set if word in sentence_lower)
                            if matches >= 1 and len(sentence) > 20:
                                has_number = bool(re.search(r'\d', sentence))
                                relevant_sentences.append((matches + (3 if has_number else 0), sentence.strip()))
                        
                        if relevant_sentences:
                            relevant_sentences.sort(key=lambda x: x[0], reverse=True)
                            best_sentence = relevant_sentences[0][1]
                            answer = best_sentence[:300] + "..." if len(best_sentence) > 300 else best_sentence
                            answer = re.sub(r'^i don\'t know[.,]?\s*', '', answer, flags=re.IGNORECASE)
                            answer = re.sub(r'^don\'t know[.,]?\s*', '', answer, flags=re.IGNORECASE)
                            answer = re.sub(r'the context mentions?\s*', '', answer, flags=re.IGNORECASE)
                            answer = answer.strip()
                            if answer:
                                answer = answer[0].upper() + answer[1:] if len(answer) > 1 else answer.upper()
                
                elif len(answer_clean.split()) > 10 and re.search(r'\d', answer_clean):
                    numbers = re.findall(r'(\d{1,3}(?:,\d{3})*)', answer_clean)
                    if numbers:
                        largest_num = max([int(n.replace(',', '')) for n in numbers])
                        if largest_num > 1000000:
                            billions = largest_num / 1000000.0
                            answer = f"${billions:.1f} billion"
                        elif largest_num > 1000:
                            millions = largest_num / 1000.0
                            answer = f"${millions:.1f} million"
            
            st.divider()
            st.markdown("### Answer")
            st.info(answer)
            
            with st.expander("📎 Sources", expanded=False):
                answer_clean = answer.strip()
                answer_numbers = re.findall(r'(\d+\.?\d*)', answer_clean)
                answer_lower = answer_clean.lower()
                query_lower = user_q.lower()
                
                answer_parts = extract_answer_parts(answer_clean)
                
                answer_value_full = None
                dollar_match = re.search(r'\$\s*([\d,\.]+)\s*(billion|million|thousand)', answer_clean, re.IGNORECASE)
                if dollar_match:
                    answer_value_full = dollar_match.group(0)
                
                relevant_docs = []
                doc_scores = []
                
                for d in docs:
                    content = d.page_content.strip()
                    content_lower = content.lower()
                    
                    if len(content) < 50:
                        continue
                    
                    score = 0
                    
                    query_words = set(user_q.lower().split())
                    content_words = set(content_lower.split())
                    matches = len(query_words.intersection(content_words))
                    score += matches * 2
                    
                    if answer_numbers:
                        for num in answer_numbers:
                            num_clean = num.replace('.', '')
                            patterns = [
                                rf'\b{re.escape(num)}\b',
                                rf'\b{num_clean}\b',
                                rf'\$\s*{re.escape(num)}',
                                rf'{re.escape(num)}\s*(?:billion|million|thousand)',
                            ]
                            for pattern in patterns:
                                if re.search(pattern, content, re.IGNORECASE):
                                    score += 20
                                    break
                    
                    if 'revenue' in answer_lower or 'revenue' in user_q.lower():
                        if 'revenue' in content_lower:
                            score += 10
                    if 'income' in answer_lower or 'income' in user_q.lower():
                        if 'income' in content_lower:
                            score += 10
                    if 'employee' in answer_lower or 'employee' in user_q.lower():
                        if any(term in content_lower for term in ['employee', 'people', 'workforce']):
                            score += 10
                    
                    if score == 0:
                        continue
                    
                    doc_scores.append((score, d))
                
                doc_scores.sort(key=lambda x: x[0], reverse=True)
                relevant_docs = [doc for _, doc in doc_scores[:3]]
                
                if not relevant_docs:
                    relevant_docs = docs[:3]
                
                for i, d in enumerate(relevant_docs, 1):
                    page = d.metadata.get("page", "?")
                    content = d.page_content.strip()
                    
                    lines = content.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        line = line.strip()
                        if len(line) >= 5 and not (line.startswith('(') and len(line) < 20):
                            cleaned_lines.append(line)
                    
                    cleaned_content = ' '.join(cleaned_lines)
                    if len(cleaned_content) > 1000:
                        cut_point = cleaned_content[:1000].rfind('.')
                        if cut_point < 700:
                            cut_point = cleaned_content[:1000].rfind(' ')
                        if cut_point > 500:
                            cleaned_content = cleaned_content[:cut_point] + "..."
                        else:
                            cleaned_content = cleaned_content[:1000] + "..."
                    
                    highlighted_content = cleaned_content
                    
                    if answer_parts:
                        for part in answer_parts:
                            if len(part) > 5:
                                part_escaped = re.escape(part)
                                
                                patterns_part = [
                                    rf'\b{part_escaped}\b',
                                    rf'{part_escaped}',
                                ]
                                
                                found_match = False
                                for pattern in patterns_part:
                                    matches = list(re.finditer(pattern, highlighted_content, re.IGNORECASE))
                                    if matches:
                                        for match in reversed(matches):
                                            matched_text = match.group(0)
                                            if not re.search(rf'<strong[^>]*>{re.escape(matched_text)}', highlighted_content, re.IGNORECASE):
                                                start, end = match.span()
                                                before = highlighted_content[:start]
                                                after = highlighted_content[end:]
                                                highlighted_content = before + f'<strong style="background-color: #ffc107; padding: 3px 5px; border-radius: 3px; font-weight: bold;">{matched_text}</strong>' + after
                                                found_match = True
                                        if found_match:
                                            break
                                
                                if not found_match and len(part.split()) <= 3:
                                    words_in_part = part.split()
                                    for word in words_in_part:
                                        word_clean = re.sub(r'[^\w]', '', word)
                                        if len(word_clean) > 5:
                                            word_pattern = rf'\b{re.escape(word_clean)}\b'
                                            if re.search(word_pattern, highlighted_content, re.IGNORECASE):
                                                if not re.search(rf'<strong[^>]*>{re.escape(word_clean)}', highlighted_content, re.IGNORECASE):
                                                    highlighted_content = re.sub(
                                                        word_pattern,
                                                        r'<strong style="background-color: #ffc107; padding: 3px 5px; border-radius: 3px; font-weight: bold;">\g<0></strong>',
                                                        highlighted_content,
                                                        flags=re.IGNORECASE
                                                    )
                    
                    if answer_value_full:
                        answer_value_escaped = re.escape(answer_value_full)
                        if not re.search(rf'<strong[^>]*>{answer_value_escaped}', highlighted_content, re.IGNORECASE):
                            highlighted_content = re.sub(
                                answer_value_escaped,
                                r'<strong style="background-color: #ffc107; padding: 3px 5px; border-radius: 3px; font-weight: bold;">\g<0></strong>',
                                highlighted_content,
                                flags=re.IGNORECASE
                            )
                    
                    if answer_numbers:
                        for num in answer_numbers:
                            num_clean = num.replace('.', '')
                            num_with_dot = num
                            
                            try:
                                num_float = float(num_with_dot)
                                num_int = int(num_float)
                            except (ValueError, TypeError):
                                num_int = None
                            
                            patterns_exact = [
                                (rf'Total\s+Revenues?\s*\$?\s*{re.escape(num_with_dot)}', 'exact'),
                                (rf'Total\s+Revenues?\s*\$?\s*{re.escape(num_clean)}', 'exact'),
                                (rf'\$\s*{re.escape(num_with_dot)}\s*(?:billion|million|thousand)', 'exact'),
                                (rf'\$\s*{re.escape(num_with_dot)}\b', 'exact'),
                            ]
                            
                            if num_int:
                                patterns_exact.extend([
                                    (rf'Total\s+Revenues?\s*\$?\s*{num_int}\.?\d*', 'exact'),
                                    (rf'\$\s*{num_int}\.?\d*\s*(?:billion|million|thousand)', 'exact'),
                                ])
                            
                            if 'revenue' in query_lower or 'revenue' in answer_lower:
                                patterns_exact.insert(0, (rf'Total\s+Revenues?\s*\$?\s*[\d,\.]+', 'exact'))
                            
                            for pattern, priority in patterns_exact:
                                matches = list(re.finditer(pattern, highlighted_content, re.IGNORECASE))
                                if matches:
                                    for match in reversed(matches):
                                        matched_text = match.group(0)
                                        if not re.search(rf'<strong[^>]*>{re.escape(matched_text)}', highlighted_content, re.IGNORECASE):
                                            start, end = match.span()
                                            before = highlighted_content[:start]
                                            after = highlighted_content[end:]
                                            highlighted_content = before + f'<strong style="background-color: #ffc107; padding: 3px 5px; border-radius: 3px; font-weight: bold;">{matched_text}</strong>' + after
                                    break
                    
                    if answer_numbers:
                        for num in answer_numbers:
                            num_clean = num.replace('.', '')
                            num_with_dot = num
                            if not re.search(rf'<strong[^>]*>{re.escape(num_with_dot)}', highlighted_content, re.IGNORECASE):
                                patterns_to_highlight = [
                                    (rf'\b{re.escape(num_with_dot)}\s*(?:billion|million|thousand)\b', 'high'),
                                    (rf'\$\s*{re.escape(num_with_dot)}\b', 'high'),
                                ]
                                for pattern, priority in patterns_to_highlight:
                                    if re.search(pattern, highlighted_content, re.IGNORECASE):
                                        highlighted_content = re.sub(
                                            pattern,
                                            r'<strong style="background-color: #ffc107; padding: 3px 5px; border-radius: 3px; font-weight: bold;">\g<0></strong>',
                                            highlighted_content,
                                            flags=re.IGNORECASE
                                        )
                    
                    highlighted_content = re.sub(
                        r'(\b(?:TOTAL|NET|GROSS|OPERATING)\s+(?:REVENUE|INCOME|SALES)\s*\$?\s*[\d,\.]+(?:\s*(?:billion|million|thousand))?)',
                        r'<strong style="background-color: #fff3cd; padding: 2px 4px; border-radius: 3px;">\1</strong>',
                        highlighted_content,
                        flags=re.IGNORECASE
                    )
                    
                    highlighted_content = re.sub(
                        r'(\$\s*[\d,\.]+\s*(?:billion|million|thousand))',
                        r'<strong style="background-color: #fff3cd; padding: 2px 4px; border-radius: 3px;">\1</strong>',
                        highlighted_content,
                        flags=re.IGNORECASE
                    )
                    
                    query_words = [w for w in user_q.lower().split() if len(w) > 3 and w not in ['what', 'was', 'were', 'when', 'where', 'which', 'who', 'how', 'accenture', 'quantos', 'quantas', 'qual', 'quais', 'onde', 'quando', 'quem', 'como', 'quanto']]
                    for word in query_words:
                        if not re.search(rf'<strong[^>]*>{re.escape(word)}', highlighted_content, re.IGNORECASE):
                            highlighted_content = re.sub(
                                f'\\b({re.escape(word)})\\b',
                                r'<strong style="background-color: #e7f3ff; padding: 2px 4px; border-radius: 3px;">\1</strong>',
                                highlighted_content,
                                flags=re.IGNORECASE
                            )
                    
                    st.caption(f"**Source {i} - Page {page}**")
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.9em;'>{highlighted_content}</div>", unsafe_allow_html=True)
                    if i < len(relevant_docs):
                        st.divider()
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

st.divider()
st.caption("Powered by LangChain + HuggingFace + FAISS")

