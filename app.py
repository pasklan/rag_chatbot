import os
import tempfile
import streamlit as st

from decouple import config

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI


os.environ['GEMINI_API_KEY'] = config('GEMINI_API_KEY')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
api_key = os.environ['GEMINI_API_KEY']
persist_directory =  'db'
embedding = GoogleGenerativeAIEmbeddings(
    model='models/embedding-001',
)

# Function to process PDF files and split them into chunks
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=-False, suffix='.pdf') as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    os.remove(tmp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_splitter.split_documents(documents=docs)

    return chunks

# Function to load existing vector store if it exists
def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        return vector_store
    return None

# Function to add chunks to the vector store
def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_directory,
        )

    return vector_store

# Function to ask a question using the selected model and vector store
def ask_question(model, query, vector_store):
    llm = GoogleGenerativeAI(model=model)
    retriever = vector_store.as_retriever()

    system_prompt = '''
    Use o contexto para responder as perguntas.
    Se não encontrar uma resposta no contexto,
    explique que não há informações disponíveis.
    Reponda em formato de markdown e com
    visualizações elaboradas e interativas.
    Contexto: {context}
    '''
    messages = [('system', system_prompt),]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({'input': query})
    return response.get('answer')

vector_store = load_existing_vector_store()

# Streamlit app configuration
st.set_page_config(
    page_title="Chat PyGPT",
    page_icon="🦾",
)
st.header("🤖 Chat com seus documentos (RAG)")

# Sidebar for file upload and model selection
with st.sidebar:
    st.header("Upload de arquivos")
    uploaded_files = st.file_uploader(
        label='Faça o upload de arquivos PDF',
        type=['pdf'],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner('Processando documentos...'):
            all_chunks = []
            for upfile in uploaded_files:
                chunks = process_pdf(file=upfile)
                all_chunks.extend(chunks)
                print(all_chunks)

            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )

    model_options = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gemini-2.0-flash-001"
    ]

    selected_model = st.sidebar.selectbox(
        label="Selecione o modelo LLM",
        options=model_options,
        index=2,
    )

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input("Como posso ajudar?")

# Handle user input and generate response
if vector_store and question:

        for message in st.session_state['messages']:
            st.chat_message(message.get('role')).write(message.get('content'))

        st.chat_message('user').write(question)
        st.session_state.messages.append({'role': 'user', 'content': question})

        with st.spinner('Gerando resposta...'):

            response = ask_question(
                model=selected_model,
                query=question,
                vector_store=vector_store,
            )

            st.chat_message('ai').write(response)
            st.session_state.messages.append({'role': 'ai', 'content': response})