import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
openaimodel = "gpt-4o"

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model=openaimodel)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model=openaimodel)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, url):
    if "vector_stores" not in st.session_state:
        st.session_state.vector_stores = {}
    if url not in st.session_state.vector_stores:
        st.session_state.vector_stores[url] = get_vectorstore_from_url(url)
    retriever_chain = get_context_retriever_chain(st.session_state.vector_stores[url])
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_histories[url],
        "input": user_input
    })
    return response['answer']

def display_chat():
    current_chat_history = st.session_state.chat_histories.get(st.session_state.current_url, [])
    for message in current_chat_history:
        if isinstance(message, AIMessage):
            st.markdown(f"<p style='color:gray; background-color:#F0F0F0; padding:10px; border-radius:10px;'>{message.content}</p>", unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
            st.markdown(f"<p style='color:black; background-color:#D9EDF7; padding:10px; border-radius:10px;'>{message.content}</p>", unsafe_allow_html=True)

st.set_page_config(page_title="Chat with websites", page_icon="🤖", layout='wide')
st.title("Chat with websites")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL", key="new_url")
    if st.button("Load Website"):
        st.session_state.current_url = website_url
        if "chat_histories" not in st.session_state:
            st.session_state.chat_histories = {}
        if website_url not in st.session_state.chat_histories:
            st.session_state.chat_histories[website_url] = [AIMessage(content="Hello, I am a bot. How can I help you based on the new website?")]

display_chat()

# Input for new messages
input_container = st.empty()
user_input = input_container.text_input("Type your message here...", key="user_input")
if user_input:
    response = get_response(user_input, st.session_state.current_url)
    st.session_state.chat_histories[st.session_state.current_url].append(HumanMessage(content=user_input))
    st.session_state.chat_histories[st.session_state.current_url].append(AIMessage(content=response))
    display_chat()
    input_container.text_input("Type your message here...", key="user_input", value="")  # Reset the input field
