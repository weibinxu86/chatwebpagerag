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

openaimodel="gpt-4o"
load_dotenv()

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
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, url):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_stores[url])
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_histories[url],
        "input": user_input
    })
    return response['answer']

st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL", key="new_url")
    if st.button("Load Website"):
        if "vector_stores" not in st.session_state:
            st.session_state.vector_stores = {}
            st.session_state.chat_histories = {}
        if website_url not in st.session_state.vector_stores:
            st.session_state.vector_stores[website_url] = get_vectorstore_from_url(website_url)
            st.session_state.chat_histories[website_url] = [AIMessage(content="Hello, I am a bot. How can I help you based on the new website?")]
        st.session_state.current_url = website_url

current_chat_history = st.session_state.chat_histories.get(st.session_state.current_url, [])
user_query = st.text_input("Type your message here...")
if user_query:
    response = get_response(user_query, st.session_state.current_url)
    current_chat_history.append(HumanMessage(content=user_query))
    current_chat_history.append(AIMessage(content=response))
    st.session_state.chat_histories[st.session_state.current_url] = current_chat_history

for message in current_chat_history:
    if isinstance(message, AIMessage):
        with st.container():
            st.write("AI:", message.content)
    elif isinstance(message, HumanMessage):
        with st.container():
            st.write("Human:", message.content)
