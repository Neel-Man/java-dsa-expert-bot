import streamlit as st
import time
from chat_bot import start_dsa_bot
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1️⃣ Page Configuration & Styling
st.set_page_config(page_title="Java DSA Expert", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    .stChatInput { border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("👨‍💻 Java DSA Study Assistant")
st.caption("Powered by Llama-3.3 & FlashRank Re-ranking")

# 2️⃣ Initialize Backend Components (Cached to avoid reloading)
@st.cache_resource
def initialize_engine():
    # We call your function but need to make sure it returns the chain
    return start_dsa_bot()

# NOTE: For this to work, ensure start_dsa_bot() returns the 'rag_chain'
engine_components = initialize_engine()

# Check if start_dsa_bot returns the dictionary we planned or just starts a loop
# For evaluation/UI, we need the dictionary of components
try:
    simple_retriever = engine_components["simple"]
    mq_retriever = engine_components["mq"]
    rerank_retriever = engine_components["rerank"]
    llm = engine_components["llm"]
    prompt_template = engine_components["prompt"]
    format_func = engine_components["format_func"]
    
    # Final RAG Chain for the UI
    rag_chain = (
        {"context": rerank_retriever | format_func, "input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
except:
    st.error("Backend logic needs to return components! Please check chat_bot.py updates.")
    st.stop()

# 3️⃣ Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4️⃣ Sidebar for Project Stats
with st.sidebar:
    st.header("📊 Project Overview")
    st.write(f"**Created by:** {st.session_state.get('user_name', 'Neelesh Manjhi')}")
    st.info("A RAG based chatbot to make DSA easy for you.")
    st.success("Status: Advanced RAG Active")
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# 5️⃣ User Input & Response Logic
if user_query := st.chat_input("Ask a DSA question..."):
    # Display user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        # 1. Get both the answer AND the source documents
        # We invoke the retriever directly to get the actual Document objects
        with st.status("🔍 Thinking...", expanded=False) as status:
            source_docs = engine_components["rerank"].invoke(user_query)
            full_response = engine_components["rag_chain"].invoke(user_query)
            status.update(label="✅ Analysis Complete", state="complete")

        # 2. Display the main answer
        st.markdown(full_response)

        # 3. Add the Source Attribution Expander
        with st.expander("📚 View The Chunks(Hallucination Check)"):
            for i, doc in enumerate(source_docs):
                page = doc.metadata.get("page", "Unknown")
                st.write(f"**Source {i+1}: Page {page}**")
                st.caption(doc.page_content[:300] + "...") # Preview text
                st.divider()
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})