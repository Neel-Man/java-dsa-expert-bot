import os
import time
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_groq import ChatGroq
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()


def start_dsa_bot():
    # 1️⃣ Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # 2️⃣ Connect to Existing Vector DB
    if not os.path.exists("./dsa_bot_db"):
        print("❌ Error: Vector database folder './dsa_bot_db' not found!")
        return

    vector_db = Chroma(
        persist_directory="./dsa_bot_db",
        embedding_function=embeddings,
    )

    # 3️⃣ Setup LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.1,
        max_tokens=1024,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    # 1. Define your custom prompt for generating variations
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant helping to improve search results for Java Data Structures and Algorithms.
            
        Generate 3 different versions of the given question to retrieve relevant documents from a technical PDF.
        Include variations that:
        1. Use technical synonyms (e.g., swap 'fast' with 'time complexity' or 'O(n log n)')
        2. Focus on implementation (e.g., 'how to code' vs 'logic of')
        3. Rephrase as a technical statement.

        Original question: {question}

        Output only the variations, one per line:"""
    )

    multi_query = MultiQueryRetriever.from_llm(
        retriever = retriever,
        llm = llm,
        prompt=QUERY_PROMPT
    )

    compressor = FlashrankRerank()

    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=multi_query
    )

    # 4️⃣ Prompt Template
    system_prompt = (
    "You are a Java Data Structures and Algorithms Expert.\n\n"

    "Answer the user's question STRICTLY using only the provided context. "
    "Do NOT use prior knowledge. If the answer is not in the context, say: "
    "'I don't have enough information in the provided notes.'\n\n"

    "### STEP 1: INTENT DETECTION\n"
    "Analyze the 'Question Word' or the user's goal to choose the right template:\n"
    "- **'HOW' or 'IMPLEMENT' (Process Intent):** Use the ALGORITHM format.\n"
    "- **'WHAT' or 'DEFINE' (Definition Intent):** Use the DATA STRUCTURE format.\n"
    "- **'WHY', 'WHERE', or 'WHEN' (Context Intent):** Use the CONCEPT format.\n"
    "- **'FIX', 'ERROR', or 'WHY IS THIS WRONG' (Troubleshooting Intent):** Use the ERROR format.\n\n"
    "- **'COMPARE' or 'DIFFERENCE' (Analysis Intent):** Use the COMPARISON format.\n\n"

    "### STEP 2: FORMATTING RULES\n"
    "1. **PROCESS INTENT ('How'):**\n"
    "- **Step-by-Step Logic Flow:** (Detailed procedural explanation)\n"
    "- **Complexity Table:** (Time & Space)\n"
    "- **Java Implementation:** (Clean, well-commented code snippet)\n\n"

    "2. **DEFINITION INTENT ('What'):**\n"
    "- **Technical Definition**\n"
    "- **Key Properties & Characteristics**\n"
    "- **Basic Operations & Complexity**\n"
    "- **Example Structure Code**\n\n"

    "3. **CONTEXT INTENT ('Why/Where/When'):**\n"
    "- **Conceptual Overview**\n"
    "- **Real-world Applications**\n"
    "- **Trade-offs:** (Pros vs. Cons compared to other structures)\n"
    "- **Best Practices in Java**\n\n"

    "4. **TROUBLESHOOTING INTENT ('Fix/Error'):**\n"
    "- **Root Cause:** (Why this error occurs in Java)\n"
    "- **Standard Solution:** (Step-by-step fix)\n"
    "- **Corrected Code Example**\n\n"

    "5. **ANALYSIS INTENT ('Compare/Difference'):**\n"
    "- **Side-by-Side Comparison Table:** (Compare parameters like Time, Space, Stability, or Usage)\n"
    "- **Key Differences:** (Bullet points highlighting the main distinctions)\n"
    "- **Winner/Best Choice:** (When to pick one over the other based on context)\n\n"

    "Use bold headers and bullet points. If you are unsure of the intent, default to the **DEFINITION** format.\n\n"

    "Context:\n{context}"
)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 5️⃣ Format Documents Function
    def format_docs_with_sources(docs):
        formatted = []
        for doc in docs:
            page_num = doc.metadata.get("page", "Unknown")
            content = f"[Source: Page {page_num}]\n{doc.page_content}"
            formatted.append(content)
        return "\n\n---\n\n".join(formatted)
    # 6️⃣ Modern LCEL RAG Chain (LangChain 1.x)
    rag_chain = (
        {
            "context": rerank_retriever | format_docs_with_sources,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return {
        "simple": retriever,
        "mq": multi_query,
        "rerank": rerank_retriever,
        "llm": llm,
        "prompt": prompt,
        "format_func": format_docs_with_sources,
        "rag_chain": rag_chain
    }


if __name__ == "__main__":
    # This block ONLY runs if you type 'python chat_bot.py'
    # It will NOT run when Streamlit imports the file.
    components = start_dsa_bot()
    chain = components["rag_chain"]
    
    print("\n🚀 Terminal Mode Active!")
    while True:
        query = input("\n👨‍💻 Ask a question: ")
        if query.lower() in ["exit", "quit"]: break
        print(chain.invoke(query))