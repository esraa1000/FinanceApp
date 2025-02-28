from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
import yfinance as yf

os.environ["GOOGLE_API_KEY"] = "AIzaSyDKCUK95yC7-KxkIZV_E_2hD0E9xaxb5V4"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("fiess_persona_index")

def get_conversational_chain():
    prompt_template = """ 
    You are a financial expert with expertise in budgeting, saving, investing, debt management, and wealth-building strategies. 
    Use this context to generate suitable answers. If you cannot find the answer in the context, use your own knowledge.
    Ensure that your answers are easy to understand and concise.

    🔹 **Context**: \n{context}\n
    🔹 **Chat History**: \n{chat_history}\n
    🔹 **Question**: \n{question}\n

    Answer: 
    """

    custom_model_name = "gemini-2.0-flash"
    model = ChatGoogleGenerativeAI(model=custom_model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def prepare_the_vector_db(pdf_files):
    text = get_pdf_text(pdf_files)
    print("Extracted text from all PDFs.")
    
    chunks = get_text_chunks(text)
    print("Number of chunks:", len(chunks))
    
    get_vector_store(chunks)
    assert os.path.exists("fiess_persona_index"), "FAISS index not created!"
    print("FAISS index created successfully.")

company_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
company_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

company_advice_prompt = PromptTemplate(
    input_variables=["query", "chat_history", "live_data", "context"],
    template=""" 
    You are a financial advisor specializing in stock markets and economic trends.
    
    🔹 **Live Stock Data**: {live_data}
    🔹 **User's Previous Conversations**: {chat_history}
    🔹 **Relevant Context**: {context}
    🔹 **User's Current Question**: {query}

    Based on this, provide **accurate, actionable, and data-driven financial advice**.
    Avoid generic responses and ensure rational market analysis and make your answer easy to understand and not too long. if you could not find the answer in the context you must use your knowledge to answer the question 
    """
)

def get_live_financial_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return f"The current stock price of {stock_symbol} is ${price:.2f}."
    except Exception as e:
        return f"Live financial data is currently unavailable. Error: {str(e)}"

def give_company_advice(query, stock_symbol=None):
    live_data = get_live_financial_data(stock_symbol) if stock_symbol else "No stock symbol provided."

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = FAISS.load_local("fiess_company_index", embeddings, allow_dangerous_deserialization=True)

    relevant_docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = company_advice_prompt.format(
        query=query,
        chat_history=company_memory.load_memory_variables({}).get("chat_history", ""),
        live_data=live_data,
        context=context
    )

    response = company_model.invoke(prompt)

    company_memory.save_context({"query": query}, {"response": response.content})

    return response.content


