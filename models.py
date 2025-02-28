import os
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from yahoo_fin import stock_info as si
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory  # For chat history management
import yfinance as yf  # For live stock data


#function: give_personal_financial_advice(user_question)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDKCUK95yC7-KxkIZV_E_2hD0E9xaxb5V4"

# Initialize ConversationBufferMemory for chat history
personal_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

personal_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    """
    Extract text from a list of PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    """
    Split text into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(chunks):
    """
    Create a vector store from text chunks and save it to disk.
    """
    documents = [Document(page_content=chunk) for chunk in chunks]  # Wrap chunks in Document objects
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")) # Pass the API key here
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("fiess_persona_index")

# Function to create a conversational chain
def get_conversational_chain():
    """
    Create a conversational chain for question answering.
    """
    prompt_template = """ 
    You are a financial expert with expertise in budgeting, saving, investing, debt management, and wealth-building strategies. 
    Use this context to generate suitable answers. If you cannot find the answer in the context, use your own knowledge.
    Ensure that your answers are easy to understand and concise.

    🔹 **Context**: \n{context}\n
    🔹 **Chat History**: \n{chat_history}\n
    🔹 **Question**: \n{question}\n

    Answer: 
    """

    # Load your custom-tuned model here
    custom_model_name = "gemini-2.0-flash"  # Your custom model name
    model = ChatGoogleGenerativeAI(model=custom_model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))  # Pass the API key here

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to give personal financial advice
def give_personal_financial_advice(user_question):
    """
    Query the vector store and generate a response using the custom model.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))  # Pass the API key here
    
    # Allow dangerous deserialization since you trust the source of the FAISS index
    new_db = FAISS.load_local("fiess_persona_index", embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve relevant documents
    docs = new_db.similarity_search(user_question)
    
    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Load chat history from memory
    chat_history = personal_memory.load_memory_variables({}).get("chat_history", "")
    
    # Generate the response
    response = chain(
        {
            "input_documents": docs,
            "chat_history": chat_history,  # Pass chat history from memory
            "question": user_question,
        },
        return_only_outputs=True,
    )

    # Save the interaction to memory
    personal_memory.save_context({"input": user_question}, {"output": response.get("output_text", "No response generated.")})

    return response.get("output_text", "No response generated.")

# Function to prepare the vector database
def prepare_the_vector_db(pdf_files):
    """
    Test the entire pipeline with multiple PDF files.
    """
    # Step 1: Read text from all PDFs
    text = get_pdf_text(pdf_files)
    print("Extracted text from all PDFs.")
    
    # Step 2: Split text into chunks
    chunks = get_text_chunks(text)
    print("Number of chunks:", len(chunks))
    
    # Step 3: Create vector store
    get_vector_store(chunks)
    assert os.path.exists("fiess_persona_index"), "FAISS index not created!"
    print("FAISS index created successfully.")

# List of PDF files to process
pdf_files = [
    "the-intelligent-investor.pdf",  # Replace with your PDF file paths
    "The Total Money Makeover - Dave Ramsey.pdf",
    "kotobati - The Richest Man in Babylon.pdf"
]

# Run the pipeline with multiple PDFs
prepare_the_vector_db(pdf_files)

# Example usage
response = give_personal_financial_advice("What is the best way to save money?")
print(response)


### here is the end################################################################################################################################

#function: give_company_advice(query, stock_symbol=None)
# Initialize the Google Generative AI model
company_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize ConversationBufferMemory for chat history
company_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the prompt template for financial advice
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

# Function to fetch live financial data using yfinance
def get_live_financial_data(stock_symbol):
    """
    Fetches real-time stock price for a given symbol using yfinance.
    """
    try:
        stock = yf.Ticker(stock_symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return f"The current stock price of {stock_symbol} is ${price:.2f}."
    except Exception as e:
        return f"Live financial data is currently unavailable. Error: {str(e)}"

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    """
    Extract text from a list of PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    """
    Split text into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(chunks):
    """
    Create a vector store from text chunks and save it to disk.
    """
    documents = [Document(page_content=chunk) for chunk in chunks]  # Wrap chunks in Document objects
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("fiess_company_index")

# Function to give financial advice
def give_company_advice(query, stock_symbol=None):
    """
    Returns financial advice based on user query and optional stock data.
    """
    # Fetch live financial data if a stock symbol is provided
    live_data = get_live_financial_data(stock_symbol) if stock_symbol else "No stock symbol provided."

    # Load the vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = FAISS.load_local("fiess_company_index", embeddings, allow_dangerous_deserialization=True)

    # Retrieve relevant documents from the vector store
    relevant_docs = vector_store.similarity_search(query, k=3)  # Retrieve top 3 relevant documents
    context = "\n".join([doc.page_content for doc in relevant_docs])  # Combine documents into a single context string

    # Format the prompt with the query, chat history, live data, and context
    prompt = company_advice_prompt.format(
        query=query,
        chat_history=company_memory.load_memory_variables({}).get("chat_history", ""),
        live_data=live_data,
        context=context
    )

    # Invoke the model to generate a response
    response = company_model.invoke(prompt)

    # Save the interaction to memory
    company_memory.save_context({"query": query}, {"response": response.content})

    return response.content


# List of PDF files to process
pdf_files = [
  "m6K5J3_corporate finance 5.pdf",
  "m6K5J3_corporate finance 5.pdf",
  "Quantitative Financial Risk Management - 2015 - Zopounidis.pdf",
  "principles-of-corporate-finance-finance-insurance-and-real-estate-10th-ed-10nbsped-0073530735-9780073530734_compress.pdf"
]

# Run the pipeline with multiple PDFs
prepare_the_vector_db(pdf_files)




#####################################################################################


graph_analyzer_model = genai.GenerativeModel("gemini-2.0-flash")



def load_image(image_path):
    return Image.open(image_path)

# 🔹 Send the Image to Gemini for Analysis
def analyze_graph_image(image_path):
    """Analyzes a graph from an image using Gemini Pro Vision."""
    
    # Load the image
    image = load_image(image_path)

    # Define prompt
    prompt = """
    This is a graph image. Analyze the trends, anomalies, and key insights.
    Provide details on the type of graph, patterns, and possible predictions.
    """

    # Generate response
    response = graph_analyzer_model.generate_content([prompt, image])
    
    return response.text




# 🔹 Initialize Gemini Model & Memory
financial_insight_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
financial_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🔹 Define Prompt Template
financial_advice_prompt = PromptTemplate(
    input_variables=["query", "chat_history", "financial_data"],
    template=""" 
    You are a financial advisor specializing in personal financial management.
    
    🔹 **User's Financial Data**: {financial_data}
    🔹 **User's Previous Conversations**: {chat_history}
    🔹 **User's Current Question**: {query}
    
    Based on this, provide **actionable financial advice**. Ensure insights are data-driven and personalized.
    """
)


## this is for personal finance management syatem 
# 🔹 Sample Transaction Data
transactions = [
    {"date": "2024-02-01", "category": "Food", "amount": 50, "type": "Expense"},
    {"date": "2024-02-03", "category": "Salary", "amount": 3000, "type": "Income"},
    {"date": "2024-02-07", "category": "Rent", "amount": 800, "type": "Expense"},
    {"date": "2024-02-10", "category": "Investment", "amount": 500, "type": "Income"},
    {"date": "2024-02-15", "category": "Shopping", "amount": 200, "type": "Expense"},
    {"date": "2024-02-20", "category": "Entertainment", "amount": 150, "type": "Expense"},
]

df = pd.DataFrame(transactions)
df["date"] = pd.to_datetime(df["date"])

# 🔹 Function to Handle Analysis and Insights
def financial_analysis(query, plot_type=None):
    """
    Generates financial insights and visualizations based on user queries.

    :param query: User's financial question.
    :param plot_type: Type of plot to generate (e.g., 'spending_trends', 'income_vs_expenses', etc.).
    :return: AI-generated financial insights.
    """

    # Plot functions
    def plot_spending_trends():
        expense_df = df[df["type"] == "Expense"].groupby("date")["amount"].sum()
        plt.figure(figsize=(10, 5))
        plt.plot(expense_df.index, expense_df.values, marker='o', linestyle='-')
        plt.xlabel("Date"), plt.ylabel("Total Expenses"), plt.title("Spending Trends Over Time")
        plt.xticks(rotation=45), plt.grid()
        plt.show()

    def plot_income_vs_expenses():
        grouped = df.groupby(["date", "type"])["amount"].sum().unstack(fill_value=0)
        grouped.plot(kind="bar", stacked=False, figsize=(10, 5), color=["red", "green"])
        plt.xlabel("Date"), plt.ylabel("Amount"), plt.title("Income vs. Expenses")
        plt.xticks(rotation=45), plt.legend(["Expense", "Income"]), plt.grid()
        plt.show()

    def plot_expense_breakdown():
        expense_df = df[df["type"] == "Expense"].groupby("category")["amount"].sum()
        expense_df.plot(kind="pie", autopct="%1.1f%%", startangle=140, cmap="coolwarm", figsize=(7, 7))
        plt.title("Expense Breakdown by Category"), plt.ylabel("")
        plt.show()

    def plot_monthly_expense_trend():
        df["month"] = df["date"].dt.to_period("M")
        expense_df = df[df["type"] == "Expense"].groupby(["month", "category"])["amount"].sum().unstack(fill_value=0)
        expense_df.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="viridis")
        plt.xlabel("Month"), plt.ylabel("Total Expenses"), plt.title("Monthly Expense Trend")
        plt.xticks(rotation=45), plt.legend(title="Category"), plt.grid()
        plt.show()

    def plot_cumulative_savings():
        df["net_savings"] = df.apply(lambda x: x["amount"] if x["type"] == "Income" else -x["amount"], axis=1)
        df["cumulative_savings"] = df["net_savings"].cumsum()
        plt.fill_between(df["date"], df["cumulative_savings"], color="green", alpha=0.5)
        plt.plot(df["date"], df["cumulative_savings"], marker="o", linestyle="-", color="green")
        plt.xlabel("Date"), plt.ylabel("Cumulative Savings"), plt.title("Cumulative Savings Over Time")
        plt.xticks(rotation=45), plt.grid()
        plt.show()

    def plot_anomalies():
        plt.scatter(df["date"], df["amount"], c=df["amount"], cmap="coolwarm", edgecolors="black")
        plt.xlabel("Date"), plt.ylabel("Transaction Amount"), plt.title("Financial Anomalies (Unusual Transactions)")
        plt.xticks(rotation=45), plt.colorbar(label="Transaction Amount"), plt.grid()
        plt.show()

    # Generate plot based on the requested type
    plot_mapping = {
        "spending_trends": plot_spending_trends,
        "income_vs_expenses": plot_income_vs_expenses,
        "expense_breakdown": plot_expense_breakdown,
        "monthly_expense_trend": plot_monthly_expense_trend,
        "cumulative_savings": plot_cumulative_savings,
        "anomalies": plot_anomalies
    }

    if plot_type in plot_mapping:
        plot_mapping[plot_type]()

    # Generate AI-based Financial Insights
    prompt = financial_advice_prompt.format(
        query=query,
        chat_history=financial_memory.load_memory_variables({}).get("chat_history", ""),
        financial_data=df.to_dict(orient="records")
    )
    response = financial_insight_model.invoke(prompt)
    financial_memory.save_context({"query": query}, {"response": response.content})

    return response.content

