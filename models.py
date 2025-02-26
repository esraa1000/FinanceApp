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


# 🔹 Step 1: Set Up Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBxHDrSpG4vt9BNbTDFaJP-2HO5fsuB_nY"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 🔹 Step 2: Initialize Gemini Models
personal_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
company_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
graph_analyzer_model = genai.GenerativeModel("gemini-2.0-flash")

# 🔹 Step 3: Memory for Context-Aware Conversations
personal_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
company_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🔹 Step 4: Structured Prompts
personal_financial_advice_prompt = PromptTemplate(
    input_variables=["query", "chat_history"],
    template=""" 
    You are a financial advisor with expertise in budgeting, saving, investing, debt management, and wealth-building strategies.

    🔹 **User's Previous Conversations**: {chat_history}
    🔹 **User's Current Question**: {query}

    Based on the user's past inquiries and the current question, provide **clear, actionable, and practical financial advice** that can be applied broadly.
    Avoid generic responses and instead offer **specific, structured, and insightful guidance** tailored to common financial situations.
    """
)


company_advice_prompt = PromptTemplate(
    input_variables=["query", "chat_history", "live_data"],
    template=""" 
    You are a financial advisor specializing in stock markets and economic trends.
    
    🔹 **Live Stock Data**: {live_data}
    🔹 **User's Previous Conversations**: {chat_history}
    🔹 **User's Current Question**: {query}

    Based on this, provide **accurate, actionable, and data-driven financial advice**.
    Avoid generic responses and ensure rational market analysis.
    """
)

# 🔹 Step 5: Define Function to Fetch Live Stock Data
def get_live_financial_data(stock_symbol):
    """Fetches real-time stock price for a given symbol."""
    try:
        price = si.get_live_price(stock_symbol)
        return f"The current stock price of {stock_symbol} is ${price:.2f}."
    except Exception:
        return "Live financial data is currently unavailable."

# 🔹 Step 6: Define Functions to Generate Advice
def give_personal_advice(query):
    """Returns personalized advice based on user query and chat history."""
    prompt = personal_financial_advice_prompt.format(
        query=query,
        chat_history=personal_memory.load_memory_variables({}).get("chat_history", "")
    )
    response = personal_model.invoke(prompt)
    personal_memory.save_context({"query": query}, {"response": response.content})
    return response.content

def give_company_advice(query, stock_symbol=None):
    """Returns financial advice based on user query and optional stock data."""
    live_data = get_live_financial_data(stock_symbol) if stock_symbol else "No stock symbol provided."
    prompt = company_advice_prompt.format(
        query=query,
        chat_history=company_memory.load_memory_variables({}).get("chat_history", ""),
        live_data=live_data
    )
    response = company_model.invoke(prompt)
    company_memory.save_context({"query": query}, {"response": response.content})
    return response.content



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

