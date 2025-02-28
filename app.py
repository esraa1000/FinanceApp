from dataclasses import dataclass
from typing import Literal
import streamlit as st
from pages.user_chatbot import give_personal_advice
#from pages.company import give_company_advice
import streamlit.components.v1 as components
from pages.stocks import build_lstm_model, preprocess_data, train_model, predict_stock
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pages.tracker import add_transaction, get_balance, get_expenses
from auth import app as auth_app


# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Chatbot_User"

# Main App Flow
if not st.session_state.get('signedout'):
    auth_app()  # Show login/signup page
else:
    # Get account type from session state
    account_type = st.session_state.get('account_type', 'individual')
    


def chatbot_user_page():
    
    @dataclass
    class Message:
      """Class for keeping track of a chat message."""
      origin: Literal["human", "ai"]
      message: str

    def load_css():
      with open("static/styles.css", "r") as f:
          css = f"<style>{f.read()}</style>"
          st.markdown(css, unsafe_allow_html=True)

    def initialize_session_state():
        if "history" not in st.session_state:
            st.session_state.history = []
        if "context" not in st.session_state:
            st.session_state.context = None  # Can be "personal" or "company"
    def on_click_callback():
        human_prompt = st.session_state.human_prompt
        if human_prompt.strip():  # Check if the input is not empty
            # Generate AI response using personal financial advice function
            ai_response = give_personal_advice(human_prompt)

            # Append the conversation to the history
            st.session_state.history.append(Message("human", human_prompt))
            st.session_state.history.append(Message("ai", ai_response))

            # Clear the input field
            st.session_state.human_prompt = ""

    
      # Load CSS and initialize session state
    load_css()
    initialize_session_state()

  # UI Components
    st.title("Ask Mahmoud 🤖")

    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")

    with chat_placeholder:
        for chat in st.session_state.history:
            st.markdown(
                f'<div class="chat-row {"" if chat.origin == "ai" else "row-reverse"}">'
                f'<div class="chat-bubble {"ai-bubble" if chat.origin == "ai" else "human-bubble"}">'
                f'{chat.message}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    with prompt_placeholder:
        st.markdown("**Chat**")
        cols = st.columns((6, 1))
        cols[0].text_input(
            "Chat",
            value=st.session_state.get("human_prompt", ""),  # Bind to session state
            label_visibility="collapsed",
            key="human_prompt",
        )
        cols[1].form_submit_button(
            "Submit", 
            type="primary", 
            on_click=on_click_callback, 
        )

    # JavaScript for handling Enter key
    components.html("""
    <script>
    const streamlitDoc = window.parent.document;

    const buttons = Array.from(
        streamlitDoc.querySelectorAll('.stButton > button')
    );
    const submitButton = buttons.find(
        el => el.innerText === 'Submit'
    );

    streamlitDoc.addEventListener('keydown', function(e) {
        switch (e.key) {
            case 'Enter':
                submitButton.click();
                break;
        }
    });
    </script>
    """, 
        height=0,
        width=0,
    )

def chatbot_company_page():
    
    @dataclass
    class Message:
      """Class for keeping track of a chat message."""
      origin: Literal["human", "ai"]
      message: str

    def load_css():
      with open("static/styles.css", "r") as f:
          css = f"<style>{f.read()}</style>"
          st.markdown(css, unsafe_allow_html=True)

    def initialize_session_state():
        if "history" not in st.session_state:
            st.session_state.history = []
        if "context" not in st.session_state:
            st.session_state.context = None  # Can be "personal" or "company"
    def on_click_callback():
        human_prompt = st.session_state.human_prompt
        if human_prompt.strip():  # Check if the input is not empty
            # Generate AI response using personal financial advice function
            ai_response = give_personal_advice(human_prompt)

            # Append the conversation to the history
            st.session_state.history.append(Message("human", human_prompt))
            st.session_state.history.append(Message("ai", ai_response))

            # Clear the input field
            st.session_state.human_prompt = ""

    
      # Load CSS and initialize session state
    load_css()
    initialize_session_state()

  # UI Components
    st.title("Ask Mahmoud 🤖")

    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")

    with chat_placeholder:
        for chat in st.session_state.history:
            st.markdown(
                f'<div class="chat-row {"" if chat.origin == "ai" else "row-reverse"}">'
                f'<div class="chat-bubble {"ai-bubble" if chat.origin == "ai" else "human-bubble"}">'
                f'{chat.message}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    with prompt_placeholder:
        st.markdown("**Chat**")
        cols = st.columns((6, 1))
        cols[0].text_input(
            "Chat",
            value=st.session_state.get("human_prompt", ""),  # Bind to session state
            label_visibility="collapsed",
            key="human_prompt",
        )
        cols[1].form_submit_button(
            "Submit", 
            type="primary", 
            on_click=on_click_callback, 
        )

    # JavaScript for handling Enter key
    components.html("""
    <script>
    const streamlitDoc = window.parent.document;

    const buttons = Array.from(
        streamlitDoc.querySelectorAll('.stButton > button')
    );
    const submitButton = buttons.find(
        el => el.innerText === 'Submit'
    );

    streamlitDoc.addEventListener('keydown', function(e) {
        switch (e.key) {
            case 'Enter':
                submitButton.click();
                break;
        }
    });
    </script>
    """, 
        height=0,
        width=0,
    )



def stocks_company_page():
    st.title('Company Stock Prediction')

    st.markdown("""
    This app allows users to:
    - Upload a CSV file containing stock data
    - View closing price plots
    - Get **LSTM-based stock predictions** for their companies.
    """)

    uploaded_file = st.sidebar.file_uploader("Upload CSV file (Company Stock Data)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

    # Find the correct closing price column
        possible_close_names = ["Close", "Adj Close", "closing_price"]
        close_column = next((col for col in possible_close_names if col in df.columns), None)

        if close_column is None:
            st.error(f"CSV must contain a closing price column (e.g., 'Close', 'Adj Close'). Found: {df.columns.tolist()}")
            st.stop()

        # Convert 'Date' to DateTime if exists
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

            # Plot stock prices
            st.write("### Stock Closing Price Over Time")
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(10,5))

            # Plot stock prices
            ax.plot(df.index, df[close_column], label="Closing Price", color='blue')

            ax.set_xlabel('Date')
            ax.set_ylabel('Closing Price')
            ax.set_title('Stock Closing Prices')
            ax.legend()

            # Pass fig to st.pyplot()
            st.pyplot(fig)
            # Preprocess data
            sequence_length = 100
            x_train, y_train, scaler = preprocess_data(df.iloc[:-sequence_length])
            x_test, y_test, _ = preprocess_data(df.iloc[-sequence_length:])

            # Build and Train Model (instead of loading a pre-trained model)
            model = build_lstm_model(input_shape=(x_train.shape[1], 1))
            model, history = train_model(model, x_train, y_train, x_test, y_test)

            # Predict
            predicted_prices = predict_stock(model, scaler, df)

            # Display Predictions
            st.write("### Predicted Stock Prices")
            plt.figure(figsize=(10,5))
            plt.plot(df.index[-len(predicted_prices):], df['Close'].values[-len(predicted_prices):], label="Actual Prices", color='blue')
            plt.plot(df.index[-len(predicted_prices):], predicted_prices, label="Predicted Prices", color='red', linestyle='dashed')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.title('Actual vs. Predicted Stock Prices')
            plt.legend()
            st.pyplot()

            st.write(f"### Predicted Next Closing Price: **${predicted_prices[-1]:.2f}**")


def fetch_stock_data(tickers, period="1y", interval="1d"):
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by='ticker',
        auto_adjust=True,
        prepost=True,
        threads=True
    )
    return data

def stocks_user_page():
    st.title('📈 Multicompany Stock Analysis & Prediction')

    st.markdown("""
    This app allows users to:
    - Select multiple stock symbols
    - View closing price plots
    - Get **LSTM-based stock predictions** for selected companies.
    """)

    # User input for stock symbols
    selected_stocks = st.sidebar.multiselect('✅ Select Stocks', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'], default=['AAPL', 'GOOGL'])

    if selected_stocks:
        # Fetch stock data
        data = fetch_stock_data(selected_stocks)

        # Display the data
        st.header('Stock Data')
        st.write(data)

        # Download button for CSV
        csv = data.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Stock Data as CSV",
            data=csv,
            file_name='stock_data.csv',
            mime='text/csv',
        )

        def price_plot(symbol):
            df_plot = pd.DataFrame(data[symbol].Close)
            df_plot['Date'] = df_plot.index
            plt.figure(figsize=(10, 5))
            plt.fill_between(df_plot.Date, df_plot.Close, color='skyblue', alpha=0.3)
            plt.plot(df_plot.Date, df_plot.Close, color='blue', alpha=0.8)
            plt.xticks(rotation=45)
            plt.title(f"{symbol} Closing Prices", fontweight='bold')
            plt.xlabel('Date', fontweight='bold')
            plt.ylabel('Closing Price (USD)', fontweight='bold')
            st.pyplot(plt)

        num_company = st.sidebar.slider('🔢 Number of Companies to Display', 1, len(selected_stocks), 5)

        if st.button('📊 Show Plots'):
            st.header('📌 Stock Closing Price')
            for i in selected_stocks[:num_company]:
                price_plot(i)

        def get_predictions(symbol):
            df_stock = pd.DataFrame(data[symbol].Close).dropna()
            train_size = int(len(df_stock) * 0.8)
            df_train = df_stock[:train_size]
            df_test = df_stock[train_size:]

            x_train, y_train, scaler = preprocess_data(df_train)
            x_test, y_test, _ = preprocess_data(df_test)

            model = build_lstm_model(input_shape=(x_train.shape[1], x_train.shape[2]))
            trained_model, history = train_model(model, x_train, y_train, x_test, y_test)

            dates, predicted = predict_stock(trained_model, scaler, df_test)

            return dates, predicted

        if st.button('📈 Show Predictions'):
            st.header('🔮 Stock Price Predictions')
            for i in selected_stocks[:num_company]:
                dates, predicted = get_predictions(i)
                actual = data[i]['Close'].values[-len(predicted):]  # Slice actual prices to match predictions
                
                if dates is None or predicted is None:
                    st.error(f"Failed to generate predictions for {i}. Please check the data.")
                    continue
                
                plt.figure(figsize=(10, 5))
                plt.plot(dates, actual, label="Actual Prices", color='blue')
                plt.plot(dates, predicted, label="Predicted Prices", color='red', linestyle='dashed')
                plt.xticks(rotation=45)
                plt.xlabel("Date")
                plt.ylabel("Stock Price (USD)")
                plt.title(f"Predicted vs. Actual: {i}")
                plt.legend()
                st.pyplot(plt)



def tracker_page():
    st.title("💰 Finance Tracker")

    # Define display_expenses() FIRST to avoid UnboundLocalError
    def display_expenses():
        expenses = get_expenses() or []  # Ensure get_expenses is imported
        
        st.header("📊 All Transactions")
        if expenses:
            st.table(expenses)
        else:
            st.write("No transactions recorded.")
        
        st.header("💵 Balance")
        st.write(f"Current Balance: **${get_balance():.2f}**")

    # --- Input fields and buttons below ---
    st.header("Add Transaction")
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_type = st.selectbox("Transaction Type", ["Income", "Expense"])
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")

    with col2:
        category = st.selectbox("Category", ["Food", "Clothes", "Entertainment", "Other"])
        description = st.text_input("Description")

    if st.button("Add Transaction"):
        add_transaction(transaction_type, amount, category, description)
        st.success("Transaction added!")

    # Handle "Show Expenses" button state
    if "show_expenses" not in st.session_state:
        st.session_state.show_expenses = False

    if st.button("Show Expenses"):
        st.session_state.show_expenses = not st.session_state.show_expenses  # Toggle visibility

    if st.session_state.show_expenses:
        display_expenses()  # Now correctly defined above



def news_page():
    st.title("News")
    st.write("Welcome to the News page!")
    # Add your news code here

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Chatbot_User"

# Custom CSS for the sidebar navigation
st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #F0F2F6; /* Light grey background */
    padding: 10px;
    border-radius: 10px;
}
.sidebar .stButton button {
    width: 100%;
    text-align: left;
    padding: 10px 20px;
    margin: 5px 0;
    border-radius: 5px;
    background-color: transparent;
    color: #333333; /* Dark text for contrast */
    font-size: 16px;
    transition: background-color 0.3s, color 0.3s;
}
.sidebar .stButton button:hover {
    background-color: #FF4B4B; /* Red hover effect */
    color: white; /* White text on hover */
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")

# Company-specific navigation
if account_type == 'company':
    page_options = {
        "🗨️ Chatbot (Company)": "Chatbot_Company",
        "📈 Stocks (Company)": "Company_Stocks",
        "📊 Business Tracker": "Tracker",
        "📰 Industry News": "News"
    }
else:
    # Individual navigation
    page_options = {
        "🗨️ Personal Chatbot": "Chatbot_User",
        "📈 Personal Stocks": "User_Stocks",
        "💰 Expense Tracker": "Tracker",
        "📰 Personal Finance News": "News"
    }

selected = st.sidebar.radio("Go to", list(page_options.keys()))
st.session_state.page = page_options[selected]

# Page routing
if st.session_state.page == "Chatbot_User":
    chatbot_user_page()
elif st.session_state.page == "Chatbot_Company":
    chatbot_company_page()
elif st.session_state.page == "Company_Stocks":
    stocks_company_page()
elif st.session_state.page == "User_Stocks":
    stocks_user_page()
elif st.session_state.page == "Tracker":
    tracker_page(account_type)
elif st.session_state.page == "News":
    news_page(account_type)

# Logout button
if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()



    
