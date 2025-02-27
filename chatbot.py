from dataclasses import dataclass
from typing import Literal
import streamlit as st
from models import give_personal_advice, give_company_advice  
import streamlit.components.v1 as components
from stocks import build_lstm_model, preprocess_data, train_model, predict_stock
import pandas as pd



def chatbot_page():
    
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
              # Determine the context if not already set
              if st.session_state.context is None:
                  if "personal" in human_prompt.lower():
                      st.session_state.context = "personal"
                  elif "company" in human_prompt.lower():
                      st.session_state.context = "company"
                  else:
                      # If no context is specified, ask for clarification
                      ai_response = "Please specify whether you want personal or company advice."
                      st.session_state.history.append(Message("human", human_prompt))
                      st.session_state.history.append(Message("ai", ai_response))
                      st.session_state.human_prompt = ""  # Clear the input field
                      return  # Stop further processing until context is set
    
              # Generate AI response based on the context
              if st.session_state.context == "personal":
                  ai_response = give_personal_advice(human_prompt)
              elif st.session_state.context == "company":
                  ai_response = give_company_advice(human_prompt)
    
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

    uploaded_file = st.sidebar.file_uploader("Upload CSV file (Company Stock Data)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Ensure required columns exist
        required_columns = {'Date', 'Close/Last'}
        if not required_columns.issubset(df.columns):
            st.error(f"CSV must contain these columns: {', '.join(required_columns)}")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            st.write("### Uploaded Company Data")
            st.dataframe(df.head())

            # Plot stock prices
            st.write("### Stock Closing Price Over Time")
            plt.figure(figsize=(10,5))
            plt.plot(df.index, df['Close'], label="Closing Price", color='blue')
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            plt.title('Stock Closing Prices')
            plt.legend()
            st.pyplot()

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




def stocks_user_page():
    st.title('📈 Multicompany Stock Analysis & Prediction')

    st.markdown("""
    This app allows users to:
    - Select multiple stock symbols
    - View closing price plots
    - Get **LSTM-based stock predictions** for selected companies.
    """)

    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file (with 'Symbol' column)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Symbol' not in df.columns:
            st.error("❌ CSV must contain a 'Symbol' column")
        else:
            st.write("### Uploaded Company List")
            st.dataframe(df)

            selected_stocks = st.sidebar.multiselect('✅ Select Stocks', df['Symbol'], df['Symbol'])

            if selected_stocks:
                data = yf.download(
                    tickers=selected_stocks,
                    period="1y",  
                    interval="1d",
                    group_by='ticker',
                    auto_adjust=True,
                    prepost=True,
                    threads=True
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

                    model = build_lstm_model(input_shape=(x_train.shape[1], 1))
                    trained_model, history = train_model(model, x_train, y_train, x_test, y_test)

                    predictions = predict_stock(trained_model, scaler, df_test)

                    return df_test.index, df_test['Close'].values, predictions

                if st.button('📈 Show Predictions'):
                    st.header('🔮 Stock Price Predictions')
                    for i in selected_stocks[:num_company]:
                        dates, actual, predicted = get_predictions(i)
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
    st.title("Tracker")
    st.write("Welcome to the Tracker page!")
    # Add your tracker code here

def news_page():
    st.title("News")
    st.write("Welcome to the News page!")
    # Add your news code here

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Chatbot"

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
with st.sidebar:
    st.title("Navigation")
    if st.button("🗨️ Chatbot"):
        st.session_state.page = "Chatbot"
    if st.button("📈 Stocks (Company)"):
        st.session_state.page = "Company_Stocks"
    if st.button("📈 Stocks (User)"):
        st.session_state.page = "User_Stocks"
    if st.button("📊 Tracker"):
        st.session_state.page = "Tracker"
    if st.button("📰 News"):
        st.session_state.page = "News"

# Display the selected page
if st.session_state.page == "Chatbot":
    chatbot_page()
elif st.session_state.page == "Company_Stocks":
    stocks_company_page()
elif st.session_state.page == "User_Stocks":
    stocks_user_page()
elif st.session_state.page == "Tracker":
    tracker_page()
elif st.session_state.page == "News":
    news_page()



    