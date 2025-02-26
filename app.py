from dataclasses import dataclass
from typing import Literal
import streamlit as st
from models import give_personal_advice, give_company_advice  
import streamlit.components.v1 as components

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

  def on_click_callback():
      human_prompt = st.session_state.human_prompt
      if human_prompt.strip():  # Check if the input is not empty
          # Call your model functions here
          if "personal" in human_prompt.lower():
              ai_response = give_personal_advice(human_prompt)
          elif "company" in human_prompt.lower():
              ai_response = give_company_advice(human_prompt)
          else:
              ai_response = "Please specify whether you want personal or company advice."

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



def stocks_page():
    st.title("Stocks")
    st.write("Welcome to the Stocks page!")
    # Add your stocks code here

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

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Chatbot", "Stocks", "Tracker", "News"))

# Update session state based on sidebar selection
st.session_state.page = page

# Display the selected page
if st.session_state.page == "Chatbot":
    chatbot_page()
elif st.session_state.page == "Stocks":
    stocks_page()
elif st.session_state.page == "Tracker":
    tracker_page()
elif st.session_state.page == "News":
    news_page()

st.sidebar.markdown("""
- 🗨️ **Chatbot**
- 📈 **Stocks**
- 📊 **Tracker**
- 📰 **News**
""")