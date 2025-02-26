from dataclasses import dataclass
from typing import Literal
import streamlit as st
from models import give_personal_advice, give_company_advice  
import streamlit.components.v1 as components

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

    # Call your custom model functions based on the input
    if "personal" in human_prompt.lower():
        ai_response = give_personal_advice(human_prompt)  # Call personal advice function
    elif "company" in human_prompt.lower():
        ai_response = give_company_advice(human_prompt)  # Call company advice function
    else:
        ai_response = "Please specify whether you want personal or company advice."

    # Append the conversation to the history
    st.session_state.history.append(Message("human", human_prompt))
    st.session_state.history.append(Message("ai", ai_response))

# Load CSS and initialize session state
load_css()
initialize_session_state()

# UI Components
st.title("Ask Mahmoud 🤖")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row 
    {'' if chat.origin == 'ai' else 'row-reverse'}">
    <img class="chat-icon" src="./static/{
        'ai_icon.png' if chat.origin == 'ai' 
                      else 'user_icon.png'}"
         width=32 height=32>
    <div class="chat-bubble
    {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
        &#8203;{chat.message}
    </div>
</div>
        """
        st.markdown(div, unsafe_allow_html=True)
    
    for _ in range(3):
        st.markdown("")

with chat_placeholder:
    for chat in st.session_state.history:
        cols = st.columns([0.1, 0.9])  # Adjust column widths as needed
        with cols[0]:
            if chat.origin == "ai":
                st.image("static/ai_icon.png", width=32)
            else:
                st.image("static/user_icon.png", width=32)
        with cols[1]:
            st.markdown(
                f'<div class="chat-bubble {"ai-bubble" if chat.origin == "ai" else "human-bubble"}">'
                f'{chat.message}'
                f'</div>',
                unsafe_allow_html=True
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