from dataclasses import dataclass
from typing import Literal
import streamlit as st
from models import give_personal_advice, give_company_advice  


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

# def on_click_callback():
#     human_prompt = st.session_state.human_prompt

#     # Call your custom model functions based on the input
#     if "personal" in human_prompt.lower():
#         ai_response = give_personal_advice(human_prompt)  # Call personal advice function
#     elif "company" in human_prompt.lower():
#         ai_response = give_company_advice(human_prompt)  # Call company advice function
#     else:
#         ai_response = "Please specify whether you want personal or company advice."

#     # Append the conversation to the history
#     st.session_state.history.append(Message("human", human_prompt))
#     st.session_state.history.append(Message("ai", ai_response))
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