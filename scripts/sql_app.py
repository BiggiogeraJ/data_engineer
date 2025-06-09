import sys
sys.path.append("/Users/jacopo.biggiogera@igenius.ai/Desktop/GenAI_projects/data_engineer")

import random

import streamlit as st
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dataeng.sqlagent.agent import ask, create_history
from dataeng.sqlagent.config import Config
from dataeng.sqlagent.models import create_llm
from dataeng.sqlagent.tools import get_available_tools, with_sql_cursor

load_dotenv()

LOADING_MESSAGES = [
    "Thinking...",
    "Processing your request...",
    "Generating response...",
    "Fetching data...",
    "Analyzing information...",
    "Formulating answer...",
    "Please wait a moment...",
]

@st.cache_resource(show_spinner=False)
def get_model() -> BaseChatModel:
    """Initialize and return the LLM."""
    llm = create_llm(Config.MODEL)
    llm = llm.bind_tools(get_available_tools())
    return llm

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="SQL Agent",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

load_css("styles.css")

st.header("SQL Agent")
st.subheader("Ask questions about your SQL database")

with st.sidebar:
    st.write('#Database Information')  
    st.write(f"**File**: {Config.Path.DATABASE_PATH.relative_to(Config.Path.APP_HOME)}")
    db_size = Config.Path.DATABASE_PATH.stat().st_size/ (1024 * 1024)  # Size in MB
    st.write(f"**Size**: {db_size:.2f} MB")

    with with_sql_cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        st.write(f"**Tables:**")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            st.write(f"- {table} ({count} rows)")

if "messages" not in st.session_state:
    st.session_state.messages = create_history()

for message in st.session_state.messages:
    if type(message) is SystemMessage:
        continue
    is_user = type(message) is HumanMessage
    avatar = "user" if is_user else "assistant"
    with st.chat_message("user" if is_user else "assistant", avatar=avatar):
        st.markdown(message.content)

if prompt := st.chat_input("Ask a question about your SQL database:"):
    with st.chat_message("user", avatar="user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar="assistant"):
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")

        response = ask(
            prompt=prompt,
            model=get_model(),
            history=st.session_state.messages
        )
        message_placeholder.markdown(response)
        st.session_state.messages.append(AIMessage(content=response))