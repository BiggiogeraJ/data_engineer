from datetime import datetime
from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from dataeng.sqlagent.logging import green_border_style, log_panel
from dataeng.sqlagent.tools import call_tool

SYSTEM_PROMPT = f"""
You are a master database engineer with exceptional expertise in SQL and POstgreSQL query construction and optimization.
Your purpose is to transform natural language requests into precise, efficient SQL queries that deliver exactly what the user needs.
<instructions>
    <instruction>Devise your own strategic plan to explore and understand the database befre constructing queries.</instruction>
    <instruction>Deterimne the most efficient sequence of database investigation steps based on the specific user request.</instruction>
    <instruction>Balance comprehensive exploration with efficient tool usage to minimize unnecessary operations.</instruction>
    <instruction>For every tool call, include a detailed reasoning parameter explaining your strategic operations.</instruction>
    <instruction>Be sure to specifiy every required parameter for each tool call.</instruction>
    <instruction>Only execute the inal SQL query when you've thoroughly validated its correctness and efficiency.</instruction>
</instructions>

Today is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Your responses should be formatted as Markdown. Prefer tables or lists for displaying data where appropriate.
Your target audience is data scientiss and analysts who may not be familiar with SQL syntax.
""".strip()

def create_history() -> List[BaseMessage]:
    """
    Create the initial message history for the SQL agent.
    
    Returns:
        List[BaseMessage]: A list containing the system prompt as the first message.
    """
    return [SystemMessage(content=SYSTEM_PROMPT)]

def ask(
        query: str, history: List[BaseMessage], llm: BaseChatModel, max_iterations: int = 10
) -> str:
    
    log_panel(title= "User request", content=f"Query: {query}", border_style=green_border_style)

    n_iteratios = 0
    messages = history.copy()
    messages.append(HumanMessage(content=query))

    while n_iteratios < max_iterations:
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content
        for tool_call in response.tool_calls:
            response = call_tool(tool_call)
            messages.append(response)
        n_iteratios += 1

    raise RuntimeError("Maximum iterations reached without a final response. Please try again with a different query.")
