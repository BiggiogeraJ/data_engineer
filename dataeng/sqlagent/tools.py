import sqlite3
from contextlib import contextmanager
from typing import Any, List

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool

from dataeng.sqlagent.config import Config
from dataeng.sqlagent.logging import log_panel, log

def get_available_tools() -> List[BaseTool]:
    """
    Get a list of available tools for the SQL agent.
    
    Returns:
        List[BaseTool]: A list of tools that can be used by the SQL agent.
    """
    return [list_tables, sample_table, describe_table, execute_sql]

def call_tool(tool_call: ToolCall) -> Any:
    """
    Call a tool based on the provided tool call.
    
    Args:
        tool_call (ToolCall): The tool call to execute.
        
    Returns:
        ToolMessage: The result of the tool call.
    """
    tools_by_name = {tool.name: tool for tool in get_available_tools()}
    tool = tools_by_name[tool_call['name']]
    response = tool.invoke(tool_call['args'])
    return ToolMessage(content=response, tool_call_id = tool_call["id"])

@contextmanager
def with_sql_cursor(readonly = True):
    """
    Context manager to handle SQLite database connections and cursors.
    
    Args:
        readonly (bool): If True, opens the connection in read-only mode.
        
    Yields:
        sqlite3.Cursor: A cursor for executing SQL commands.
    """
    conn = sqlite3.connect(Config.Paths.DATABASE_PATH)
    cur = conn.cursor()
    try:
        yield cur
        if not readonly:
            conn.commit()
    except Exception:
        if not readonly:
            conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()

@tool(parse_docstring=True)
def list_tables(reasoning: str) -> str:
    """
    List all tables in the database.

    Args:
        reasoning (str): Detailed explanation of why you need to see all tables (relate to the user's query).
    
    Returns:
        str: A formatted string listing all tables in the database.
    """

    log_panel(title="List Tables Tool", content=f"Reasoning: {reasoning}",)

    try:
        with with_sql_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT like 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]
        return str(tables)
    except Exception as e:
        log(f"[red]Error listing tables: {str(e)}[/red]")
        return f"Error listing tables: {str(e)}"
    
@tool(parse_docstring=True)
def sample_table(reasoning: str, table_name:str, row_sample_size: int) -> str:
    """
    Retrieve a small sample of rows to understand the data structure and content of a specific table.
    
    Args:
        reasoning (str): Detailed explanation of why you need to sample this table (relate to the user's query).
        table_name (str): Exact name of the table to sample (case-sensitive, no quotes needed).
        row_sample_size (int): Number of rows to retrieve (reccomended: 3-5 rows for readibility).

    Returns:
        str: String with one row per line, showing all columns for each row as tuples.
    """

    log_panel(title="Sample Table Tool", content=f"Table: {table_name}\nRows: {row_sample_size}\nReasoning: {reasoning}")

    try:
        with with_sql_cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {row_sample_size};")
            rows = cursor.fetchall()
        return "\n".join([str(row) for row in rows])
    except Exception as e:
        log(f"[red]Error sampling table: {str(e)}[/red]")
        return f"Error sampling table: {str(e)}"
    
@tool(parse_docstring=True)
def describe_table(reasoning: str, table_name: str) -> str:
    """
    Returns detailed schema information about a table (columns, types, constraints).
    
    Args:
        reasoning (str): Detailed explanation of why you need to understand this table's structure (relate to the user's query).
        table_name (str): Exact name of the table to describe (case-sensitive, no quotes needed).
    
    Returns:
        str: String containing table schema information.
    """

    log_panel(title="Describe Table Tool", content=f"Table: {table_name}\nReasoning: {reasoning}")

    try:
        with with_sql_cursor() as cursor:
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            rows = cursor.fetchall()
        return "\n".join([str(row) for row in rows])
    except Exception as e:
        log(f"[red]Error describing table: {str(e)}[/red]")
        return f"Error describing table: {str(e)}"
    

@tool(parse_docstring=True)
def execute_sql(reasoning: str, sql_query: str) -> str:
    """
    Execute a custom SQL query and return the results.
    
    Args:
        reasoning (str): Detailed explanation of why you need to execute this query (relate to the user's query).
        sql_query (str): Complete, properly formatted SQL query (must be a valid SQL statement).
    
    Returns:
        str: String containing the results of the executed SQL query.
    """

    log_panel(title="Execute SQL Tool", content=f"SQL Query: {sql_query}\nReasoning: {reasoning}")

    try:
        with with_sql_cursor(readonly=False) as cursor:
            cursor.execute(sql_query)
            rows = cursor.fetchall()
        return "\n".join([str(row) for row in rows])
    except Exception as e:
        log(f"[red]Error running query: {str(e)}[/red]")
        return f"Error running query: {str(e)}"
