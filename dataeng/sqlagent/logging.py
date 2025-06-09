from rich.console import Console
from rich.panel import Panel
from rich.style import Style

blue_border_style = Style(color="#0EA5E9")
green_border_style = Style(color="#10B981", bold=True)
console = Console()


def log_panel(title: str, content: str, border_style: Style = blue_border_style):
    """
    Log a panel with a title and content to the console.

    Args:
        title (str): The title of the panel.
        content (str): The content to display in the panel.
        border_style (Style): The style for the panel border.
    """

    console.log(
        Panel(
        content=content,
        title=title,
        border_style=border_style,
    )
    )

def log(content: str):
    """
    Log a message to the console.

    Args:
        content (str): The message to log.
    """
    console.log(content)

