import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(console=console, show_time=False, show_path=False)],
)

logger = logging.getLogger('movie_translator')
