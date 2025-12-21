import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()

logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s',
    handlers=[RichHandler(console=console, show_time=False, show_path=False)],
)

logger = logging.getLogger('movie_translator')


def set_verbose(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logging.getLogger().setLevel(level)
