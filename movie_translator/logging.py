import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()

# Default to WARNING level (quiet mode)
# Use set_verbose(True) to enable INFO level logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s',
    handlers=[RichHandler(console=console, show_time=False, show_path=False)],
)

logger = logging.getLogger('movie_translator')


def set_verbose(verbose: bool) -> None:
    """Enable or disable verbose logging."""
    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
