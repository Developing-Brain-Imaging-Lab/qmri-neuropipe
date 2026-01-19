from rich.console import Console

# Shared console instance to ensure thread-safe and consistent output
# throughout the application (CLI, Logging, Progress bars)
console = Console()
