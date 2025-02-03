import sys
from pathlib import Path
from functools import wraps
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console
from loguru import logger

# Configure loguru logger
def setup_logger():
    # Remove default logger
    logger.remove()
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Add file logger with rotation
    log_file = log_dir / "app.log"
    logger.add(
        log_file,
        rotation="500 MB",
        retention="10 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Add stdout logger with rich formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )

def create_rich_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=Console()
    )

def task_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        task_name = func.__name__
        
        logger.info(f"Starting {task_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Successfully completed {task_name}")
            logger.info(f"Time taken: {datetime.now() - start_time}")
            return result
        except Exception as e:
            logger.error(f"Error in {task_name}: {str(e)}")
            logger.exception(e)
            raise
    return wrapper 