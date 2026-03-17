
import logging
import sys
from pathlib import Path

def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    format_str: str | None = None
) -> logging.Logger:
    if format_str is None:
        format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    logger = logging.getLogger("driftguard")
    
    if logger.handlers:
        # Prevent double logging via root logger propagation.
        logger.propagate = False
        return logger
        
    logger.setLevel(getattr(logging, level.upper()))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_str)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Keep logs on this logger only; otherwise Ray/root handlers can duplicate lines.
    logger.propagate = False
    
    return logger

def get_logger(name: str | None = None) -> logging.Logger:

    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            name = 'unknown'
    
    return logging.getLogger(f"driftguard.{name}")


def set_log_level(level: str):
    logging.getLogger("driftguard").setLevel(getattr(logging, level.upper()))

# setup_logging(level="DEBUG")  
setup_logging(level="INFO")  
