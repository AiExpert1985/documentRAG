# utils/paths.py
import os

def get_project_root() -> str:
    """Returns the absolute path to the project's root directory."""
    # This assumes this file is in project_root/utils/paths.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_log_file_path() -> str:
    """
    Creates the log directory if it doesn't exist and returns the full log file path.
    """
    project_root = get_project_root()
    log_dir = os.path.join(project_root, 'log')
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    return os.path.join(log_dir, 'rag_system.log')