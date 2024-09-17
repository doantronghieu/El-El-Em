import sys
import os
from dotenv import load_dotenv
from loguru import logger

def find_apps_dir_and_load_env():
    current_dir = os.path.abspath('')
    env_files_loaded = []

    while True:
        # Check for and load .env file in the current directory
        env_file = os.path.join(current_dir, '.env')
        if os.path.exists(env_file):
            load_dotenv(env_file)
            env_files_loaded.append(env_file)

        # Check if we've reached the 'Apps' directory
        if os.path.basename(current_dir) == 'Apps':
            break

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # reached root without finding 'Apps'
            raise Exception("'Apps' directory not found in parent hierarchy")
        current_dir = parent_dir

    return current_dir, env_files_loaded

def setup_paths_and_env():
    apps_dir, env_files = find_apps_dir_and_load_env()
    
    # Add Apps directory to sys.path
    if apps_dir not in sys.path:
        sys.path.insert(0, apps_dir)
    
    # Add toolkit to sys.path
    toolkit_path = os.path.join(apps_dir, 'toolkit')
    if os.path.exists(toolkit_path) and toolkit_path not in sys.path:
        sys.path.insert(0, toolkit_path)
    else:
        logger.warning(f"Warning: 'toolkit' directory not found in {apps_dir}")

    return {
        'apps_dir': apps_dir,
        'toolkit_path': toolkit_path,
        'env_files_loaded': env_files
    }

# Run the setup
setup_result = setup_paths_and_env()

# Print out the results
logger.info(f"Apps directory: {setup_result['apps_dir']}")
logger.info(f"Toolkit path: {setup_result['toolkit_path']}")
logger.info("Environment files loaded:")
for env_file in setup_result['env_files_loaded']:
    logger.info(f"  - {env_file}")

# logger.info("Updated sys.path:")
# for path in sys.path:
#     logger.info(f"  - {path}")

# Define APP_PATH for compatibility with your original script
APP_PATH = setup_result['apps_dir']
