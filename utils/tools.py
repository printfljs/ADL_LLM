import pandas as pd
from pathlib import Path
import re

def load_dataframes_from_files(folder_path, file_pattern='window_*.csv', raise_exception=False):
    """
    Loads CSV files matching a given pattern from a specified folder and returns a list of DataFrames.

    Args:
        folder_path (str): The path to the folder.
        file_pattern (str): The filename matching pattern, defaults to 'window_*.csv'.
        raise_exception (bool): If True, raises an exception on error; otherwise, prints a warning.

    Returns:
        list[pd.DataFrame]: A list of DataFrames.
    """
    dataframes = []
    folder = Path(folder_path)

    # Check if the folder exists
    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist.")
        return dataframes

    # Get a list of files matching the pattern
    files = list(folder.glob(file_pattern))

    # Check if there are any files in the folder
    if not files:
        print(f"Warning: No files matching pattern '{file_pattern}' found in folder {folder_path}.")
        return dataframes

    # Function to extract window number from filename
    def get_window_number(file_path):
        match = re.search(r'window_(\d+)\.csv', str(file_path))
        return int(match.group(1)) if match else 0

    # Sort files by window number
    sorted_files = sorted(files, key=get_window_number)

    # Load each file as a DataFrame
    for file_path in sorted_files:
        # Check if the path is a file
        if not file_path.is_file():
            print(f"Warning: {file_path} is not a file.")
            continue

        # Check if the file is a CSV file
        if file_path.suffix.lower() != '.csv':
            print(f"Warning: {file_path} is not a CSV file.")
            continue

        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
        except Exception as e:
            error_message = f"Error loading file {file_path}: {e}"
            if raise_exception:
                raise Exception(error_message)
            else:
                print(f"Warning: {error_message}")

    return dataframes