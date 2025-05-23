import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta


def get_periods_yyyymm(start_period: str, end_period: str) -> list:
    """Generate all monthly periods between start_period and end_period (inclusive).

    Args:
        start_period (str): The start period in YYYYMM format.
        end_period (str): The end period in YYYYMM format.

    Returns:
        list: A list of monthly periods in YYYYMM format.
    """
    # Convert input strings to datetime objects
    start_date = datetime.strptime(start_period, "%Y%m")
    end_date = datetime.strptime(end_period, "%Y%m")

    # Check that start_date is not after end_date
    if start_date > end_date:
        raise ValueError("start_period must not be after end_period")

    # Generate periods
    periods = []
    current_date = start_date
    while current_date <= end_date:
        periods.append(current_date.strftime("%Y%m"))
        next_month = current_date.month % 12 + 1
        next_year = current_date.year + (current_date.month // 12)
        current_date = current_date.replace(year=next_year, month=next_month)

    return periods


def merge_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge a list of dataframes, excluding None values.

    Assume they shared the same columns.

    Args:
        dataframes (list[pd.DataFrame]): A list of dataframes to merge.

    Returns:
        pd.DataFrame: Concatenated dataframe, or None if all inputs are None.
    """
    # Filter out None values from the list
    not_none_df = [df for df in dataframes if df is not None]

    # Check if all columns match
    if len(not_none_df) > 1:
        first_columns = set(not_none_df[0].columns)
        for df in not_none_df[1:]:
            if set(df.columns) != first_columns:
                raise ValueError("DataFrames have mismatched columns and cannot be concatenated.")

    # Concatenate if there are valid dataframes, else return None
    return pd.concat(not_none_df) if not_none_df else None


def first_day_of_future_month(date: str, months_to_add: int) -> str:
    """Compute the first day of the month after adding a given number of months.

    Args:
        date (str): A date in the "YYYYMM" format.
        months_to_add (int): Number of months to add.

    Returns:
        str: The resulting date in "YYYY-MM-DD" format.
    """
    # Parse the input date string
    input_date = datetime.strptime(date, "%Y%m")
    target_date = input_date + relativedelta(months=months_to_add)

    return target_date.strftime("%Y-%m-01")


def save_to_parquet(data: pd.DataFrame, filename: Path):
    """Safely saves a DataFrame to a Parquet file.

    Args:
        data (pd.DataFrame): The DataFrame to save.
        filename (str): The path where the Parquet file will be saved.
    """
    try:
        # Ensure the data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The 'data' parameter must be a pandas DataFrame.")

        # Create directories if necessary
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Save the DataFrame as a Parquet file
        data.to_parquet(
            filename,
            engine="pyarrow",
            index=False,
        )

    except Exception as e:
        raise Exception(f"An unexpected error occurred while saving the parquet file: {e}") from e


def load_files_with_pattern(path: Path, pattern: str) -> pd.DataFrame:
    """Load multiple Parquet files matching a pattern from a given directory and merge them into a single DataFrame.

    Args:
        path (Path): The directory path to search for files.
        pattern (str): The glob pattern to match files.

    Returns:
        pd.DataFrame: The merged DataFrame containing data from all matched files.
    """
    dfs = [read_parquet_extract(file) for file in path.glob(pattern)]
    return merge_dataframes(dfs)


def read_parquet_extract(parquet_file: str) -> pd.DataFrame:
    """Reads a Parquet file and returns its contents as a pandas DataFrame.

    Args:
        parquet_file (str): Path to the Parquet file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the Parquet file.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        Exception: For any other errors encountered while reading the file.
    """
    ou_source = pd.DataFrame()
    try:
        ou_source = pd.read_parquet(parquet_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error while loading the extract: File was not found {parquet_file}.") from e
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Error while loading extract: empty file {parquet_file}.") from e
    except Exception as e:
        raise Exception(f"Error while loading the extract: {parquet_file}. Error: {e}") from e

    return ou_source


def read_json_file(file_path: Path) -> dict | list:
    """Reads a JSON file and handles potential errors.

    Args:
        file_path (Path): The path to the JSON file.

    Returns:
        dict or list: Parsed JSON data if successful.
        None: If an error occurs.
    """
    try:
        with Path.open(file_path, "r") as file:
            return json.load(file)
        # return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file '{file_path}' was not found.") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Failed to decode JSON file '{file_path}'. Details: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error while reading the file '{file_path}': {e}") from e


def split_list(src_list: list, length: int) -> Iterator[list]:
    """Split list into chunks.

    Args:
        src_list (list): The list to split.
        length (int): The length of each chunk.

    Yields:
        list: Chunks of the original list of the specified length.
    """
    for i in range(0, len(src_list), length):
        yield src_list[i : i + length]


# my queue class
class Queue:
    """A persistent FIFO queue backed by an SQLite database for storing and managing periods.

    Methods
    -------
    enqueue(period: str)
        Add a new period to the queue if it does not already exist.
    dequeue()
        Remove and return the first period in the queue.
    peek()
        Return the first period in the queue without removing it.
    count() -> int
        Return the number of items in the queue.
    reset()
        Clear all contents from the queue and reset the indexing.
    view_queue()
        Print all elements in the queue without removing them.
    count_queue_items() -> int
        Count the number of items in the queue.
    """

    def __init__(self, db_path: str):
        """Initialize the queue with the given SQLite database path."""
        self.db_path = db_path
        self._initialize_queue()

    def _initialize_queue(self):
        """Create the queue table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period TEXT NOT NULL
                )
            """)
            conn.commit()

    def enqueue(self, period: str):
        """Add a new period to the queue only if it does not already exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM queue WHERE period = ?", (period,))
            exists = cursor.fetchone()[0]

            if not exists:  # Insert only if the period does not exist
                cursor.execute("INSERT INTO queue (period) VALUES (?)", (period,))
                conn.commit()

    def dequeue(self) -> str | None:
        """Remove and return the first period in the queue.

        Returns:
            str or None: The first period in the queue, or None if the queue is empty.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, period FROM queue ORDER BY id LIMIT 1")
            item = cursor.fetchone()
            if item:
                cursor.execute("DELETE FROM queue WHERE id = ?", (item[0],))
                conn.commit()
                return item[1]
        return None

    def peek(self) -> str | None:
        """Return the first period in the queue without removing it.

        Returns:
            str or None: The first period in the queue, or None if the queue is empty.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT period FROM queue ORDER BY id LIMIT 1")
            item = cursor.fetchone()
            return item[0] if item else None

    def count(self) -> int:
        """Return the number of items in the queue.

        Returns:
            int: The number of items in the queue.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM queue")
            return cursor.fetchone()[0]

    def reset(self):
        """Clear all contents from the queue and reset the indexing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS queue;")  # Drop table to reset indexing
            conn.commit()
            self._initialize_queue()  # Recreate table

    def view_queue(self):
        """View all elements in the queue without removing them."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, period FROM queue ORDER BY id")
            items = cursor.fetchall()
            if items:
                print("Queue contents:")
                for item in items:
                    print(f"ID: {item[0]}, Period: {item[1]}")
            else:
                print("The queue is empty.")

    def count_queue_items(self) -> int:
        """Count the number of items in the queue.

        Returns:
            int: The number of items in the queue.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM queue")
        return cursor.fetchone()[0]
