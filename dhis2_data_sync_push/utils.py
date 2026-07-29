import fnmatch
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
import requests
from dateutil.relativedelta import relativedelta
from openhexa.sdk import current_run, workspace
from openhexa.sdk.datasets.dataset import DatasetVersion
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.periods import period_from_string


def connect_to_dhis2(connection_str: str, cache_dir: Path | None = None) -> DHIS2:
    """Establishes a connection to DHIS2 using the provided connection string and cache directory.

    Parameters
    ----------
    connection_str : str
        The connection string for DHIS2.
    cache_dir : Path
        The directory to use for caching DHIS2 data.

    Returns
    -------
    DHIS2
        An instance of the DHIS2 client.

    Raises
    ------
    Exception
        If there is an error while connecting to DHIS2.
    """
    try:
        connection = workspace.dhis2_connection(connection_str)
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        dhis2_client = DHIS2(connection=connection, cache_dir=cache_dir)
        current_run.log_info(f"Connected to DHIS2 connection: {connection.url}")
        return dhis2_client
    except Exception as e:
        raise Exception(f"Error while connecting to DHIS2 {connection_str}: {e}") from e


def load_configuration(config_path: Path) -> dict:
    """Reads a JSON file configuration and returns its contents as a dictionary.

    Args:
        config_path (str): Root path of the pipeline to find the file.

    Returns:
        dict: Dictionary containing the JSON data.
    """
    try:
        with Path.open(config_path, "r") as file:
            data = json.load(file)

        current_run.log_info(f"Configuration loaded from {config_path}.")
        return data
    except FileNotFoundError as e:
        raise Exception(f"The file '{config_path}' was not found {e}") from e
    except json.JSONDecodeError as e:
        raise Exception(f"Error decoding JSON: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error while loading configuration '{config_path}' {e}") from e


def retrieve_ou_list(dhis2_client: DHIS2, ou_level: int) -> list:
    """Retrieve a list of organisational unit IDs from DHIS2 filtered by the specified organisational unit level.

    Parameters
    ----------
    dhis2_client : DHIS2
        An instance of the DHIS2 client.
    ou_level : int
        The organisational unit level to filter by.

    Returns
    -------
    list
        A list of organisational unit IDs matching the specified level.
    """
    try:
        # Retrieve organisational units and filter by ou_level
        ous = pd.DataFrame(dhis2_client.meta.organisation_units())
        ou_list = ous.loc[ous.level == ou_level].id.to_list()
        current_run.log_info(f"Retrieved DHIS2 org units id list {len(ou_list)} at level {ou_level}")
        return ou_list
    except Exception as e:
        raise Exception(f"Error while retrieving OU id list for level {ou_level}: {e}") from e


def select_descendants(df: pd.DataFrame, parent_ids: list[str]) -> pd.DataFrame:
    """Select all rows from a hierarchical DataFrame that are descendants of the given parent IDs.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing at least two columns: "id" and "parent_id". Each row represents
        a node in a hierarchy, where "parent_id" references the parent node's "id".
    parent_ids : list[str]
        A list of parent IDs for which to retrieve all descendant rows.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing the rows with IDs in the input `parent_ids` and all
        of their descendants.

    Notes
    -----
    - The returned DataFrame is a **copy**, not a view of the original `df`.
      Modifying it will not affect the input DataFrame.
    - Works for hierarchies of any depth.
    """
    # Use a set to accumulate all descendant IDs
    all_ids = set(parent_ids)
    new_children = set(parent_ids)

    # Iteratively find children
    while new_children:
        # Find rows where parent_id is in new_children
        children = df[df["parent_id"].isin(new_children)]
        # Get their IDs
        child_ids = set(children["id"])
        # Only keep the new ones
        new_children = child_ids - all_ids
        # Add to all_ids
        all_ids.update(new_children)

    # Filter DataFrame to include only the parent and all descendants
    return df[df["id"].isin(all_ids)]


def update_extract(
    new_data_df: pd.DataFrame,
    target_df: pd.DataFrame,
    key_columns: list | None = None,
) -> pd.DataFrame:
    """Updates the values in the target_df with matching values from new_data_df.

    - Existing rows are updated with values from new_data_df.
    - Rows only in new_data_df are appended.
    - Rows only in target_df are kept as-is, unless there's a match in new_data_df,
      in which case they are updated (even if the new value is NaN).

    Parameters
    ----------
    new_data_df : pd.DataFrame
        DataFrame containing new or updated data.
    target_df : pd.DataFrame
        DataFrame to be updated.
    key_columns : list | None, optional
        Columns to merge on. Defaults to standard key columns if None.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame.

    Raises
    ------
    ValueError
        If any key columns are missing from either DataFrame.
    """
    if not key_columns:
        key_columns = [
            "data_type",
            "dx_uid",
            "period",
            "org_unit",
            "category_option_combo",
            "attribute_option_combo",
            "rate_type",
            "domain_type",
        ]

    # Validate key columns exist in both DataFrames
    for df, name in [(new_data_df, "new_data_df"), (target_df, "target_df")]:
        missing = [c for c in key_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Key columns {missing} not found in {name}.")

    # Mark rows coming from new_data_df
    new_data_df = new_data_df.copy()
    new_data_df["_from_new"] = True

    updated_df = target_df.merge(
        new_data_df,
        on=key_columns,
        how="outer",
        suffixes=("_old", ""),
    )

    # Only restore old value if the row has NO match in new_data_df at all
    if "value_old" in updated_df.columns:
        no_match = updated_df["_from_new"].isna()
        updated_df.loc[no_match, "value"] = updated_df.loc[no_match, "value_old"]

    # Always clean up sentinel and old value columns
    cols_to_drop = [c for c in ["value_old", "_from_new"] if c in updated_df.columns]
    if cols_to_drop:
        updated_df = updated_df.drop(columns=cols_to_drop)

    return updated_df


def merge_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame | None:
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


def save_to_parquet(data: pd.DataFrame | pl.DataFrame, filename: Path) -> None:
    """Safely saves a pandas or polars DataFrame to a Parquet file using a temporary file and atomic replace.

    Args:
        data (pd.DataFrame | pl.DataFrame): The DataFrame to save.
        filename (Path): The path where the Parquet file will be saved.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    def write_fn(path: Path) -> None:
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, engine="pyarrow", index=False)
        elif isinstance(data, pl.DataFrame):
            data.write_parquet(path)
        else:
            raise TypeError(f"Unsupported DataFrame type: {type(data)}. Expected pandas or polars DataFrame.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", dir=filename.parent, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        write_fn(tmp_path)
        tmp_path.replace(filename)

    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        raise


def read_parquet_extract(parquet_file: Path) -> pd.DataFrame:
    """Reads a Parquet file and returns its contents as a pandas DataFrame.

    Parameters
    ----------
    parquet_file : Path
        The path to the Parquet file to be read.

    Returns
    -------
    pd.DataFrame
        The contents of the Parquet file as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    pd.errors.EmptyDataError
        If the Parquet file is empty.
    Exception
        For any other unexpected errors during reading.
    """
    try:
        ou_source = pd.read_parquet(parquet_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error while loading the extract: File was not found {parquet_file}.") from None
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Error while loading the extract: File is empty {parquet_file}.") from None
    except Exception as e:
        raise RuntimeError(f"Error while loading the extract: {parquet_file}. Error: {e}") from None

    return ou_source


def configure_logging(task_name: str, logs_path: Path = Path("/home/jovyan/tmp/logs")) -> tuple[logging.Logger, Path]:
    """Set up a logger for a specific task, with immediate flush behavior.

    Returns
    -------
    tuple[logging.Logger, Path]
        A tuple containing the configured logger and the path to the log file.
    """

    class HandlerThatAlwaysFlushes(logging.FileHandler):
        def emit(self, record: logging.LogRecord) -> None:
            super().emit(record)
            self.flush()
            if self.stream and not self.stream.closed:
                os.fsync(self.stream.fileno())

    # Ensure logs directory exists
    logs_path.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d-%H_%M")
    log_file = logs_path / f"{task_name}_{now}.log"

    # Create or get logger
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = HandlerThatAlwaysFlushes(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger, log_file


def save_logs(logs_file: Path, output_dir: Path) -> None:
    """Moves all .log files from logs_path to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if logs_file.is_file():
        dest_file = output_dir / logs_file.name
        shutil.copy(logs_file.as_posix(), dest_file.as_posix())


def read_json_file(file_path: Path) -> dict:
    """Reads a JSON file and handles potential errors.

    Args:
        file_path (Path): The path to the JSON file.

    Returns:
        dict: Parsed JSON data if successful.
    """
    try:
        with Path.open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.") from None
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to decode JSON : '{file_path}'. Details: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error while reading the file '{file_path}': {e}") from e


def is_valid_yyyymm(date: str) -> bool:
    """Validates if the provided string is in YYYYMM format and represents a valid month and year.

    Returns
    -------
    bool
        True if the date is valid, False otherwise.
    """
    if not re.match(r"^\d{6}$", date):
        return False
    year = int(date[:4])
    month = int(date[4:])
    return 2000 <= year <= 2100 and 1 <= month <= 12


def is_after_today(yyyymm: str) -> bool:
    """Checks if the provided YYYYMM date string represents a month after the current month.

    Returns
    -------
    bool
        True if the provided date is after the current month, False otherwise.
    """
    try:
        date = datetime.strptime(yyyymm, "%Y%m")
    except ValueError:
        return False  # Invalid format
    now = datetime.now()
    current_yyyymm = now.year * 100 + now.month
    input_yyyymm = date.year * 100 + date.month
    return input_yyyymm > current_yyyymm


def adjust_to_previous_month_if_current(date_str: str) -> str:
    """If the provided date_str is the current month, adjust it to the previous month. Otherwise, return it unchanged.

    Returns
    -------
    str
        Adjusted date string in YYYYMM format.
    """
    if date_str is None:
        return None
    date_obj = datetime.strptime(date_str, "%Y%m")
    now = datetime.now()
    current_yyyymm = now.strftime("%Y%m")
    if date_str == current_yyyymm:
        prev_month = date_obj - relativedelta(months=1)
        prev_month_str = prev_month.strftime("%Y%m")
        current_run.log_info(
            f"Adjusting current to previous month to avoid empty data request: {date_str} -> {prev_month_str}"
        )
        return prev_month_str
    return date_str


def resolve_dates_and_validate(start_date: str, end_date: str, config: dict) -> tuple[str | None, str | None]:
    """Resolves and validates start and end dates for data extraction.

    Returns
    -------
    tuple[str | None, str | None]
        Resolved and validated start and end dates.
    """
    months_lag = config["SETTINGS"].get("NUMBER_MONTHS_WINDOW", 3)  # default 3 months window

    # start date resolution and validation
    if start_date:
        start_result = resolve_user_provided_date(start_date)
    else:
        current_run.log_info("No start date provided, using setting defaults.")
        try:
            if not config["SETTINGS"]["STARTDATE"]:
                start_result = (datetime.now() - relativedelta(months=months_lag)).strftime("%Y%m")
            else:
                start_result = config["SETTINGS"]["STARTDATE"]
        except Exception as e:
            raise Exception(f"Error in start/end date configuration: {e}") from e

    # end date resolution and validation
    if end_date:
        end_result = resolve_user_provided_date(end_date)
    else:
        current_run.log_info("No end date provided, using setting defaults.")
        try:
            if not config["SETTINGS"]["ENDDATE"]:
                end_result = (datetime.now() - relativedelta(months=1)).strftime("%Y%m")  # go back 1 month.
            else:
                end_result = config["SETTINGS"]["ENDDATE"]
        except Exception as e:
            raise Exception(f"Error in start/end date configuration: {e}") from e

    # Date validations
    if start_date and end_date and start_date > end_date:
        raise ValueError(f"Start date {start_date} cannot be after end date {end_date}.")

    if start_result < "201701":
        current_run.log_warning(f"Start date {start_result} cannot be before 201701. Defaulting to 201701.")
        start_result = "201701"

    if end_result < "201701":
        current_run.log_warning(f"End date {end_result} cannot be before 201701. Defaulting to 201701.")
        end_result = "201701"

    return start_result, end_result


def get_extract_periods(start: str, end: str) -> list[str]:
    """Generates a list of periods between start and end in YYYYMM format.

    Returns
    -------
    list[str]
        List of periods in YYYYMM format.
    """
    try:
        # Get periods
        start_period = period_from_string(start)
        end_period = period_from_string(end)
        extract_periods = (
            [str(p) for p in start_period.get_range(end_period)]
            if str(start_period) < str(end_period)
            else [str(start_period)]
        )
    except Exception as e:
        raise Exception(f"Error in start/end date configuration: {e!s}") from e
    return extract_periods


def resolve_user_provided_date(date: str) -> str:
    """Resolves and validates user-provided date.

    Returns
    -------
    str:
        Resolved and validated start and end dates.
    """
    if not is_valid_yyyymm(date):
        raise ValueError(f"Invalid date format: {date}. Expected YYYYMM ([2000/2100][01/12]).")

    if is_after_today(date):
        raise ValueError(f"Date cannot be in the future. Provided date: {date}.")

    return adjust_to_previous_month_if_current(date)


def add_files_to_dataset(
    dataset_id: str,
    file_paths: list[Path],
    ds_version_prefix: str = "DS",
    ds_desc: str = "Dataset version created by pipeline",
) -> bool:
    """Add files to a new dataset version.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to which files will be added.
    file_paths : list[Path]
        A list of file paths to be added to the dataset.
    ds_version_prefix : str, optional
        The prefix for the dataset version name. Default is "DS".
    ds_desc : str, optional
        The description for the dataset version. Default is "Dataset version created by pipeline".

    Returns
    -------
    bool
        True if at least one file was added successfully, False otherwise.

    Raises
    ------
    ValueError
        If the dataset ID is not specified.
    """
    if not dataset_id:
        raise ValueError("Dataset ID is not specified.")

    supported_extensions = {".parquet", ".csv", ".geojson", ".json"}
    added_any = False
    new_version = None

    for src in file_paths:
        if not src.exists():
            current_run.log_warning(f"File not found: {src}")
            continue

        ext = src.suffix.lower()
        if ext not in supported_extensions:
            current_run.log_warning(f"Unsupported file format: {src.name}")
            continue

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp_path = Path(tmp.name)

            shutil.copy2(src, tmp_path)

            if not added_any:
                new_version = get_new_dataset_version(
                    ds_id=dataset_id,
                    prefix=ds_version_prefix,
                    ds_desc=ds_desc,
                )
                current_run.log_info(f"New dataset version created: {new_version.name}")
                added_any = True

            new_version.add_file(str(tmp_path), filename=src.name)
            current_run.log_info(f"File {src.name} added to dataset version: {new_version.name}")

        except Exception as e:
            current_run.log_warning(f"File {src.name} cannot be added: {e}")

        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    if not added_any:
        current_run.log_warning("No valid files found. Dataset version was not created.")
        return False

    return True


def get_new_dataset_version(ds_id: str, prefix: str = "DS", ds_desc: str = "Dataset") -> DatasetVersion:
    """Create and return a new dataset version.

    Parameters
    ----------
    ds_id : str
        The ID of the dataset for which a new version will be created.
    prefix : str, optional
        Prefix for the dataset version name (default is "DS").
    ds_desc : str, optional
        Description for the dataset (default is "Dataset version created by pipeline").

    Returns
    -------
    DatasetVersion
        The newly created dataset version.

    Raises
    ------
    Exception
        If an error occurs while creating the new dataset version.
    """
    try:
        dataset = workspace.get_dataset(ds_id)
    except Exception as e:
        current_run.log_warning(f"Error retrieving dataset: {ds_id}")
        current_run.log_debug(f"Error retrieving dataset {ds_id}: {e}")
        dataset = None

    if dataset is None:
        current_run.log_warning(f"Creating new Dataset with ID: {ds_id}")
        dataset = workspace.create_dataset(name=ds_id.replace("-", "_").upper(), description=ds_desc)

    version_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    try:
        return dataset.create_version(version_name)
    except Exception as e:
        raise Exception("An error occurred while creating the new dataset version.") from e


def save_json_file(file_path: Path, contents: dict) -> None:
    """Save the last update timestamp to a JSON file."""
    try:
        with Path.open(file_path, "w") as f:
            json.dump(contents, f)
    except Exception as e:
        raise OSError(f"Error saving last update timestamp to file: {e}") from e


def get_file_from_dataset(dataset_id: str, filename: str) -> pd.DataFrame | gpd.GeoDataFrame | dict:
    """Get a file from a dataset.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset.
    filename : str
        The name of the file to retrieve.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame | dict
        The DataFrame, GeoDataFrame or dict containing the data.
    """
    dataset = workspace.get_dataset(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset with ID {dataset_id} not found.")

    version = dataset.latest_version
    if not version:
        raise ValueError(f"No versions found for dataset {dataset_id}.")

    file_path = version.get_file(filename)
    if not file_path:
        raise ValueError(f"File {filename} not found in dataset {dataset_id}.")

    suffix = Path(filename).suffix.lower()
    url = file_path.download_url
    r = requests.get(url)

    if r.status_code != 200:
        raise ValueError(f"Failed to download file: {r.status_code} - {r.text}")

    if len(r.content) < 100:
        raise ValueError(f"Downloaded file is suspiciously small ({len(r.content)} bytes)")

    if suffix in [".csv", ".parquet", ".geojson", ".gpkg", ".shp", ".json"]:
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tfile:
                tfile.write(r.content)
                tfile.flush()
                tfile_path = tfile.name
            if suffix == ".csv":
                return pd.read_csv(tfile_path)
            if suffix == ".parquet":
                return pd.read_parquet(tfile_path)
            if suffix == ".json":
                with Path(tfile_path).open(encoding="utf-8") as f:
                    return json.load(f)
            return gpd.read_file(tfile_path)
        finally:
            if tfile_path and Path(tfile_path).exists():
                Path(tfile_path).unlink()

    raise ValueError(f"Unsupported file type: {suffix}")


def get_matching_filenames_from_dataset(dataset_id: str, pattern: str) -> list[str]:
    """Get the filename from openhexa dataset last version that matches the pattern.

    Returns
    -------
    str
        The filename that matches the pattern, if not found returns None.
    """
    dataset = workspace.get_dataset(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset with ID {dataset_id} not found.")

    version = dataset.latest_version
    if not version:
        raise ValueError(f"No versions found for dataset {dataset_id}.")

    return [file.filename for file in version.files if fnmatch.fnmatch(file.filename, pattern)]
