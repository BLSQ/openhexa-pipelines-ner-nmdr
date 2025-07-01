import calendar
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import get_datasets
from openhexa.toolbox.dhis2.periods import Month, Week
from utils import (
    Queue,
    save_to_parquet,
)


@pipeline("dhis2_snis_extract", timeout=28800)
@parameter(
    "extract_orgunits",
    name="Extract Organisation Units",
    help="",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "extract_analytics",
    name="Extract analytics",
    help="",
    type=bool,
    default=True,
    required=False,
)
def dhis2_snis_extract(extract_orgunits: bool, extract_analytics: bool):
    """Simple pipeline to retrieve a monthly PNLP related data extract from DHIS2 SNIS instance."""
    # setup variables
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_snis_extract"
    try:
        # load config
        config = load_configuration(pipeline_path=pipeline_path)

        # Validate dates
        done_validate_period = validate_period_range(config)

        # connect to DHIS2 SNIS
        dhis2_client = connect_to_dhis2(config=config, cache_dir=workspace.files_path, done=done_validate_period)

        # retrieve pyramid for alignment
        done_pyramid = extract_snis_pyramid(
            pipeline_path=pipeline_path,
            dhis2_snis_client=dhis2_client,
            run_pipeline=extract_orgunits,
        )

        # retrieve data extracts
        extract_snis_data(
            pipeline_path=pipeline_path,
            config=config,
            dhis2_client=dhis2_client,
            run_pipeline=extract_analytics,
            done=done_pyramid,
        )

    except Exception as e:
        current_run.log_error(f"Error occurred: {e}")


def load_configuration(pipeline_path: Path) -> dict:
    """Reads a JSON file configuration and returns its contents as a dictionary.

    Args:
        pipeline_path (str): Root path of the pipeline to find the file.

    Returns:
        dict: Dictionary containing the JSON data.
    """
    try:
        file_path = pipeline_path / "config" / "snis_extraction_config.json"
        with Path.open(file_path, "r") as file:
            data = json.load(file)

        current_run.log_info("Configuration loaded.")
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file '{file_path}' was not found {e}") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON: {e}") from e
    except Exception as e:
        raise Exception(f"An error occurred while loading the configuration: {e}") from e


@dhis2_snis_extract.task
def connect_to_dhis2(config: dict, cache_dir: str, done) -> DHIS2:
    """Establishes a connection to the DHIS2 and set a cache directory.

    Args:
        config (dict): Configuration dictionary containing DHIS2 connection settings.
        cache_dir (str): Directory path for caching.

    Returns:
        DHIS2: An instance of the DHIS2 client.

    Raises:
        Exception: If an error occurs while connecting to DHIS2 SNIS.
    """
    try:
        connection = workspace.dhis2_connection(config["EXTRACT_SETTINGS"]["DHIS2_CONNECTION"])
        dhis2_client = DHIS2(connection=connection, cache_dir=cache_dir)

        current_run.log_info(f"Connected to DHIS2 connection: {config['EXTRACT_SETTINGS']['DHIS2_CONNECTION']}")
        return dhis2_client
    except Exception as e:
        raise Exception(f"Error while connecting to DHIS2 SNIS: {e}") from e


@dhis2_snis_extract.task
def extract_snis_pyramid(pipeline_path: Path, dhis2_snis_client: DHIS2, run_pipeline: bool) -> int:
    """Pyramid extraction task.

    extracts and saves a pyramid dataframe for all levels (could be set via config in the future)

    Returns:
        bool: True if the extraction and saving process completes or is skipped.
    """
    if not run_pipeline:
        return True

    current_run.log_info("Retrieving SNIS DHIS2 pyramid data")

    try:
        # Retrieve all available levels..
        levels = pl.DataFrame(dhis2_snis_client.meta.organisation_unit_levels())
        org_levels = levels.select("level").unique().sort(by="level").to_series().to_list()
        org_units = dhis2_snis_client.meta.organisation_units(
            fields="id,name,shortName,openingDate,closedDate,parent,level,path,geometry"
        )
        org_units = pd.DataFrame(org_units)
        ou_length = len(org_units[org_units.level == org_levels[-1]].id.unique())
        current_run.log_info(f"Extracted {ou_length} units at organisation unit level {org_levels[-1]}")

        pyramid_fname = pipeline_path / "data" / "pyramid" / "snis_pyramid.parquet"

        # Save as Parquet
        save_to_parquet(data=org_units, filename=pyramid_fname)
        current_run.log_info(f"SNIS DHIS2 pyramid data saved: {pyramid_fname}")
        return True

    except Exception as e:
        raise Exception(f"Error while extracting SNIS DHIS2 Pyramid: {e}") from e


@dhis2_snis_extract.task
def extract_snis_data(pipeline_path: Path, config: dict, dhis2_client: DHIS2, run_pipeline: bool, done) -> bool:
    """Data extraction task.

    Returns:
        bool: True if the extraction and saving process completes or is skipped.
    """
    if not run_pipeline:
        return True

    current_run.log_info("Starting analytics extracts.")

    # initialize update queue
    push_queue = Queue(pipeline_path / "config" / ".queue.db")
    new_data = False
    monthly_periods = get_monthly_periods(config)
    de_files_processed = []
    ds_files_processed = []
    rates_files_processed = []
    try:
        # Load dataset metadata
        ds_metadata = get_datasets(dhis2=dhis2_client)
        dict_coc_cc = get_coc_from_cc(config=config, dhis2=dhis2_client)
        config = add_coc_to_config(config, dict_coc_cc)

        # NOTE: now we can potentially create separated tasks for each handler
        for period in monthly_periods:
            de_files = handle_data_element_extracts(
                dhis2_client=dhis2_client,
                month=period,
                config=config,
                pipeline_path=pipeline_path,
                de_files_processed=de_files_processed,
            )
            if de_files:
                de_files_processed.extend(de_files)
                new_data = True

            ds_files = handle_data_set_extracts(
                dhis2_client=dhis2_client,
                ds_metadata=ds_metadata,
                month=period,
                config=config,
                pipeline_path=pipeline_path,
                ds_files_processed=ds_files_processed,
            )
            if ds_files:
                ds_files_processed.extend(ds_files)
                new_data = True

            rate_files = handle_rate_extracts(
                dhis2_client=dhis2_client,
                ds_metadata=ds_metadata,
                month=period,
                config=config,
                pipeline_path=pipeline_path,
                rates_files_processed=rates_files_processed,
            )
            if rate_files:
                rates_files_processed.extend(rate_files)
                new_data = True

            # indicator_files = handle_indicator_extracts(
            #     dhis2_client=dhis2_client,
            #     ds_metadata=ds_metadata,
            #     month=period,
            #     config=config,
            #     pipeline_path=pipeline_path,
            #     indicator_files_processed=indicator_files_processed,
            # )
            # if rate_files:
            #     indicator_files_processed.extend(rate_files)
            #     new_data = True

            if new_data:
                new_data = False
                push_queue.enqueue(period)

        current_run.log_info("Process finished.")

    except Exception as e:
        raise Exception(f"Extract task error : {e}") from e


def get_coc_from_cc(config: dict, dhis2: DHIS2) -> dict:
    """
    From the config, construct a list that maps the Category Combos to their Category Option Combos.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the Category Combos that we want to extract.

    Returns
    -------
    dict
        A dictionary mapping Category Combos to their Category Option Combos.
    """
    current_run.log_info("Retrieving Category Option Combos from Category Combos")

    unique_cat_combos = set()
    dict_coc_cc = {}
    params = {"fields": "id,displayName,categoryOptionCombos[id,displayName]"}

    for value in config.get("DATA_SETS", []):
        cat_combos = value.get("CAT_COMBOS", [])
        for combo in cat_combos:
            unique_cat_combos.add(combo)

    unique_cat_combos = list(unique_cat_combos)

    for cat_combo in unique_cat_combos:
        response = dhis2.api.get(
            endpoint=f"categoryCombos/{cat_combo}",
            params=params,
        )
        if response:
            category_option_combos = response.get("categoryOptionCombos", [])
            dict_coc_cc[cat_combo] = [coc["id"] for coc in category_option_combos]
        else:
            current_run.log_warning(f"Category Combo {cat_combo} not found in DHIS2.")

    return dict_coc_cc


def add_coc_to_config(config: dict, dict_coc_cc: dict) -> dict:
    """
    Add the categoryOptionCombo to the input config file

    Returns
    -------
    The config file with the CoC added to it.
    """
    current_run.log_info("Adding Category Option Combos to config")

    for value in config.get("DATA_SETS", []):
        cat_combos = value.get("CAT_COMBOS", [])
        for cat_combo in cat_combos:
            if cat_combo in dict_coc_cc:
                value["CAT_OPTION_COMBOS"].extend(dict_coc_cc[cat_combo])

    return config


def handle_data_element_extracts(
    dhis2_client: DHIS2, month: str, config: dict, pipeline_path: Path, de_files_processed: list
) -> list[Path]:
    """Handles the extraction of data elements from DHIS2 for a given month and configuration.

    Parameters
    ----------
    dhis2_client : DHIS2
        An instance of the DHIS2 client used for API calls to retrieve data.
    month : str
        The period in 'YYYYMM' format to process.
    config : dict
        Configuration dictionary containing extraction settings.
    pipeline_path : Path
        Path to the pipeline workspace.
    de_files_processed : list
        List of previously processed files.

    Returns
    -------
    list[Path]
        Returns a list of file paths corresponding to the downloaded extracts, or None if not implemented.
    """
    # NOTE: Missing implementation
    # frequency.lower()
    return None


def handle_data_set_extracts(
    dhis2_client: DHIS2,
    ds_metadata: pl.DataFrame,
    month: str,
    config: dict,
    pipeline_path: Path,
    ds_files_processed: list,
) -> list[str]:
    """Extracts dataset extracts from DHIS2 for the specified data elements, organisation units, and periods.

    Parameters
    ----------
    dhis2_client : DHIS2
        An instance of the DHIS2 client used for API calls to retrieve data.
    ds_metadata : pl.DataFrame
        DataFrame containing dataset metadata including organisation units.
    month : str
        The period in 'YYYYMM' format to process.
    config : dict
        Configuration dictionary containing dataset extraction settings.
    pipeline_path : Path
        Path to the pipeline workspace.
    ds_files_processed: list[str]
        List of previously processed files (track yearly files).

    Returns
    -------
        Returns the list of downloaded extracts file paths (str)
    """
    # for month in monthly_periods:
    current_run.log_info(f"Running for period: {month}")
    period_filenames = []
    overwrite = config["EXTRACT_SETTINGS"].get("OVERWRITE", False)
    dict_done_names = {}

    for item in config["DATA_SETS"]:
        # I have introduced the __ so that I have the same dataset id twice in the dictionary.
        datasets = item.get("DATA_SETS", [])
        dataset_name = "__".join(datasets)
        if dataset_name in dict_done_names.keys():
            count = dict_done_names[dataset_name] + 1
            dict_done_names[dataset_name] = count
        else:
            dict_done_names[dataset_name] = 1

        # set parameters
        ou_ids = ds_metadata.filter(pl.col("id").is_in(datasets))["organisation_units"].to_list()[0]
        data_element_ids = item.get("DATA_ELEMENTS", [])
        # NOTE: if no dataelements, should select all
        if len(data_element_ids) == 0:
            data_element_ids = ds_metadata.filter(pl.col("id").is_in(datasets))["data_elements"].to_list()[0]
        frequency = item.get("FREQUENCY", "Monthly").lower()
        cat_opt_combos = item.get("CAT_OPTION_COMBOS", [])
        current_run.log_info(
            f"Extracting {len(data_element_ids)} data elements from dataset {dataset_name} with frequency {frequency}"
        )

        file_path = retrieve_dataset_extract_by_frequency(
            dhis2_client=dhis2_client,
            dataset_name=dataset_name,
            data_elements=data_element_ids,
            org_units=ou_ids,
            month=month,
            frequency=frequency,
            cat_opt_combos=cat_opt_combos,
            files_processed=ds_files_processed,
            output_path=pipeline_path / "data" / "data_sets",
            overwrite=overwrite,
        )
        if file_path:
            period_filenames.append(str(file_path))

    merge_files(
        prefix="ds",
        period=month,
        input_path=pipeline_path / "data" / "data_sets",
        output_path=pipeline_path / "data" / "extracts",
    )
    return period_filenames


def retrieve_rate_extract_by_frequency(
    dhis2_client: DHIS2,
    dataset_id: str,
    reporting_dataset: str,
    org_units: list[str],
    month: str,
    frequency: str,
    files_processed: list,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Retrieves rate extracts for a specified dataset saving the result as a Parquet file.

    Parameters
    ----------
    dhis2_client : DHIS2
        An instance of the DHIS2 client used for API calls to retrieve data.
    dataset_id : str
        Dataset ID (DHIS2 UID).
    reporting_dataset : str
         dataset UID.
    org_units : list[str]
        List of organisation unit IDs for which data needs to be extracted.
    month : str
        The period for which to extract data (e.g., "202401").
    frequency : str
        Frequency of the periods ("weekly", "monthly", "yearly").
    files_processed : list
        List of already processed file paths to avoid duplicates.
    output_path : Path
        Output path to save the file.
    overwrite : bool, optional
        Whether to overwrite existing files (default is False).

    Returns
    -------
    Path
        Path to the saved Parquet file, or None if skipped.
    """
    if frequency == "yearly":
        filename = f"rate_{dataset_id}_{month[:4]}.parquet"
    else:
        filename = f"rate_{dataset_id}_{month}.parquet"
    file_path = output_path / frequency / filename

    # Download data
    if frequency == "yearly" and str(file_path) in files_processed:
        current_run.log_info(f"Yearly file {filename} ready, skip!")
        return None

    if file_path.exists() and not overwrite:
        current_run.log_info(f"File {filename} ready, skip!")
        return None

    try:
        response = dhis2_client.analytics.get(
            data_elements=reporting_dataset,
            periods=[month],
            org_units=org_units,
            include_cocs=False,
        )
    except Exception as e:
        raise Exception(f"Error while retrieving rate(s) data: {e}") from e

    if len(response) == 0:
        current_run.log_warning(f"No data found for rate period(s): {month}")
    else:
        current_run.log_info(f"{len(response)} Data points found in period: {month}")
        df_formatted = pd.DataFrame(response).rename(columns={"pe": "period", "ou": "orgUnit"})
        df_formatted = map_to_dhis2_format(df_formatted, data_type="DATASET")  # NOTE: maybe this is not the right NAME?
        save_to_parquet(data=df_formatted, filename=file_path)

    return file_path


def retrieve_dataset_extract_by_frequency(
    dhis2_client: DHIS2,
    dataset_name: str,
    data_elements: list[str],
    org_units: list[str],
    month: str,
    frequency: str,
    cat_opt_combos: list[str],
    files_processed: list[str],
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Retrieves dataset extracts from DHIS2 for the specified data elements, org units, and period.

    Avoids downloading the same files again depending on "overwrite".
    Only adds new downloaded files to the output list.

    Parameters
    ----------
    dhis2_client : DHIS2
        An instance of the DHIS2 client used for API calls to retrieve data.
    dataset_name : str
        The name of the dataset (DHIS2 uid + suffix).
    data_elements : list[str]
        List of data element UIDs to extract.
    org_units : list[str]
        List of organisation unit IDs for which data needs to be extracted.
    month : str
        The month for which data should extract data.
    frequency : str
        Frequency of the periods (weekly, monthly, yearly)
    cat_opt_combos : list[str]
        List of cat option combos to select (filter)
    files_processed : list[str]
        List of already processed file paths to avoid duplicates.
    output_path : Path
        output path to save the file
    overwrite: bool
        download the files again yes o no

    Returns
    -------
        Returns a list of files corresponding to the downloaded period
    """
    if frequency == "yearly":
        filename = f"ds_{dataset_name}_{month[:4]}.parquet"
    else:
        filename = f"ds_{dataset_name}_{month}.parquet"
    file_path = output_path / frequency / filename

    # Download data
    if frequency == "yearly" and str(file_path) in files_processed:
        current_run.log_info(f"Yearly file {filename} ready, skip!")
        return None

    if file_path.exists() and not overwrite:
        current_run.log_info(f"File {filename} ready, skip!")
        return None

    extract_periods = get_periods_by_month(frequency=frequency, month=month)

    try:
        response = dhis2_client.data_value_sets.get(
            data_elements=data_elements,
            periods=extract_periods,
            org_units=org_units,
        )
    except Exception as e:
        raise Exception(f"Error retrieving dataset data elements {month} : {e}") from e

    if len(response) == 0:
        current_run.log_warning(f"No data found for period(s): {extract_periods}")
        return None
    else:
        current_run.log_info(f"{len(response)} Data points found in period(s): {extract_periods}")
        df_formatted = pd.DataFrame(response)
        if len(cat_opt_combos) > 0:
            df_formatted = df_formatted[(df_formatted.categoryOptionCombo.isin(cat_opt_combos))]

        if not df_formatted.empty:
            df_formatted = map_to_dhis2_format(df_formatted, data_type="DATAELEMENT")
            save_to_parquet(data=df_formatted, filename=file_path)
            current_run.log_info(f"Dataset extract saved: {file_path} with {len(df_formatted)} records")
        else:
            current_run.log_warning(
                f"No valid data found for period(s): {extract_periods} with cat option combos {cat_opt_combos}"
            )
            return None

    return file_path


def merge_files(prefix: str, period: str, input_path: Path, output_path: Path) -> None:
    """Merges multiple Parquet files into a single file for a given period.

    Parameters
    ----------
    prefix : list[str]
        Prefix used to idenfity the files.
    period : str
        The period identifier used for naming the merged output file.
    input_path : Path
        Source directory of the dhis2 data
    output_path : Path
        The directory where the merged file will be saved.
    """
    df_list = []
    # All .parquet files in all subdirectories
    all_period_files = [f.resolve() for f in input_path.rglob(f"{prefix}_*_{period}.parquet") if f.is_file()]
    all_yearly_files = [f.resolve() for f in input_path.rglob(f"{prefix}_*_{period[:4]}.parquet") if f.is_file()]
    all_files = all_period_files + all_yearly_files

    if len(all_files) == 0:
        current_run.log_info(f"No files found for {prefix} in period {period}. Skipping merge.")
        return

    current_run.log_info(f"Merging {len(all_files)} files")
    for f in all_files:
        try:
            df = pl.read_parquet(f)
            df_list.append(df)
        except Exception as e:
            raise Exception(f"Failed to read {f}: {e}") from e

    merged_df = pl.concat(df_list)
    merged_df = merged_df.unique()
    save_to_parquet(data=merged_df.to_pandas(), filename=output_path / f"{prefix}_extract_{period}.parquet")


def get_periods_by_month(frequency: str, month: str) -> list:
    """Get periods based on frequency.

    Parameters
    ----------
    frequency : str
        The frequency of the data (e.g., "weekly", "monthly", "yearly").
    month : str
        The monh date in 'YYYYMM' format.

    Returns
    -------
    list
        A list of periods based on the specified frequency.
    """
    if frequency == "weekly":
        period_list = get_weekly_periods(month, month)
    elif frequency == "monthly":
        period_list = [month]
    elif frequency == "yearly":
        period_list = [month[:4]]
    else:
        raise ValueError(f"Frequency {frequency} not supported.")
    return period_list


def get_weekly_periods(start: str, end: str) -> list:
    """Get a list of weekly periods between start and end dates.

    Returns
    -------
      a list of periods in the format YYYYWW (e.g., '2024W12').
    """
    try:
        start_week_str, _ = get_start_end_weeks(start)
        _, end_week_str = get_start_end_weeks(end)
        week_start = Week.from_string(start_week_str)
        week_end = Week.from_string(end_week_str)
    except Exception as e:
        raise Exception(f"Error computing weekly periods {e}") from e
    return [str(w) for w in week_start.get_range(week_end)]


def get_start_end_weeks(yyyymm: str) -> tuple[str, str]:
    """Given a year and month in 'YYYYMM' format, returns the ISO week strings for the first and last day of the month.

    Parameters
    ----------
    yyyymm : str
        The year and month in 'YYYYMM' format.

    Returns
    -------
    tuple[str, str]
        A tuple containing the ISO week string for the first day and the last day of month (eg:'2024W13', '2024W17').
    """
    # Parse the year and month
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])

    start_date = datetime(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end_date = datetime(year, month, last_day)

    # Get ISO week numbers
    start_week = start_date.isocalendar()  # (ISO_year, ISO_week, ISO_weekday)
    end_week = end_date.isocalendar()

    # Format as YYYYWww
    start_str = f"{start_week[0]}W{start_week[1]}"
    end_str = f"{end_week[0]}W{end_week[1]}"

    return start_str, end_str


def handle_rate_extracts(
    dhis2_client: DHIS2,
    ds_metadata: pl.DataFrame,
    month: str,
    config: dict,
    pipeline_path: Path,
    rates_files_processed: list,
) -> list[str]:
    """Extracts and processes rate extracts from DHIS2 for the specified datasets, organisation units, and periods.

    Parameters
    ----------
    dhis2_client : DHIS2
        An instance of the DHIS2 client used for API calls to retrieve data.
    ds_metadata : pl.DataFrame
        DataFrame containing dataset metadata including organisation units.
    month : str
        The period for which to extract data (e.g., "202401").
    config : dict
        Configuration dictionary containing rate extraction settings.
    pipeline_path : Path
        Path to the pipeline workspace.
    rates_files_processed : list
        List of already processed file paths to avoid duplicates.

    Returns
    -------
    list[str]
        Returns the list of downloaded extracts file paths (str).
    """
    # for month in monthly_periods:
    current_run.log_info(f"Retrieving rate extracts for period: {month}")
    period_filenames = []
    overwrite = config["EXTRACT_SETTINGS"].get("OVERWRITE", False)

    for r in config["RATE_UIDS"].items():
        reporting_dataset_id = r[0]
        metrics = r[1].get("METRICS", None)
        frequency = r[1].get("FREQUENCY", None)
        reporting_ds = [f"{ds}.{metric}" for ds, metric in product([reporting_dataset_id], metrics)]
        ou_ids = ds_metadata.filter(pl.col("id") == reporting_dataset_id)["organisation_units"].to_list()[0]

        current_run.log_info(f"Extracting {len(reporting_ds)} reporting rates with frequency {frequency}")

        # NOTE: frequency not implemented !
        file_path = retrieve_rate_extract_by_frequency(
            dhis2_client=dhis2_client,
            dataset_id=reporting_dataset_id,
            reporting_dataset=reporting_ds,
            org_units=ou_ids,
            month=month,
            frequency=frequency.lower(),
            files_processed=rates_files_processed,
            output_path=pipeline_path / "data" / "rates",
            overwrite=overwrite,
        )
        if file_path:
            period_filenames.append(str(file_path))

    merge_files(
        prefix="rate",
        period=month,
        input_path=pipeline_path / "data" / "rates",
        output_path=pipeline_path / "data" / "extracts",
    )
    return period_filenames


def map_to_dhis2_format(
    dhis_data: pd.DataFrame,
    data_type: str = "DATAELEMENT",
    domain_type: str = "AGGREGATED",
) -> pd.DataFrame:
    """Maps DHIS2 data to a standardized data extraction table.

    Parameters
    ----------
    dhis_data : pd.DataFrame
        Input DataFrame containing DHIS2 data. Must include columns like `period`, `orgUnit`,
        `categoryOptionCombo(DATAELEMENT)`, `attributeOptionCombo(DATAELEMENT)`, `dataElement`
        and `value` based on the data type.
    data_type : str
        The type of data being mapped. Supported values are:
        - "DATAELEMENT": Includes `categoryOptionCombo` and maps `dataElement` to `dx_uid`.
        - "DATASET": Maps `dx` to `dx_uid` and `rate_type` by split the string by `.`.
        - "INDICATOR": Maps `dx` to `dx_uid`.
        Default is "DATAELEMENT".
    domain_type : str, optional
        The domain of the data if its per period (Agg ex: monthly) or datapoint (Tracker ex: per day):
        - "AGGREGATED": For aggregated data (default).
        - "TRACKER": For tracker data.

    Returns
    -------
    pd.DataFrame
        A DataFrame formatted to SNIS standards, with the following columns:
        - "data_type": The type of data (DATAELEMENT, DATASET, or INDICATOR).
        - "dx_uid": Data element, dataset, or indicator UID.
        - "period": Reporting period.
        - "orgUnit": Organization unit.
        - "categoryOptionCombo": (Only for DATAELEMENT) Category option combo UID.
        - "rate_type": (Only for DATASET) Rate type.
        - "domain_type": Data domain (AGGREGATED or TRACKER).
        - "value": Data value.
    """
    if dhis_data.empty:
        return None

    # NOTE: Think about renaming this things , perhaps DATASET should be replaced by:
    # DATASET_RATE (data elements retrieved from a DATASET can just be DATAELEMENTS)
    if data_type not in ["DATAELEMENT", "DATASET", "INDICATOR"]:
        raise ValueError("Incorrect 'data_type' configuration ('DATAELEMENT', 'DATASET', 'INDICATOR')")

    try:
        dhis2_format = pd.DataFrame(
            columns=[
                "data_type",
                "dx_uid",
                "period",
                "org_unit",
                "category_option_combo",
                "attribute_option_combo",
                "rate_type",
                "domain_type",
                "value",
            ]
        )
        dhis2_format["period"] = dhis_data.period
        dhis2_format["org_unit"] = dhis_data.orgUnit
        dhis2_format["domain_type"] = domain_type
        dhis2_format["value"] = dhis_data.value
        dhis2_format["data_type"] = data_type
        if data_type in ["DATAELEMENT"]:
            dhis2_format["dx_uid"] = dhis_data.dataElement
            dhis2_format["category_option_combo"] = dhis_data.categoryOptionCombo
            dhis2_format["attribute_option_combo"] = dhis_data.attributeOptionCombo
        if data_type == "DATASET":
            dhis2_format[["dx_uid", "rate_type"]] = dhis_data.dx.str.split(".", expand=True)
        if data_type == "INDICATOR":
            dhis2_format["dx_uid"] = dhis_data.dx
        return dhis2_format

    except AttributeError as e:
        raise AttributeError(f"Routine data is missing a required attribute: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected Error while creating routine format table: {e}") from e


@dhis2_snis_extract.task
def validate_period_range(config: dict) -> None:
    """Validate that start and end are valid yyyymm values and ordered correctly."""
    start = config["EXTRACT_SETTINGS"].get("STARTDATE", None)
    end = config["EXTRACT_SETTINGS"].get("ENDDATE", None)
    if start is None or end is None:
        raise ValueError("Start or end period is missing.")

    validate_yyyymm(start)
    validate_yyyymm(end)

    if start > end:
        raise ValueError("Start period must not be after end period.")

    return True


def validate_yyyymm(yyyymm_str: str) -> None:
    """Validate that the input is an integer in yyyymm format."""
    if len(yyyymm_str) != 6:
        raise ValueError("Input must be a 6-digit string in yyyymm format.")

    # year = int(yyyymm_str[:4])
    month = int(yyyymm_str[4:])
    if month < 1 or month > 12:
        raise ValueError(f"Month must be between 01 and 12 in {yyyymm_str}, got {month}.")


def get_monthly_periods(config: dict) -> list[str]:
    """Generate a list of monthly periods between the start and end dates specified in the configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing 'EXTRACT_SETTINGS' with 'STARTDATE', 'ENDDATE',
        and optionally 'NUMBER_MONTHS_WINDOW'.

    Returns
    -------
    list[str]
        List of monthly period strings in 'YYYYMM' format.
    """
    #  Set start and end periods ---> with a window of 6 months we are safe: 6 months DEFAULT
    months_lag = config["EXTRACT_SETTINGS"].get("NUMBER_MONTHS_WINDOW", 6)  # default 6 months window

    start = config.get("EXTRACT_SETTINGS", {}).get("STARTDATE")
    if not start:
        start = (datetime.now() - relativedelta(months=months_lag)).strftime("%Y%m")

    end = config.get("EXTRACT_SETTINGS", {}).get("ENDDATE")
    if not end:
        end = (datetime.now() - relativedelta(months=1)).strftime("%Y%m")  # go back 1 month.

    # Get periods
    if start == end:
        monthly_periods = [start]
    else:
        month_start = Month.from_string(start)
        month_end = Month.from_string(end)
        monthly_periods = [str(m) for m in month_start.get_range(month_end)]

    return monthly_periods


# NOTE: The folowing functions could be eventually used (no needed for Niger)
# I keep them here for reference and for useful snippets in the code.
# they can be removed after Niger testing.

# def retrieve_ou_list(dhis2_client: DHIS2, ou_level: int) -> list:
#     """Retrieve a list of organisation unit IDs from DHIS2 at specific OU level.

#     Args:
#         dhis2_client (DHIS2): An instance of the DHIS2 client.
#         ou_level (int): The organisational unit level to filter by.

#     Returns:
#         list: A list of organisation unit IDs at the specified level.

#     Raises:
#         Exception: If an error occurs while retrieving the organisation unit list.
#     """
#     try:
#         # Retrieve organisational units and filter by ou_level
#         ous = pd.DataFrame(dhis2_client.meta.organisation_units())
#         ou_list = ous.loc[ous.level == ou_level].id.to_list()

#         # Log the result based on the OU level
#         if ou_level == 5:
#             current_run.log_info(f"Retrieved SNIS DHIS2 FOSA id list {len(ou_list)}")
#         elif ou_level == 4:
#             current_run.log_info(f"Retrieved SNIS DHIS2 Aires de Sante id list {len(ou_list)}")
#         else:
#             current_run.log_info(f"Retrieved SNIS DHIS2 OU level {ou_level} id list {len(ou_list)}")
#         return ou_list
#     except Exception as e:
#         raise Exception(f"Error while retrieving OU id list for level {ou_level}: {e}") from e


# def handle_extract_for_period(dhis2_client: DHIS2, period: str, org_unit_list: list, config: dict, root_path: Path):
#     """Function to retrieve and processes data from the DHIS2 system for a given period.

#     Extracts are saved in a Parquet files.

#     Parameters
#     ----------
#     dhis2_client : DHIS2
#         An instance of the DHIS2 client used for API calls to retrieve data.
#     period : str
#         The period for data extraction in the "YYYYMM" format
#         (e.g., "202409" for September 2024 -> MONTHLY)
#     org_unit_list : list
#         A list of organizational unit IDs for which data needs to be extracted.
#     config : dict
#         A dictionary containing the extraction configuration.
#     root_path : str
#         The root directory or the pipeline
#     """
#     # fname period extract (We can use the config["DHIS2_CONNECTION"] as reference in the file)
#     extract_fname = root_path / "data" / "extracts" / f"snis_data_{period}.parquet"
#     queue_db_path = root_path / "config" / ".queue.db"

#     # initialize update queue
#     push_queue = Queue(queue_db_path)

#     # Download and replace the extract
#     raw_routine_data = retrieve_snis_routine_extract(
#         dhis2_snis_client=dhis2_client,
#         period=period,
#         org_unit_list=org_unit_list,
#         routine_ids=config["ROUTINE_DATA_ELEMENT_UIDS"],
#         last_updated=None,
#     )
#     raw_rates_data = retrieve_snis_rates_extract(dhis2_client, period, org_unit_list, config["RATE_UIDS"])
#     raw_data = merge_dataframes([raw_routine_data, raw_rates_data])  # + raw_acm_data
#     if raw_data is not None:
#         if extract_fname.exists():
#             current_run.log_info(f"Replacing extract for period {period}.")
#         save_to_parquet(raw_data, extract_fname)
#         push_queue.enqueue(period)
#     else:
#         current_run.log_info(f"Nothing to save for period {period}..")


# def handle_indicator_extracts(dhis2_client: DHIS2, period: str, config: dict) -> pd.DataFrame:
#     """Retrieves indicator data extracts from DHIS2 for a given period.

#     Parameters
#     ----------
#     dhis2_client : DHIS2
#         An instance of the DHIS2 client used for API calls to retrieve data.
#     period : str
#         The period for data extraction in the "YYYYMM" format.
#     config : dict
#         Configuration dictionary containing indicator extraction settings.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame formatted to SNIS standards containing the extracted ACM indicator data.
#     """
#     for ind in config["INDICATORS"].items():
#         indicator_ids = ind[0]  # NOTE: Define configuration for indicators!
#         try:
#             response = dhis2_client.analytics.get(
#                 indicators=indicator_ids,
#                 periods=[period],
#                 org_units=[],
#                 include_cocs=False,
#             )
#         except Exception as e:
#             raise Exception(f"Error while retrieving ACM data: {e}") from e

#     raw_data_formatted = pd.DataFrame(response).rename(columns={"pe": "period", "ou": "orgUnit"})
#     return map_to_dhis2_format(raw_data_formatted, data_type="INDICATOR")


if __name__ == "__main__":
    dhis2_snis_extract()
