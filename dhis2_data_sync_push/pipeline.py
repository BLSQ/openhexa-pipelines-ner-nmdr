from pathlib import Path

import polars as pl
from d2d_development.push import DHIS2Pusher
from openhexa.sdk import current_run, parameter, pipeline, workspace
from utils import (
    configure_logging,
    connect_to_dhis2,
    get_extract_periods,
    load_configuration,
    resolve_dates_and_validate,
    save_logs,
)


@pipeline("dhis2_data_sync_push")
@parameter(
    code="start_date",
    name="Start date",
    help="Start date of import (YYYYMM).",
    type=str,
    required=False,
    default="201701",
)
@parameter(
    code="end_date", name="End date", help="End date of import (YYYYMM).", type=str, required=False, default="201701"
)  ## default="202112"
def dhis2_data_sync_push(start_date: str, end_date: str) -> None:
    """Write your pipeline orchestration here."""
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_data_sync_push"
    data_path = Path(workspace.files_path) / "pipelines" / "dhis2_data_sync_extracts" / "data"
    # config = load_configuration(pipeline_path / "configuration" / "push_config.json")

    try:
        push_extracts(pipeline_path=pipeline_path, data_path=data_path, start_date=start_date, end_date=end_date)
    except Exception as e:
        current_run.log_error(f"Pipeline execution failed: {e}")
        raise
    current_run.log_info("Pipeline execution completed.")


def push_extracts(pipeline_path: Path, data_path: Path, start_date: str, end_date: str) -> None:
    """Push data extracts to the target DHIS2 instance."""
    current_run.log_info("Starting data push.")

    # logger, logs_file = configure_logging(logs_path=Path("/home/jovyan/tmp/logs"), task_name="push_data")
    logger, logs_file = configure_logging(logs_path=pipeline_path / "logs", task_name="push_data")  ## LOCAL

    try:
        # load configuration
        config_push = load_configuration(config_path=pipeline_path / "configuration" / "push_config.json")
        dhis2_client_target = connect_to_dhis2(connection_str=config_push["SETTINGS"]["TARGET_DHIS2_CONNECTION"])
    except Exception as e:
        current_run.log_error(f"Error occurred: {e}")
        raise

    # Parameters for the api call
    import_strategy = config_push["SETTINGS"].get("IMPORT_STRATEGY", "CREATE_AND_UPDATE")
    dry_run = config_push["SETTINGS"].get("DRY_RUN", True)
    max_post = config_push["SETTINGS"].get("MAX_POST", 500)
    current_run.log_info(f"Import strategy: {import_strategy} - Dry Run: {dry_run} - Max post: {max_post}")

    pusher = DHIS2Pusher(
        dhis2_client=dhis2_client_target,
        import_strategy=import_strategy,
        dry_run=dry_run,
        max_post=max_post,
        logger=logger,
    )

    for extract in config_push.get("DATA_ELEMENTS", {}).get("EXTRACTS", []):
        current_run.log_info(f"Processing extract: {extract['EXTRACT_UID']}")
        handle_data_element_extract_push(
            pusher=pusher,
            pipeline_path=pipeline_path,
            data_path=data_path,
            extract=extract,
            start_date=start_date,
            end_date=end_date,
            logs_file=logs_file,
        )


def handle_data_element_extract_push(
    pusher: DHIS2Pusher,
    pipeline_path: Path,
    data_path: Path,
    extract: dict,
    start_date: str,
    end_date: str,
    logs_file: Path,
) -> None:
    """Handle the push of a single extract."""
    # get dates and validate
    periods = resolve_periods(start_date, end_date, extract.get("FREQUENCY", "MONTHLY"))
    current_run.log_info(f"Pushing from: {periods[0]} to {periods[-1]} for extract: {extract['name']}")

    ex_uid = extract.get("EXTRACT_UID")
    if ex_uid is None:
        current_run.log_error(f"Extract UID not found for extract: {extract.get('name')}. Skipping.")
        return

    for period in periods:
        # Load the data to be pushed
        data_file_path = data_path / ex_uid / f"data_{ex_uid}_{period}.parquet"
        if not data_file_path.exists():
            current_run.log_warning(f"Data file {data_file_path} not found. Skipping push for period {period}.")
            continue

        data_df = pl.read_parquet(data_file_path)
        current_run.log_info(f"Loaded {len(data_df)} records from {data_file_path}.")
        data_mapped = apply_data_element_mappings(data_df, extract.get("MAPPING", {}))

        try:
            pusher.push_data(data_df=data_mapped)
            current_run.log_info(f"Pushed data for extract {data_file_path.name}")
        except Exception as e:
            current_run.log_error(f"Failed to push data for extract {ex_uid}. Error: {e}")
        finally:
            save_logs(logs_file=logs_file, task_name="push_data")


def apply_data_element_mappings(df: pl.DataFrame, extract: dict) -> pl.DataFrame:
    """Applies indicator mappings to the extracted data.

    NOTE: It also FILTERS! indicator based on category option combo (COC) if specified in the extract configuration.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing the extracted data.
    extract : dict
        This is a dictionary containing the extract mappings.

    Returns
    -------
    pl.DataFrame
        DataFrame with mapped indicator.
    """
    if len(extract) == 0:
        current_run.log_warning("No extract details provided, skipping indicator mappings.")
        return df

    extract_mappings = extract.get("MAPPINGS", {})
    if len(extract_mappings) == 0:
        current_run.log_warning("No extract mappings provided, skipping indicator mappings.")
        return df

    # Loop over the configured indicator mappings to filter by COC/AOC if provided
    chunks = []
    uid_mappings = {}
    for uid, mapping in extract_mappings.items():
        uid_mapping = mapping.get("UID")
        coc_mappings = mapping.get("CATEGORY_OPTION_COMBO", {})
        aoc_mappings = mapping.get("ATTRIBUTE_OPTION_COMBO", {})

        # Build a mask selection
        df_uid = df.filter(pl.col("dx") == uid)
        if coc_mappings:
            coc_mappings_clean = clean_mapping(coc_mappings)
            df_uid = df_uid.filter(pl.col("category_option_combo").is_in(list(coc_mappings_clean.keys())))
            df_uid = df_uid.with_columns(pl.col("category_option_combo").replace(coc_mappings_clean))

        if aoc_mappings:
            aoc_mappings_clean = clean_mapping(aoc_mappings)
            df_uid = df_uid.filter(pl.col("attribute_option_combo").is_in(list(aoc_mappings_clean.keys())))
            df_uid = df_uid.with_columns(pl.col("attribute_option_combo").replace(aoc_mappings_clean))

        if uid_mapping:
            uid_mappings[uid] = uid_mapping

        chunks.append(df_uid)

    if len(chunks) == 0:
        current_run.log_warning("No data elements matched the provided mappings, returning empty dataframe.")
        return pl.DataFrame(schema=df.schema)

    df_filtered = pl.concat(chunks)

    # set defaults
    df_filtered = df_filtered.with_columns(
        pl.col("category_option_combo").fill_null("HllvX50cXC0"),
        pl.col("attribute_option_combo").fill_null("HllvX50cXC0"),
    )

    if uid_mappings:
        uid_mappings = {k: v for k, v in uid_mappings.items() if v is not None}  # Do not replace with None
        uid_mappings_clean = {str(k).strip(): str(v).strip() for k, v in uid_mappings.items()}
        df_filtered = df_filtered.with_columns(pl.col("dx").replace(uid_mappings_clean))

    return df_filtered


def clean_mapping(mapping: dict) -> dict:
    """Cleans the provided mapping by removing entries with None values and stripping whitespace.

    Returns:
        dict: Cleaned mapping dictionary.
    """
    return {str(k).strip(): str(v).strip() for k, v in mapping.items() if v is not None}


def resolve_periods(start_date: str, end_date: str, frequency: str) -> list:
    """Resolve and validate the start and end dates, returning a list of periods.

    Returns:
        A list of periods in the format required for the data push.
    """
    start, end = resolve_dates_and_validate(
        start_date,
        end_date,
        {
            "SETTINGS": {
                "NUMBER_MONTHS_WINDOW": 6,
                "STARTDATE": "201701",  # NER historical data defaults.
                "ENDDATE": "202112",
            }
        },
    )
    ext_periods = get_extract_periods(start, end)
    if frequency == "YEARLY":
        return sorted(set(str(p[:4]) for p in ext_periods))
    return ext_periods


if __name__ == "__main__":
    dhis2_data_sync_push()
