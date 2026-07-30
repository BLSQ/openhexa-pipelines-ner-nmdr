from pathlib import Path

import polars as pl
from d2d_development.push import DHIS2Pusher
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2.dataframe import get_data_elements
from utils import (
    configure_logging,
    connect_to_dhis2,
    get_extract_periods,
    load_configuration,
    resolve_dates_and_validate,
    save_logs,
)

DF_PUSH_SCHEMA = {
    "dx": pl.String,
    "period": pl.String,
    "org_unit": pl.String,
    "category_option_combo": pl.String,
    "attribute_option_combo": pl.String,
    "value": pl.String,
}


@pipeline("dhis2_data_sync_push")
@parameter(
    code="start_date",
    name="Start date",
    help="Start date of import (YYYYMM).",
    type=str,
    required=False,
    default="201701",  # Historical data defaults.
)
@parameter(
    code="end_date",
    name="End date",
    help="End date of import (YYYYMM).",
    type=str,
    required=False,
    default="202112",  # Historical data defaults.
)
def dhis2_data_sync_push(start_date: str, end_date: str) -> None:
    """Write your pipeline orchestration here."""
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_data_sync_push"
    data_path = Path(workspace.files_path) / "pipelines" / "dhis2_data_sync_extract" / "data"

    try:
        push_extracts(pipeline_path=pipeline_path, data_path=data_path, start_date=start_date, end_date=end_date)
    except Exception as e:
        current_run.log_error(f"Pipeline execution failed: {e}")
        raise

    current_run.log_info("Pipeline execution completed.")


def push_extracts(pipeline_path: Path, data_path: Path, start_date: str, end_date: str) -> None:
    """Push data extracts to the target DHIS2 instance."""
    current_run.log_info("Starting data push.")

    logger, logs_file = configure_logging(logs_path=Path("/home/jovyan/tmp/logs"), task_name="push_data")
    # logger, logs_file = configure_logging(logs_path=pipeline_path / "logs", task_name="push_data")  ## LOCAL

    try:
        # load configuration
        config_push = load_configuration(config_path=pipeline_path / "configuration" / "push_config.json")
        dhis2_client_target = connect_to_dhis2(connection_str=config_push["SETTINGS"]["TARGET_DHIS2_CONNECTION"])
    except Exception as e:
        current_run.log_error(f"{e}")
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

    # Retrieve data element details for DHIS2 import type matching
    de_metadata = get_data_elements(dhis2_client_target)
    for extract in config_push.get("DATA_ELEMENTS", {}).get("EXTRACTS", []):
        handle_data_element_extract_push(
            pipeline_path=pipeline_path,
            pusher=pusher,
            data_path=data_path,
            extract=extract,
            start_date=start_date,
            end_date=end_date,
            de_metadata=de_metadata,
            logs_file=logs_file,
        )


def handle_data_element_extract_push(
    pipeline_path: Path,
    pusher: DHIS2Pusher,
    data_path: Path,
    extract: dict,
    start_date: str,
    end_date: str,
    de_metadata: pl.DataFrame,
    logs_file: Path,
) -> None:
    """Handle the push of a single extract."""
    # get dates and validate
    periods = resolve_periods(start_date, end_date, extract.get("FREQUENCY", "MONTHLY"))

    ex_uid = extract.get("EXTRACT_UID")
    if ex_uid is None:
        current_run.log_error(f"Extract UID not found for extract: '{extract.get('name')}'. Skipping.")
        return

    current_run.log_info(f"Pushing extract: '{ex_uid}' from: {periods[0]} to {periods[-1]}")
    for period in periods:
        # Load the data to be pushed
        data_file_path = data_path / "extracts" / ex_uid / f"data_{ex_uid}_{period}.parquet"
        if not data_file_path.exists():
            current_run.log_warning(f"Data file '{data_file_path}' not found. Skipping push for period {period}.")
            continue

        # data_df = pl.read_parquet(data_file_path)
        # current_run.log_info(f"Loaded {len(data_df)} records from '{data_file_path}'.")
        data_df = load_extract_data(data_file_path)

        try:
            data_mapped = apply_data_element_mappings(data_df, extract.get("MAPPINGS", {}))
        except Exception as e:
            current_run.log_error(f"Failed to apply mappings for extract '{ex_uid}'. Error: {e}")
            raise

        data_mapped = apply_data_element_value_types(data_mapped, de_metadata)

        try:
            pusher.push_data(apply_schema(data_mapped, DF_PUSH_SCHEMA))
            current_run.log_info(f"Push finished for file: '{data_file_path.name}'")
        except Exception as e:
            current_run.log_error(f"Failed to push data for extract '{ex_uid}'. Error: {e}")
        finally:
            save_logs(logs_file=logs_file, output_dir=pipeline_path / "logs")


def load_extract_data(data_file_path: Path) -> pl.DataFrame:
    """Loads a data extract Parquet file and cleans up literal 'None' string values.

    Args:
        data_file_path: Path to the Parquet file to load.

    Returns:
        DataFrame with 'value' cells storing the literal string "None" converted to actual null.
    """
    data_df = pl.read_parquet(data_file_path)
    current_run.log_info(f"Loaded {len(data_df)} records from '{data_file_path}'.")

    none_string_count = data_df.filter(pl.col("value") == "None").height
    if none_string_count > 0:
        current_run.log_warning(
            f"Found {none_string_count} value(s) stored as the literal string 'None' in "
            f"'{data_file_path.name}'; converting to actual null."
        )
        data_df = data_df.with_columns(
            pl.when(pl.col("value") == "None").then(None).otherwise(pl.col("value")).alias("value")
        )

    return data_df


def apply_data_element_mappings(df: pl.DataFrame, mappings: dict) -> pl.DataFrame:
    """Applies indicator mappings to the extracted data.

    NOTE: It also FILTERS! indicator based on category option combo (COC) if specified in the extract configuration.

    Args:
    ----
    df: pl.DataFrame
        DataFrame containing the extracted data.
    mappings: dict
        Dictionary containing the indicator mappings.

    Returns:
    -------
    pl.DataFrame
        DataFrame with mapped indicator.
    """
    if len(mappings) == 0:
        current_run.log_warning("No extract details provided, skipping indicator mappings.")
        return df

    # Loop over the configured indicator mappings to filter by COC/AOC if provided
    chunks = []
    uid_mappings = {}
    for uid, mapping in mappings.items():
        uid_mapping = mapping.get("UID")
        coc_mappings = mapping.get("CATEGORY_OPTION_COMBO", {})
        aoc_mappings = mapping.get("ATTRIBUTE_OPTION_COMBO", {})

        # Build a mask selection
        df_uid = df.filter(pl.col("dx") == uid)
        if coc_mappings:
            if set(coc_mappings) == {"ALL"}:
                df_uid = df_uid.with_columns(pl.lit(coc_mappings["ALL"]).alias("category_option_combo"))
            else:
                coc_mappings_clean = clean_mapping(coc_mappings)
                df_uid = df_uid.filter(pl.col("category_option_combo").is_in(list(coc_mappings_clean.keys())))
                df_uid = df_uid.with_columns(pl.col("category_option_combo").replace(coc_mappings_clean))

        if aoc_mappings:
            aoc_mappings_clean = clean_mapping(aoc_mappings)
            df_uid = df_uid.filter(pl.col("attribute_option_combo").is_in(list(aoc_mappings_clean.keys())))
            df_uid = df_uid.with_columns(pl.col("attribute_option_combo").replace(aoc_mappings_clean))

        if uid_mapping:
            uid_mappings[uid] = uid_mapping

        df_uid = (
            df_uid.with_columns(pl.col("value").cast(pl.Float64))
            .group_by(["dx", "period", "org_unit", "category_option_combo", "attribute_option_combo"])
            .agg(pl.when(pl.col("value").is_null().all()).then(None).otherwise(pl.sum("value")).alias("value"))
        )
        chunks.append(apply_schema(df_uid, DF_PUSH_SCHEMA))

    if len(chunks) == 0:
        current_run.log_warning("No data elements matched the provided mappings, returning empty dataframe.")
        return pl.DataFrame(schema=DF_PUSH_SCHEMA)

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


def apply_schema(df: pl.DataFrame, schema: dict) -> pl.DataFrame:
    """Selects and casts a DataFrame's columns to match a given schema.

    Args:
        df: Input DataFrame.
        schema: Mapping of column name to polars dtype.

    Returns:
        DataFrame containing only the schema's columns (in schema order), cast to the schema's dtypes.
    """
    return df.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])


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


def cast_numeric_data_element_value(
    df: pl.DataFrame, de_id: str, dtype: pl.DataType, min_value: float | None = None
) -> pl.DataFrame:
    """Casts the value column to dtype for rows matching de_id, setting invalid values to None.

    Args:
        df: DataFrame containing the data.
        de_id: Data element UID whose rows should be cast.
        dtype: Target polars dtype (pl.Float64 or pl.Int64).
        min_value: If set, casted values below it are treated as invalid.

    Returns:
        DataFrame with the value column updated for rows matching de_id.
    """
    casted = pl.col("value").cast(pl.Float64, strict=False)
    if dtype != pl.Float64:
        casted = casted.cast(dtype, strict=False)

    is_invalid = casted.is_null()
    if min_value is not None:
        is_invalid = is_invalid | (casted < min_value)

    invalid_count = df.filter((pl.col("dx") == de_id) & pl.col("value").is_not_null() & is_invalid).height
    if invalid_count > 0:
        type_label = "numeric" if dtype == pl.Float64 else "integer"
        reason = f"negative or non-{type_label}" if min_value is not None else f"non-{type_label}"
        current_run.log_warning(f"Data element '{de_id}' has {invalid_count} {reason} value(s) set to None.")

    return df.with_columns(
        pl.when((pl.col("dx") == de_id) & ~is_invalid)
        .then(casted)
        .when(pl.col("dx") == de_id)
        .then(pl.lit(None, dtype=dtype))
        .otherwise(pl.col("value"))
        .alias("value")
    )


def apply_data_element_value_types(df: pl.DataFrame, de_metadata: pl.DataFrame) -> pl.DataFrame:
    """Applies data element types to the data points based on metadata.

    Args:
        df: DataFrame containing the extracted data.
        de_metadata: DataFrame containing data element metadata.

    Returns:
        DataFrame with data element types applied.
    """
    dx_ids = df.get_column("dx").unique().to_list()
    de_types = de_metadata.filter(pl.col("id").is_in(dx_ids)).select(["id", "value_type"])

    for de_id, de_type in de_types.iter_rows():
        if de_type == "NUMBER":
            df = cast_numeric_data_element_value(df, de_id, pl.Float64)
        elif de_type == "INTEGER":
            df = cast_numeric_data_element_value(df, de_id, pl.Int64)
        elif de_type == "INTEGER_ZERO_OR_POSITIVE":
            df = cast_numeric_data_element_value(df, de_id, pl.Int64, min_value=0)
        # Add more types as needed

    return df


if __name__ == "__main__":
    dhis2_data_sync_push()
