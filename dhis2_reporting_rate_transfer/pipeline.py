import logging
import time
from itertools import product
from pathlib import Path

import pandas as pd
import polars as pl
from d2d_library.db_queue import Queue
from d2d_library.dhis2.extract import DHIS2Extractor
from d2d_library.dhis2.org_unit_aligner import DHIS2PyramidAligner
from d2d_library.dhis2.push import DHIS2Pusher
from d2d_library.utils import (
    configure_logging_flush,
    connect_to_dhis2,
    get_extract_periods,
    load_configuration,
    log_invalid_org_units,
    read_parquet_extract,
    save_logs,
)
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2.dataframe import get_datasets

# TICKET:
# https://bluesquare.atlassian.net/browse/NMDRNER-118


@pipeline("dhis2_reporting_rate_transfer", timeout=28800)  # 8 hours timeout
@parameter(
    "push_orgunits",
    name="Sync organisation units",
    help="Create and update organisation units in the target DHIS2 instance.",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    code="run_extract_data",
    name="Extract data",
    type=bool,
    default=True,
    help="Extract data elements from source DHIS2.",
)
@parameter(
    code="run_push_data",
    name="Push data",
    type=bool,
    default=True,
    help="Push data to target DHIS2.",
)
def dhis2_reporting_rate_transfer(push_orgunits: bool, run_extract_data: bool, run_push_data: bool):
    """Write your pipeline orchestration here.

    Pipeline functions should only call tasks and should never perform IO operations or expensive computations.
    """
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_reporting_rate_transfer"

    try:
        sync_ready = sync_organisation_units(
            pipeline_path=pipeline_path,
            run_task=push_orgunits,
        )

        extract_data(
            pipeline_path=pipeline_path,
            run_task=run_extract_data,
            wait=sync_ready,
        )

        push_data(
            pipeline_path=pipeline_path,
            run_task=run_push_data,
            wait=sync_ready,
        )

    except Exception as e:
        current_run.log_error(f"An error occurred: {e}")
        raise


@dhis2_reporting_rate_transfer.task
def sync_organisation_units(pipeline_path: Path, run_task: bool) -> bool:
    """Handle creation and updates of organisation units in the target DHIS2 (incremental approach only).

    Use the previously extracted pyramid (full) stored as dataframe as input.
    The format of the pyramid contains the expected columns. A dataframe that doesn't contain the
    mandatory columns will be skipped (not valid).

    Args:
        pipeline_path (Path): Root path for pipeline logs.
        run_task (bool): Whether to run the task.

    Returns:
        bool: True if the task was run, False otherwise.
    """
    if not run_task:
        current_run.log_info("Organisation units sync task skipped.")
        return True

    current_run.log_info("Starting organisation units sync.")

    # load configuration
    config_extract = load_configuration(config_path=pipeline_path / "configuration" / "extract_config.json")
    config_push = load_configuration(config_path=pipeline_path / "configuration" / "push_config.json")
    logger, logs_file = configure_logging_flush(logs_path=Path("/home/jovyan/tmp/logs"), task_name="organisation_units")

    # DHIS2 connections
    dhis2_client_source = connect_to_dhis2(
        connection_str=config_extract["SETTINGS"]["SOURCE_DHIS2_CONNECTION"], cache_dir=None
    )
    dhis2_client_target = connect_to_dhis2(
        connection_str=config_push["SETTINGS"]["TARGET_DHIS2_CONNECTION"], cache_dir=None
    )

    # Load pyramid extract
    current_run.log_info(f"Retrieving organisation units from source DHIS2 instance {dhis2_client_source.api.url}")
    orgunit_source = dhis2_client_source.meta.organisation_units(
        fields="id,name,shortName,openingDate,closedDate,parent,level,path,geometry"
    )
    orgunit_source = pd.DataFrame(orgunit_source)

    # Create aligner
    org_units_aligner = DHIS2PyramidAligner(logger=logger)
    try:
        org_units_aligner.align_to(
            target_dhis2=dhis2_client_target,
            source_pyramid=orgunit_source,
            dry_run=config_push["SETTINGS"].get("DRY_RUN", True),
        )
        log_aligner_messages(org_units_aligner)
        current_run.log_info("Organisation units push finished.")

    except Exception as e:
        current_run.log_error(f"Organisation units sync failed with error: {e!s}")
        raise

    finally:
        save_logs(logs_file, output_dir=pipeline_path / "logs" / "organisation_units")
    return True


def log_aligner_messages(ou_aligner: DHIS2PyramidAligner) -> None:
    """Log messages from org_units_aligner summary."""
    # creates
    if ou_aligner.summary["CREATE"]["CREATE_COUNT"] > 0:
        for detail in ou_aligner.summary["CREATE"]["CREATE_DETAILS"]:
            current_run.log_info(f"New organisation unit created: {detail['OU']}")
    else:
        current_run.log_info("No new organisation units created.")

    # updates
    updates = ou_aligner.summary["UPDATE"]["UPDATE_COUNT"]
    if updates > 0:
        current_run.log_warning(f"{updates} organisation units were updated. Please check the latest execution logs.")
    else:
        current_run.log_info("No organisation units updated.")

    # errors
    errors_count = ou_aligner.summary["CREATE"]["ERROR_COUNT"] + ou_aligner.summary["UPDATE"]["ERROR_COUNT"]
    if errors_count > 0:
        current_run.log_info(f"{errors_count} errors occurred during sync. Please check the latest execution logs.")


@dhis2_reporting_rate_transfer.task
def extract_data(
    pipeline_path: Path,
    run_task: bool = True,
    wait: bool = True,
):
    """Extracts data from the source DHIS2 instance and saves them in parquet format."""
    if not run_task:
        current_run.log_info("data extraction task skipped.")
        return

    current_run.log_info("data extraction task started.")

    # load configuration
    extract_config = load_configuration(config_path=pipeline_path / "configuration" / "extract_config.json")
    extract_settings = extract_config.get("SETTINGS", {})

    # connect to source DHIS2 instance (No cache for data extraction)
    dhis2_client = connect_to_dhis2(connection_str=extract_settings.get("SOURCE_DHIS2_CONNECTION"), cache_dir=None)

    # initialize queue
    db_path = pipeline_path / "configuration" / ".queue.db"
    push_queue = Queue(db_path)

    download_settings = extract_settings.get("MODE", None)
    if download_settings is None:
        download_settings = "DOWNLOAD_REPLACE"
        current_run.log_warning(f"No 'MODE' found in extraction settings. Set default: {download_settings}")

    current_run.log_info(f"Download MODE: {extract_settings['MODE']}")

    # limits
    dhis2_client.data_value_sets.MAX_DATA_ELEMENTS = 100
    dhis2_client.data_value_sets.MAX_ORG_UNITS = 100

    # Setup extractor
    # See docs about return_existing_file impact.
    dhis2_extractor = DHIS2Extractor(
        dhis2_client=dhis2_client, download_mode=download_settings, return_existing_file=False
    )

    try:
        handle_reporting_rates_extracts(
            pipeline_path=pipeline_path,
            dhis2_extractor=dhis2_extractor,
            reporting_rates_extracts=extract_config["REPORTING_RATES"].get("EXTRACTS", []),
            push_queue=push_queue,
        )
    finally:
        push_queue.enqueue("FINISH")


def handle_reporting_rates_extracts(
    pipeline_path: Path,
    dhis2_extractor: DHIS2Extractor,
    reporting_rates_extracts: list,
    push_queue: Queue,
):
    """Handles reporting rates extracts based on the configuration."""
    if len(reporting_rates_extracts) == 0:
        current_run.log_info("No reporting rates to extract.")
        return

    logger, logs_file = configure_logging_flush(logs_path=Path("/home/jovyan/tmp/logs"), task_name="extracts")
    current_run.log_info("Starting reporting rates extracts.")
    try:
        source_datasets = get_datasets(dhis2_extractor.dhis2_client)

        # loop over the available extract configurations
        for idx, extract in enumerate(reporting_rates_extracts):
            extract_id = extract.get("EXTRACT_UID")
            dataset_id = extract.get("DATASET_UID", None)
            metrics = extract.get("METRICS", None)
            period_window = extract.get("PERIOD_WINDOW", 3)
            period_freq = extract.get("PERIOD_FREQUENCY", None)
            period_start = extract.get("PERIOD_START")
            period_end = extract.get("PERIOD_END")

            if extract_id is None:
                current_run.log_warning(
                    f"No 'EXTRACT_UID' defined at position: {idx}. This is required, extract skipped."
                )
                continue

            if dataset_id is None:
                current_run.log_warning(f"No 'DATASET_UID' defined for extract: {extract_id}, extract skipped.")
                continue

            if len(metrics) == 0:
                current_run.log_warning(f"No dataset metrics defined for extract: {extract_id}, extract skipped.")
                continue

            # Select org units from pyramid or from dataset OU definition
            # We prefer dataset OUs if dataset_id is provided
            source_dataset = source_datasets.filter(pl.col("id").is_in([dataset_id]))
            org_units = source_dataset["organisation_units"][0].to_list()
            current_run.log_info(
                f"Starting data elements extract ID: '{extract_id}' ({idx + 1}) "
                f"with {len(metrics)} metrics across {len(org_units)} org units from dataset: "
                f"'{source_dataset['name'][0]}' ({dataset_id})."
            )

            try:
                # get extract periods
                extract_periods = get_extract_periods(
                    start=period_start, end=period_end, period_window=period_window, frequency=period_freq
                )
                current_run.log_info(
                    f"Download extract period from: {extract_periods[0]} to {extract_periods[-1]} "
                    f"frequency: {period_freq}."
                )
                logger.info(f"Download extract period from: {extract_periods[0]} to {extract_periods[-1]}")
            except Exception as e:
                current_run.log_warning(
                    f"Failed to compute extract periods for extract: {extract_id}, skipping extract."
                )
                logger.error(f"Extract {extract_id} - period computation error: {e}")
                continue

            # define reporting rates to extract
            reporting_rates = [f"{ds}.{metric}" for ds, metric in product([dataset_id], metrics)]

            # run extraction per period
            for period in extract_periods:
                try:
                    extract_path = dhis2_extractor.reporting_rates.download_period(
                        reporting_rates=reporting_rates,
                        org_units=org_units,
                        period=period,
                        output_dir=pipeline_path / "data" / "extracts" / "reporting_rates" / f"extract_{extract_id}",
                    )
                    if extract_path is not None:
                        push_queue.enqueue(f"{extract_id}|{extract_path}")

                except Exception as e:
                    current_run.log_warning(
                        f"Extract {extract_id} download failed for period {period}, skipping to next extract."
                    )
                    logger.error(f"Extract {extract_id} - period {period} error: {e}")
                    break  # skip to next extract

            current_run.log_info(f"Extract {extract_id} finished.")

    finally:
        save_logs(logs_file, output_dir=pipeline_path / "logs" / "extract")


@dhis2_reporting_rate_transfer.task
def push_data(
    pipeline_path: Path,
    run_task: bool = True,
    wait: bool = True,
):
    """Pushes data points to the target DHIS2 instance."""
    if not run_task:
        current_run.log_info("Data push task skipped.")
        return

    current_run.log_info("Starting data push.")

    # setup
    logger, logs_file = configure_logging_flush(logs_path=Path("/home/jovyan/tmp/logs"), task_name="push_data")
    push_config = load_configuration(config_path=pipeline_path / "configuration" / "push_config.json")
    settings = push_config.get("SETTINGS", {})
    target_dhis2 = connect_to_dhis2(connection_str=settings["TARGET_DHIS2_CONNECTION"], cache_dir=None)
    db_path = pipeline_path / "configuration" / ".queue.db"
    push_queue = Queue(db_path)

    # log parameters
    msg = (
        f"Import strategy: {settings.get('IMPORT_STRATEGY', 'CREATE_AND_UPDATE')} - "
        f"Dry Run: {settings.get('DRY_RUN', True)} - "
        f"Max Post elements: {settings.get('MAX_POST', 500)}"
    )
    logger.info(msg)
    current_run.log_info(msg)

    # Set up DHIS2 pusher
    pusher = DHIS2Pusher(
        dhis2_client=target_dhis2,
        import_strategy=settings.get("IMPORT_STRATEGY", "CREATE_AND_UPDATE"),
        dry_run=settings.get("DRY_RUN", True),
        max_post=settings.get("MAX_POST", 500),
        logger=logger,
    )

    # Map data types to their respective mapping functions
    dispatch_map = {
        "DATA_ELEMENT": (push_config["DATA_ELEMENTS"]["EXTRACTS"], apply_data_element_extract_config),
        "REPORTING_RATE": (push_config["REPORTING_RATES"]["EXTRACTS"], apply_reporting_rates_extract_config),
        "INDICATOR": (push_config["INDICATORS"]["EXTRACTS"], apply_indicators_extract_config),
    }

    conflict_list = []
    push_wait = settings.get("PUSH_WAIT_MINUTES", 5)
    # loop over the queue
    while True:
        next_extract = push_queue.peek()
        if next_extract == "FINISH":
            push_queue.dequeue()  # remove marker if present
            break

        if not next_extract:
            current_run.log_info("Push data process: waiting for updates")
            time.sleep(60 * int(push_wait))
            continue

        try:
            # Read extract
            extract_id, extract_file_path = split_on_pipe(next_extract)
            extract_path = Path(extract_file_path)
            extract_data = read_parquet_extract(parquet_file=extract_path)
        except Exception as e:
            current_run.log_error(f"Failed to read extract from queue item: {next_extract}.")
            logger.info(f"Failed to read extract from queue item: {next_extract}. Error: {e}")
            push_queue.dequeue()  # remove problematic item
            continue

        try:
            # Determine data type
            data_type = extract_data["DATA_TYPE"].unique()[0]

            current_run.log_info(f"Pushing data for extract {extract_id}: {extract_path.name}.")
            logger.info(f"Pushing data for extract {extract_id}: {extract_path.name}.")
            if data_type not in dispatch_map:
                current_run.log_warning(f"Unknown DATA_TYPE '{data_type}' in extract: {extract_path.name}. Skipping.")
                push_queue.dequeue()  # remove unknown item
                continue

            # Get config and mapping function
            cfg_list, mapper_func = dispatch_map[data_type]
            extract_config = next((e for e in cfg_list if e["EXTRACT_UID"] == extract_id), {})

            # Apply mapping and push data
            df_mapped: pd.DataFrame = mapper_func(df=extract_data, extract=extract_config)
            pusher.push_data(df_data=df_mapped)
            conflict_list.extend(pusher.summary["ERRORS"])

            # Success → dequeue
            push_queue.dequeue()

        except Exception as e:
            current_run.log_error(
                f"Fatal error for extract {extract_id} ({extract_path.name}), stopping push process. Error: {e!s}"
            )
            raise  # crash on error

        finally:
            log_invalid_org_units(conflict_list, logger)
            save_logs(logs_file, output_dir=pipeline_path / "logs" / "push")

    current_run.log_info("Push process finished.")


def apply_data_element_extract_config(
    df: pd.DataFrame,
    extract: dict,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Handles the mappings of reporting rates."""
    raise NotImplementedError("Indicator mappings is not yet implemented.")


def apply_reporting_rates_extract_config(
    df: pd.DataFrame,
    extract: dict,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Handles the mappings of reporting rates.

    Returns
    -------
    pd.DataFrame
        Mapped dataframe according to the extract configuration.
    """
    if len(extract) == 0:
        current_run.log_warning("No extract details provided, skipping reporting rates mappings.")
        return df

    extract_mappings = extract.get("MAPPINGS", {})
    if not extract_mappings:
        current_run.log_warning("No extract mappings provided, skipping reporting rates mappings.")
        return df

    df_mapped_list = []
    for key, mapping in extract_mappings.items():
        key_mask = df["DX_UID"] == key
        if not key_mask.any():
            continue
        for metric_name, metric_mappings in mapping.items():
            new_uid, new_coc, new_aoc = validate_reporting_rate_mappings(metric_mappings, key, metric_name)
            metric_mask = key_mask & (df["RATE_TYPE"] == metric_name)
            if not metric_mask.any():
                continue
            df_metric = df[metric_mask].copy()
            df_metric["DX_UID"] = new_uid
            df_metric["CATEGORY_OPTION_COMBO"] = new_coc
            df_metric["ATTRIBUTE_OPTION_COMBO"] = new_aoc
            df_mapped_list.append(df_metric)

    if not df_mapped_list:
        current_run.log_warning("No reporting rates matched the provided mappings, returning empty dataframe.")
        logger.warning("No reporting rates matched the provided mappings, returning empty dataframe.")
        return pd.DataFrame(columns=df.columns)

    df_mapped = pd.concat(df_mapped_list, ignore_index=True)

    # NOTE: Specific case: convert 100 reporting rate to 1
    df_mapped["VALUE"] = pd.to_numeric(df_mapped["VALUE"], errors="coerce")
    df_mapped.loc[(df_mapped["RATE_TYPE"] == "REPORTING_RATE") & (df_mapped["VALUE"] == 100), "VALUE"] = 1
    return df_mapped.sort_values(by="ORG_UNIT")


def validate_reporting_rate_mappings(mappings: dict, key: str, metric_name: str) -> tuple[str, str, str]:
    """Validates the reporting rate mappings and returns the mapped values.

    Parameters
    ----------
    mappings : dict
        The mappings dictionary containing the required fields.
    key : str
        The dataset key for logging purposes.
    metric_name : str
        The metric name for logging purposes.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing the UID, CAT_OPTION_COMBO, and ATTR_OPTION_COMBO.
    """
    required_fields = {
        "UID": "UID",
        "CAT_OPTION_COMBO": "CAT_OPTION_COMBO",
        "ATTR_OPTION_COMBO": "ATTR_OPTION_COMBO",
    }
    values = {}

    for field, label in required_fields.items():
        value = mappings.get(field)
        if value is None:
            raise ValueError(f"Missing '{label}' mapping for dataset '{key}' reporting rate metric '{metric_name}'.")
        values[field] = str(value).strip()

    return values["UID"], values["CAT_OPTION_COMBO"], values["ATTR_OPTION_COMBO"]


def apply_indicators_extract_config(
    df: pd.DataFrame,
    extract: dict,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Handles the mappings of reporting rates."""
    raise NotImplementedError("Data Element mappings is not yet implemented.")


def split_on_pipe(s: str) -> tuple[str, str | None]:
    """Splits a string on the first pipe character and returns a tuple.

    Parameters
    ----------
    s : str
        The string to split.

    Returns
    -------
    tuple[str, str | None]
        A tuple containing the part before the pipe and the part after the pipe (or None if no pipe is found).
    """
    parts = s.split("|", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, parts[0]


if __name__ == "__main__":
    dhis2_reporting_rate_transfer()
