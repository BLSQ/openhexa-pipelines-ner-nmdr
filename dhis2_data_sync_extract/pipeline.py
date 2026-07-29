from pathlib import Path

import polars as pl
from d2d_development.extract import DHIS2Extractor
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import get_organisation_unit_groups
from utils import (
    connect_to_dhis2,
    get_extract_periods,
    load_configuration,
    resolve_dates_and_validate,
    save_to_parquet,
)

# Ticket:
# - https://bluesquare.atlassian.net/browse/SMASDUNER-210

# GitHub:
# -https://github.com/BLSQ/openhexa-pipelines-ner-nmdr


@pipeline("dhis2_data_sync_extract")
@parameter(
    code="start_date",
    name="Start date (format: YYYYMM)",
    type=str,
    required=False,
    help=(
        "Start date for data extraction in YYYYMM format. "
        "If not set, it will default to current date minus NUMBER_MONTHS_WINDOW."
    ),
    default=None,
)
@parameter(
    code="end_date",
    name="End date (format: YYYYMM)",
    type=str,
    required=False,
    help=(
        "End date for data extraction in YYYYMM format. "
        "If not set, it will default to current date minus NUMBER_MONTHS_WINDOW."
    ),
    default=None,
)
@parameter(
    code="download_mode",
    name="Download mode",
    type=str,
    required=False,
    help=(
        "Download mode for data extraction. "
        "Options: 'DOWNLOAD_NEW' (default) to download only new data, "
        "'DOWNLOAD_REPLACE' to replace existing data."
    ),
    choices=["DOWNLOAD_REPLACE", "DOWNLOAD_NEW"],
    default="DOWNLOAD_REPLACE",
)
def dhis2_data_sync_extract(start_date: str, end_date: str, download_mode: str) -> None:
    """Write your pipeline orchestration here."""
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_data_sync_extract"
    config = load_configuration(pipeline_path / "configuration" / "extract_config.json")
    dhis2_client = connect_to_dhis2(connection_str=config["SETTINGS"]["SOURCE_DHIS2_CONNECTION"])

    # NOTE: We skip the download of the pyramid as there's no pyramid alignment for this particular project.
    # try:
    #     extract_pyramid(
    #         pipeline_path=pipeline_path,
    #         dhis2_snis_client=dhis2_client,
    #     )
    # except Exception as e:
    #     current_run.log_error(f"An error occurred data retrieval: {e}")
    #     raise

    try:
        extract_prs_data(
            pipeline_path=pipeline_path,
            dhis2_client=dhis2_client,
            config=config,
            download_mode=download_mode,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        current_run.log_error(f"An error occurred data retrieval: {e}")
        raise


def extract_pyramid(pipeline_path: Path, dhis2_snis_client: DHIS2) -> None:
    """Extracts the SNIS DHIS2 pyramid data and saves it as a Parquet file."""
    current_run.log_info("Retrieving SNIS DHIS2 pyramid data")
    pyramid_path = pipeline_path / "data" / "pyramid" / "snis_pyramid.parquet"

    try:
        # retrieve full pyramid
        org_units = dhis2_snis_client.meta.organisation_units(
            fields="id,name,shortName,openingDate,closedDate,parent,level,path,geometry"
        )
    except Exception as e:
        raise Exception(f"Error while retrieving SNIS DHIS2 Pyramid: {e}") from e

    org_units = pl.DataFrame(org_units)
    org_units = org_units.sort("level", descending=False)
    current_run.log_info(f"Retrieved {org_units['id'].n_unique()} organisation units")

    # Save as Parquet
    save_to_parquet(data=org_units, filename=pyramid_path)
    current_run.log_info(f"DHIS2 pyramid data saved: {pyramid_path}")


def extract_prs_data(
    pipeline_path: Path,
    dhis2_client: DHIS2,
    config: dict,
    download_mode: str,
    start_date: str,
    end_date: str,
) -> None:
    """Extracts the PRS data from DHIS2 and saves it as Parquet files."""
    current_run.log_info("Retrieving PRS data from DHIS2")

    # Resolve dates and validate
    start_date, end_date = resolve_dates_and_validate(start_date, end_date, config)
    extract_periods = get_extract_periods(start_date, end_date)
    current_run.log_info(f"Extracting data for periods: {extract_periods}")

    # For NER we use OUG to determine the org units to extract.
    org_units_groups = get_organisation_unit_groups(dhis2_client)

    for extract_config in config.get("DATA_ELEMENTS", {}).get("EXTRACTS", []):
        handle_data_element_extract(
            pipeline_path=pipeline_path,
            dhis2_client=dhis2_client,
            extract_config=extract_config,
            download_mode=download_mode,
            extract_periods=extract_periods,
            org_units_groups=org_units_groups,
        )


def handle_data_element_extract(
    pipeline_path: Path,
    dhis2_client: DHIS2,
    extract_config: dict,
    download_mode: str,
    extract_periods: list,
    org_units_groups: pl.DataFrame,
) -> None:
    """Handles data elements extracts based on the configuration.

    Args:
        pipeline_path: Root path of the pipeline, used to build each extract's output directory.
        dhis2_client: An instance of the DHIS2 client used to retrieve data.
        extract_config: Configuration for the data elements extract including data element UIDs.
        download_mode: Mode for downloading data, either "DOWNLOAD_NEW" or "DOWNLOAD_REPLACE".
        org_units_groups: Org units groups DataFrame used to determine which org units to extract data for.
        extract_periods: List of periods (YYYYMM) to extract data for.
    """
    current_run.log_info("Starting data element extracts.")
    current_run.log_info(f"Download mode: {download_mode}")

    data_element_uids = extract_config.get("UIDS", [])
    if not data_element_uids:
        current_run.log_info("No data element to extract.")
        return

    dhis2_extractor = DHIS2Extractor(dhis2_client=dhis2_client, download_mode=download_mode)
    extract_id = extract_config.get("EXTRACT_UID")
    if not extract_id:
        current_run.log_warning("No 'EXTRACT_UID' defined. This is required, extract skipped.")
        return

    ou_groups = extract_config.get("ORG_UNIT_GROUPS", [])
    if not ou_groups:
        current_run.log_warning(f"No 'ORG_UNIT_GROUP' defined for extract: {extract_id}, extract skipped.")
        return

    frequency = extract_config.get("FREQUENCY", "MONTHLY")
    if frequency == "YEARLY":
        periods = sorted(set(str(p[:4]) for p in extract_periods))
    else:
        periods = extract_periods

    org_units = resolve_org_units(org_units_groups, ou_groups)
    current_run.log_info(
        f"Starting data elements extract ID: '{extract_id}' "
        f"with {len(data_element_uids)} data elements across {len(org_units)} org units."
    )

    for period in periods:
        try:
            dhis2_extractor.data_elements.download_period(
                data_elements=data_element_uids,
                org_units=org_units,
                period=period,
                output_dir=pipeline_path / "data" / "extracts" / f"{extract_id}",
                filename=f"data_{extract_id}_{period}.parquet",
            )

        except Exception:
            current_run.log_error(
                f"Extract {extract_id} download failed for period {period}, skipping to next extract."
            )
            break  # skip to next extract

        current_run.log_info(f"Extract {extract_id} finished.")


def resolve_org_units(org_units_groups: pl.DataFrame, ou_groups: list[str]) -> list:
    """Resolves the organisation units for one or more org unit groups.

    Args:
        org_units_groups: DataFrame containing organisation unit groups.
        ou_groups: List of organisation unit group IDs to resolve.

    Returns:
        A list of unique organisation unit IDs that belong to any of the specified org unit groups.
    """
    group_rows = org_units_groups.filter(pl.col("id").is_in(ou_groups))

    found_ids = set(group_rows["id"].to_list())
    for missing_group in ou_groups:
        if missing_group not in found_ids:
            current_run.log_warning(f"Organisation unit group '{missing_group}' not found.")

    if group_rows.is_empty():
        return []

    org_units = group_rows.select(pl.col("organisation_units")).to_series().explode().drop_nulls().unique().to_list()
    current_run.log_info(f"Resolved {len(org_units)} unique organisation units for {len(found_ids)} group(s).")
    return org_units


if __name__ == "__main__":
    dhis2_data_sync_extract()
