import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from utils import (
    Queue,
    read_parquet_extract,
    split_list,
    load_files_with_pattern,
)


@pipeline("dhis2_nmdr_push", timeout=28800)
@parameter(
    "push_orgunits",
    name="Export Organisation Units",
    help="",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "push_analytics",
    name="Export analytics",
    help="",
    type=bool,
    default=True,
    required=False,
)
def dhis2_pnlp_push(push_orgunits: bool, push_analytics: bool):
    """Pipeline to push the extracted data from SNIS DHIS2 to the PNLP DHIS2.

    Most of the tasks and functions are specific to this project.
    """
    # set paths
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_nmdr_push"
    extract_pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_snis_extract"
    try:
        # load config
        config = load_configuration(pipeline_path=pipeline_path)

        # connect to DHIS2
        dhis2_client = connect_to_dhis2(config=config, cache_dir=workspace.files_path)

        # push_organisation_units(
        #     root=pipeline_path,
        #     extract_pipeline_path=extract_pipeline_path,
        #     dhis2_client_target=dhis2_client,
        #     config=config,
        #     run=push_orgunits,
        # )

        push_extracts(
            root=pipeline_path,
            extract_pipeline_path=extract_pipeline_path,
            dhis2_client_target=dhis2_client,
            config=config,
            run=push_analytics,
        )

    except Exception as e:
        current_run.log_error(f"An error occurred: {e}")


def load_configuration(pipeline_path: str) -> dict:
    """Reads a JSON file configuration and returns its contents as a dictionary.

    Args:
        pipeline_path (str): Root path of the pipeline to find the file.

    Returns:
        dict: Dictionary containing the JSON data.
    """
    try:
        file_path = pipeline_path / "config" / "nmdr_push_config.json"
        with Path.open(file_path, "r") as file:
            data = json.load(file)

        current_run.log_info("Configuration loaded.")
        return data
    except FileNotFoundError as e:
        raise Exception(f"The file '{file_path}' was not found {e}") from e
    except json.JSONDecodeError as e:
        raise Exception(f"Error decoding JSON: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error while loading configuration '{file_path}' {e}") from e


def connect_to_dhis2(config: dict, cache_dir: str) -> DHIS2:
    """Connects to a DHIS2 instance using the provided configuration and cache directory.

    Args:
        config (dict): Configuration dictionary containing DHIS2 connection settings.
        cache_dir (str): Directory path for caching.

    Returns:
        DHIS2: An instance of the DHIS2 client.

    Raises:
        Exception: If there is an error while connecting to DHIS2.
    """
    try:
        connection = workspace.dhis2_connection(config["PUSH_SETTINGS"]["DHIS2_CONNECTION"])
        dhis2_client = DHIS2(connection=connection, cache_dir=cache_dir)
        current_run.log_info(f"Connected to DHIS2 connection: {config['PUSH_SETTINGS']['DHIS2_CONNECTION']}")
        return dhis2_client
    except Exception as e:
        raise Exception(f"Error while connecting to DHIS2 {config['PUSH_SETTINGS']['DHIS2_CONNECTION']}: {e}") from e


def push_organisation_units(
    root: Path, extract_pipeline_path: Path, dhis2_client_target: DHIS2, config: dict, run: bool
) -> bool:
    """Handle creation and updates of organisation units in the target DHIS2 (incremental approach only).

    Use the previously extracted pyramid (full) stored as dataframe as input.
    The format of the pyramid contains the expected columns. A dataframe that doesn't contain the
    mandatory columns will be skipped (not valid).

    Args:
        root (str): Root path for pipeline logs.
        extract_pipeline_path (str): Path to the extract pipeline.
        dhis2_client_target (DHIS2): Target DHIS2 client.
        run_task (bool): Whether to run the task.

    Returns:
        bool: True if the task ran or was skipped successfully.
    """
    if not run:
        return True

    current_run.log_info("Starting organisation units push.")
    report_path = Path(root) / "logs" / "organisationUnits"
    configure_login(logs_path=report_path, task_name="organisation_units")

    # Load pyramid extract
    ou_parquet = Path(extract_pipeline_path) / "data" / "pyramid" / "snis_pyramid.parquet"
    orgunit_source = read_parquet_extract(ou_parquet)

    if orgunit_source.shape[0] > 0:
        # Retrieve the target (PNLP) orgUnits to compare
        current_run.log_info(f"Retrieving organisation units from target DHIS2 instance {dhis2_client_target.api.url}")
        orgunit_target = dhis2_client_target.meta.organisation_units(
            fields="id,name,shortName,openingDate,closedDate,parent,level,path,geometry"
        )
        orgunit_target = pd.DataFrame(orgunit_target)

        # Get list of ids for creation and update
        ou_new = list(set(orgunit_source.id) - set(orgunit_target.id))
        ou_matching = list(set(orgunit_source.id).intersection(set(orgunit_target.id)))
        dhsi2_version = dhis2_client_target.meta.system_info().get("version")

        # Create orgUnits
        try:
            if len(ou_new) > 0:
                current_run.log_info(f"Creating {len(ou_new)} organisation units.")
                ou_to_create = orgunit_source[orgunit_source.id.isin(ou_new)]
                # NOTE: Geometry is valid for versions > 2.32
                if dhsi2_version <= "2.32":
                    ou_to_create["geometry"] = None
                    current_run.log_warning("DHIS2 version not compatible with geometry. Geometry will be ignored.")
                push_orgunits_create(
                    ou_df=ou_to_create,
                    dhis2_client_target=dhis2_client_target,
                    report_path=report_path,
                    dry_run=config["PUSH_SETTINGS"].get("DRY_RUN", True),
                )
        except Exception as e:
            raise Exception(f"Unexpected error occurred while creating organisation units. Error: {e}") from e

        # Update orgUnits
        try:
            if len(ou_matching) > 0:
                current_run.log_info(f"Checking for updates in {len(ou_matching)} organisation units")
                # NOTE: Geometry is valid for versions > 2.32
                if dhsi2_version <= "2.32":
                    orgunit_source["geometry"] = None
                    orgunit_target["geometry"] = None
                    current_run.log_warning("DHIS2 version not compatible with geometry. Geometry will be ignored.")
                push_orgunits_update(
                    orgUnit_source=orgunit_source,
                    orgUnit_target=orgunit_target,
                    matching_ou_ids=ou_matching,
                    dhis2_client_target=dhis2_client_target,
                    report_path=report_path,
                    dry_run=config["PUSH_SETTINGS"].get("DRY_RUN", True),
                )
                current_run.log_info("Organisation units push finished.")
        except Exception as e:
            raise Exception(f"Unexpected error occurred while updating organisation units. Error: {e}") from e

    else:
        current_run.log_warning("No data found in the pyramid file. Organisation units task skipped.")

    return True


def push_extracts(
    root: Path,
    extract_pipeline_path: Path,
    dhis2_client_target: DHIS2,
    config: dict,
    run: bool,
):
    """Put some data processing code here."""
    if not run:
        return True

    current_run.log_info("Starting analytic extracts push.")
    report_path = root / "logs" / "extracts"
    configure_login(logs_path=report_path, task_name="extracts")
    db_path = extract_pipeline_path / "config" / ".queue.db"

    # Parameters for the import
    import_strategy = config["PUSH_SETTINGS"].get("IMPORT_STRATEGY")
    if import_strategy is None:
        import_strategy = "CREATE_AND_UPDATE"  # CREATE, UPDATE, CREATE_AND_UPDATE

    dry_run = config["PUSH_SETTINGS"].get("DRY_RUN")
    if dry_run is None:
        dry_run = True  # True or False

    max_post = config["PUSH_SETTINGS"].get("MAX_POST", None)
    if max_post is None:
        max_post = 500  # number of datapoints without a time-out limit (?)

    # log parameters
    logging.info(f"Import strategy: {import_strategy} - Dry Run: {dry_run} - Max Post elements: {max_post}")
    current_run.log_info(
        f"Pushing data elemetns with parameters import_strategy: {import_strategy}, dry_run: {dry_run}, max_post: {max_post}"
    )

    # initialize push queue (look the util file -> pipelines/dhis2_nmdr_push/config/db_reset_utils.ipynb)
    push_queue = Queue(db_path)

    try:
        dataelement_mappings = config.get("DATA_ELEMENT_MAPPINGS")
        if dataelement_mappings is None:
            raise ValueError("No DATA_ELEMENT_MAPPINGS found in the configuration file.")
        rate_mappings = config.get("RATE_MAPPINGS")
        if rate_mappings is None:
            raise ValueError("No RATE_MAPPINGS found in the configuration file.")
        indicator_mappings = config.get("INDICATOR_MAPPINGS")  ### TO BE IMPLEMENTED !!
        # if indicator_mappings is None:
        #     raise ValueError("No INDICATOR_MAPPINGS found in the configuration file.")

        while True:
            next_period = push_queue.peek()  # I dont remove yet, just take a look
            if not next_period:
                break

            try:
                extract_data = load_files_with_pattern(
                    path=extract_pipeline_path / "data" / "extracts", pattern=f"*_extract_{next_period}.parquet"
                )
                current_run.log_info(f"Push extract period: {next_period}.")
            except Exception as e:
                current_run.log_warning(
                    f"Error while reading the extracts file: snis_data_{next_period}.parquet - error: {e}"
                )
                continue

            # Use dictionary mappings to replace UIDS, OrgUnits, COC and AOC..
            df = apply_dataelement_mappings(datapoints_df=extract_data, mappings=dataelement_mappings)
            df = apply_rate_mappings(datapoints_df=df, mappings=rate_mappings)
            # df = apply_indicator_mappings(datapoints_df=df, mappings=acm_mappings)

            # Set values of 'INDICATOR' to INT. Otherwise is ignored by DHIS2
            indicator_mask = df.data_type == "INDICATOR"
            df.loc[indicator_mask, "value"] = df.loc[indicator_mask, "value"].astype(float).astype(int).astype(str)

            # mandatory fields in the input dataset
            mandatory_fields = [
                "dx_uid",
                "period",
                "org_unit",
                "category_option_combo",
                "attribute_option_combo",
                "value",
            ]
            df = df[mandatory_fields]

            # Sort the dataframe by org_unit to reduce the time (hopefully)
            df = df.sort_values(by=["org_unit"], ascending=True)

            # convert the datapoints to json and check if they are valid
            # check the implementation of DataPoint class for the valid fields (mandatory)
            datapoints_valid, datapoints_not_valid, datapoints_to_na = select_transform_to_json(data_values=df)

            # log not valid datapoints
            log_ignored_or_na(report_path=report_path, datapoint_list=datapoints_not_valid)

            # datapoints set to NA
            if len(datapoints_to_na) > 0:
                log_ignored_or_na(
                    report_path=report_path, datapoint_list=datapoints_to_na, data_type="extract", is_na=True
                )
                summary_na = push_data_elements(
                    dhis2_client=dhis2_client_target,
                    data_elements_list=datapoints_to_na,  # json for deletion see: DataPoint class to_delete_json()
                    strategy=import_strategy,
                    dry_run=dry_run,
                    max_post=max_post,
                )

                # log info
                msg = f"Data elements delete summary:  {summary_na['import_counts']}"
                current_run.log_info(msg)
                logging.info(msg)
                log_summary_errors(summary_na)

            current_run.log_info(f"Pushing {len(datapoints_valid)} valid data elements for period {next_period}.")
            # push data
            summary = push_data_elements(
                dhis2_client=dhis2_client_target,
                data_elements_list=datapoints_valid,
                strategy=import_strategy,
                dry_run=dry_run,
                max_post=max_post,
            )

            # The process is correct, so we remove the period from the queue
            _ = push_queue.dequeue()

            # log info
            msg = f"Analytics extracts summary for period {next_period}: {summary['import_counts']}"
            current_run.log_info(msg)
            logging.info(msg)
            log_summary_errors(summary)

        current_run.log_info("No more extracts to push.")

    except Exception as e:
        raise Exception(f"Analytic extracts task error: {e}") from e


def configure_login(logs_path: Path, task_name: str):
    # Configure logging
    logs_path.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d-%H_%M")
    logging.basicConfig(
        filename=logs_path / f"{task_name}_{now}.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )


def log_summary_errors(summary: dict):
    """Logs all the errors in the summary dictionary using the configured logging.

    Args:
        summary (dict): The dictionary containing import counts and errors.
    """
    errors = summary.get("ERRORS", [])
    if not errors:
        logging.info("No errors found in the summary.")
    else:
        logging.error(f"Logging {len(errors)} error(s) from export summary.")
        for i_e, error in enumerate(errors, start=1):
            logging.error(f"Error {i_e} : HTTP request failed : {error.get('error', None)}")
            error_response = error.get("response", None)
            if error_response:
                rejected_list = error_response.pop("rejected_datapoints", [])
                logging.error(f"Error response : {error_response}")
                for i_r, rejected in enumerate(rejected_list, start=1):
                    logging.error(f"Rejected data point {i_r}: {rejected}")


# Helper class definition to store/create the correct OU JSON format for creation/update
class OrgUnitObj:
    def __init__(self, orgUnit_row: pd.Series):
        """Create a new org unit instance.

        Parameters
        ----------
        orgUnit_row : pandas series
            Expects columns with names :
                ['id', 'name', 'shortName', 'openingDate', 'closedDate', 'parent','level', 'path', 'geometry']
        """
        self.initialize_from(orgUnit_row.squeeze(axis=0))

    def initialize_from(self, row: pd.Series):
        # let's keep names consistent
        self.id = row.get("id")
        self.name = row.get("name")
        self.shortName = row.get("shortName")
        self.openingDate = row.get("openingDate")
        self.closedDate = row.get("closedDate")
        self.parent = row.get("parent")
        geometry = row.get("geometry")
        self.geometry = json.loads(geometry) if isinstance(geometry, str) else geometry

    def to_json(self) -> dict:
        json_dict = {
            "id": self.id,
            "name": self.name,
            "shortName": self.shortName,
            "openingDate": self.openingDate,
            "closedDate": self.closedDate,
            "parent": {"id": self.parent.get("id")} if self.parent else None,
        }
        if self.geometry:
            geometry = json.loads(self.geometry) if isinstance(self.geometry, str) else self.geometry
            json_dict["geometry"] = {
                "type": geometry["type"],
                "coordinates": geometry["coordinates"],
            }
        return {k: v for k, v in json_dict.items() if v is not None}

    def is_valid(self):
        if self.id is None:
            return False
        if self.name is None:
            return False
        if self.shortName is None:
            return False
        if self.openingDate is None:
            return False
        if self.parent is None:
            return False

        return True

    def __str__(self):
        return f"OrgUnitObj({self.id}, {self.name})"

    def copy(self):
        attributes = self.to_json()
        new_instance = OrgUnitObj(pd.Series(attributes))
        return new_instance


# Helper class definition to store/create the correct DataElement JSON format
class DataPoint:
    def __init__(self, series_row: pd.Series):
        """Create a new org unit instance.

        Parameters
        ----------
        series_row : pandas series
            Expects columns with names :
                ['data_type',
                'dx_uid',
                'period',
                'org_unit',
                'category_option_combo',
                'attribute_option_combo',
                'rate_type',
                'domain_type',
                'value']
        """
        row = series_row.squeeze(axis=0)
        self.dataType = row.get("data_type")
        self.dataElement = row.get("dx_uid")
        self.period = row.get("period")
        self.orgUnit = row.get("org_unit")
        self.categoryOptionCombo = row.get("category_option_combo")
        self.attributeOptionCombo = row.get("attribute_option_combo")
        self.value = row.get("value")

    def to_json(self) -> dict:
        json_dict = {
            "dataElement": self.dataElement,
            "period": self.period,
            "orgUnit": self.orgUnit,
            "categoryOptionCombo": self.categoryOptionCombo,
            "attributeOptionCombo": self.attributeOptionCombo,
            "value": self.value,
        }
        # return {k: v for k, v in json_dict.items() if v is not None}
        return json_dict

    def to_delete_json(self) -> dict:
        json_dict = {
            "dataElement": self.dataElement,
            "period": self.period,
            "orgUnit": self.orgUnit,
            "categoryOptionCombo": self.categoryOptionCombo,
            "attributeOptionCombo": self.attributeOptionCombo,
            "value": "",
            "comment": "deleted value",
        }
        return json_dict

    def _check_attributes(self, exclude_value=False):
        # List of attributes to check, optionally excluding the 'value' attribute
        attributes = [self.dataElement, self.period, self.orgUnit, self.categoryOptionCombo, self.attributeOptionCombo]
        if not exclude_value:
            attributes.append(self.value)

        # Return True if all attributes are not None
        return all(attr is not None for attr in attributes)

    def is_valid(self):
        # Check if all attributes are valid (None check)
        return self._check_attributes(exclude_value=False)

    def is_to_delete(self):
        # Check if all attributes except 'value' are not None and 'value' is None
        return self._check_attributes(exclude_value=True) and self.value is None

    def __str__(self):
        return f"DataPoint({self.dataType} id:{self.dataElement} pe:{self.period} ou:{self.orgUnit} value:{self.value})"


def push_orgunits_create(ou_df: pd.DataFrame, dhis2_client_target: DHIS2, report_path: str, dry_run: bool):
    errors_count = 0
    for _, row in ou_df.iterrows():
        ou = OrgUnitObj(row)
        if ou.is_valid():
            response = push_orgunit(
                dhis2_client=dhis2_client_target,
                orgunit=ou,
                strategy="CREATE",
                dry_run=dry_run,  # dry_run=False -> Apply changes in the DHIS2
            )
            if response["status"] == "ERROR":
                errors_count = errors_count + 1
                logging.info(str(response))
            else:
                current_run.log_info(f"New organisation unit created: {ou}")
        else:
            logging.info(
                str(
                    {
                        "action": "CREATE",
                        "statusCode": None,
                        "status": "NOTVALID",
                        "response": None,
                        "ou_id": row.get("id"),
                    }
                )
            )

    if errors_count > 0:
        current_run.log_info(
            f"{errors_count} errors occurred during creation. Please check the latest execution report under {report_path}."
        )


def push_orgunits_update(
    orgunit_source: pd.DataFrame,
    orgunit_target: pd.DataFrame,
    matching_ou_ids: list,
    dhis2_client_target: DHIS2,
    report_path: str,
    dry_run: bool,
):
    """Update org units based matching id list"""
    # Use these columns to compare (check for Updates)
    comparison_cols = [
        "name",
        "shortName",
        "openingDate",
        "closedDate",
        "parent",
        "geometry",
    ]

    # build id dictionary (faster) and compare on selected columns
    index_dictionary = build_id_indexes(orgunit_source, orgunit_target, matching_ou_ids)
    orgunit_source_f = orgunit_source[comparison_cols]
    orgunit_target_f = orgunit_target[comparison_cols]

    errors_count = 0
    updates_count = 0
    progress_count = 0
    for uid, indices in index_dictionary.items():
        progress_count = progress_count + 1
        source = orgunit_source_f.iloc[indices["source"]]
        target = orgunit_target_f.iloc[indices["target"]]
        # get cols with differences
        diff_fields = source[~((source == target) | (source.isna() & target.isna()))]

        # If there are differences update!
        if not diff_fields.empty:
            # add the ID for update
            source["id"] = uid
            ou_update = OrgUnitObj(source)
            response = push_orgunit(
                dhis2_client=dhis2_client_target,
                orgunit=ou_update,
                strategy="UPDATE",
                dry_run=dry_run,  # dry_run=False -> Apply changes in the DHIS2
            )
            if response["status"] == "ERROR":
                errors_count = errors_count + 1
            else:
                updates_count = updates_count + 1
            logging.info(str(response))

        if progress_count % 5000 == 0:
            current_run.log_info(f"Organisation units checked: {progress_count}/{len(matching_ou_ids)}")

    current_run.log_info(f"Organisation units updated: {updates_count}")
    if errors_count > 0:
        current_run.log_info(
            f"{errors_count} errors occurred during OU update. Please check the latest execution report under {report_path}."
        )


def select_transform_to_json(data_values: pd.DataFrame) -> tuple[list[dict], list[pd.Series], list[dict]]:
    """Transform a DataFrame of data values into lists of valid, not valid, and to-delete datapoints.

    Args:
        data_values (pd.DataFrame): DataFrame containing data values to transform.

    Returns:
        tuple: A tuple containing three lists:
            - valid (list[dict]): List of valid datapoints as dictionaries.
            - not_valid (list[pd.Series]): List of rows that are not valid.
            - to_delete (list[dict]): List of datapoints to be deleted as dictionaries.
    """
    if data_values is None:
        return [], []
    valid = []
    not_valid = []
    to_delete = []
    for _, row in data_values.iterrows():
        dpoint = DataPoint(row)
        if dpoint.is_valid():
            valid.append(dpoint.to_json())
        elif dpoint.is_to_delete():
            to_delete.append(dpoint.to_delete_json())
        else:
            not_valid.append(row)  # row is the original data, not the json
    return valid, not_valid, to_delete


def push_orgunit(dhis2_client: DHIS2, orgunit: OrgUnitObj, strategy: str = "CREATE", dry_run: bool = True):
    if strategy == "CREATE":
        endpoint = "organisationUnits"
        payload = orgunit.to_json()

    if strategy == "UPDATE":
        endpoint = "metadata"
        payload = {"organisationUnits": [orgunit.to_json()]}

    r = dhis2_client.api.session.post(
        f"{dhis2_client.api.url}/{endpoint}",
        json=payload,
        params={"dryRun": dry_run, "importStrategy": f"{strategy}"},
    )

    return build_formatted_response(response=r, strategy=strategy, ou_id=orgunit.id)


def push_data_elements(
    dhis2_client: DHIS2,
    data_elements_list: list,
    strategy: str = "CREATE_AND_UPDATE",
    dry_run: bool = True,
    max_post: int = 1000,
):
    """dry_run: Set to true to get an import summary without actually importing data (DHIS2)."""
    # max_post instead of MAX_POST_DATA_VALUES
    summary = {
        "import_counts": {"imported": 0, "updated": 0, "ignored": 0, "deleted": 0},
        "import_options": {},
        "ERRORS": [],
    }

    total_datapoints = len(data_elements_list)
    count = 0

    for chunk in split_list(data_elements_list, max_post):
        count = count + 1
        try:
            r = dhis2_client.api.session.post(
                f"{dhis2_client.api.url}/dataValueSets",
                json={"dataValues": chunk},
                params={
                    "dryRun": dry_run,
                    "importStrategy": strategy,
                    "preheatCache": True,
                    "skipAudit": True,
                },  # speed!
            )
            r.raise_for_status()

            try:
                response_json = r.json()
                status = response_json.get("httpStatus")
                response = response_json.get("response")
            except json.JSONDecodeError as e:
                summary["ERRORS"].append(f"Response JSON decoding failed: {e}")  # period: {chunk_period}")
                response_json = None
                status = None
                response = None

            if status != "OK" and response:
                summary["ERRORS"].append(response)

            if response:
                for key in ["imported", "updated", "ignored", "deleted"]:
                    summary["import_counts"][key] += response.get("importCount", {}).get(key, 0)

        except requests.exceptions.RequestException as e:
            try:
                response = r.json().get("response")
            except (ValueError, AttributeError):
                response = None

            if response:
                for key in ["imported", "updated", "ignored", "deleted"]:
                    summary["import_counts"][key] += response["importCount"][key]

            error_response = get_response_value_errors(response, chunk=chunk)
            summary["ERRORS"].append({"error": e, "response": error_response})

        if (count * max_post) % 100000 == 0:
            current_run.log_info(
                f"{count * max_post} / {total_datapoints} data points pushed summary: {summary['import_counts']}"
            )

    return summary


def get_response_value_errors(response, chunk):
    """Collect relevant data for error logs"""
    if response is None:
        return None

    if chunk is None:
        return None

    try:
        out = {}
        for k in ["responseType", "status", "description", "importCount", "dataSetComplete"]:
            out[k] = response.get(k)
        if "conflicts" in response and response["conflicts"]:
            out["rejected_datapoints"] = []
            for i in response["rejectedIndexes"]:
                out["rejected_datapoints"].append(chunk[i])
            out["conflicts"] = {}
            for conflict in response["conflicts"]:
                out["conflicts"]["object"] = conflict.get("object")
                out["conflicts"]["objects"] = conflict.get("objects")
                out["conflicts"]["value"] = conflict.get("value")
                out["conflicts"]["errorCode"] = conflict.get("errorCode")
        return out
    except AttributeError:
        return None


def build_formatted_response(response: requests.Response, strategy: str, ou_id: str) -> dict:
    resp = {
        "action": strategy,
        "statusCode": response.status_code,
        "status": response.json().get("status"),
        "response": response.json().get("response"),
        "ou_id": ou_id,
    }
    return resp


def build_id_indexes(ou_source, ou_target, ou_matching_ids):
    # Set "id" as the index for faster lookup
    df1_lookup = {val: idx for idx, val in enumerate(ou_source["id"])}
    df2_lookup = {val: idx for idx, val in enumerate(ou_target["id"])}

    # Build the dictionary using prebuilt lookups
    index_dict = {
        match_id: {"source": df1_lookup[match_id], "target": df2_lookup[match_id]}
        for match_id in ou_matching_ids
        if match_id in df1_lookup and match_id in df2_lookup
    }
    return index_dict


#### CHECK THIS FUNCTION!! Category option default selection needs fixing
def apply_dataelement_mappings(datapoints_df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """All matching ids will be replaced.

    Is user responsability to provide the correct UIDS.
    """
    datapoints_df = datapoints_df.copy()
    uids_to_replace = set(datapoints_df["dx_uid"]).intersection(mappings.get("UIDS", {}))
    if len(uids_to_replace) > 0:
        current_run.log_info(f"{len(uids_to_replace)} OU UIDS to be replaced using mappings.")
        datapoints_df.loc[:, "dx_uid"] = datapoints_df["dx_uid"].replace(mappings.get("UIDS", {}))

    # Fields ou, coc and aoc will throw an error while pushing if wrong..
    orunits_to_replace = list(set(datapoints_df.org_unit).intersection(set(mappings.get("ORG_UNITS", {}).keys())))
    if len(orunits_to_replace) > 0:
        current_run.log_info(f"{len(orunits_to_replace)} Org units will be replaced using mappings.")
        datapoints_df.loc[:, "org_unit"] = datapoints_df["org_unit"].replace(mappings.get("ORG_UNITS", {}))

    # map category option combo default
    coc_default = mappings.get("CAT_OPTION_COMBOS").get("DEFAULT")
    if coc_default:
        current_run.log_info(f"Using {coc_default} default COC id.")
        datapoints_df.loc[:, "category_option_combo"] = datapoints_df["category_option_combo"].replace(
            {None: coc_default}
        )

    # map category option combo values
    coc_to_replace = set(datapoints_df["category_option_combo"]).intersection(
        mappings.get("CAT_OPTION_COMBOS").get("UIDS")
    )
    if len(coc_to_replace) > 0:
        current_run.log_info(f"{len(coc_to_replace)} data elements COC will be replaced using mappings.")
        datapoints_df.loc[:, "category_option_combo"] = datapoints_df["category_option_combo"].replace(
            mappings.get("CAT_OPTION_COMBO")
        )

    # map attribute option combo
    aoc_default = mappings["ATTR_OPTION_COMBO"].get("DEFAULT", None)
    if aoc_default:
        current_run.log_info(f"Using {aoc_default} default AOC id.")
        datapoints_df.loc[:, "attribute_option_combo"] = datapoints_df["attribute_option_combo"].replace(
            {None: aoc_default}
        )
    aoc_to_replace = set(datapoints_df["attribute_option_combo"]).intersection(
        mappings.get("ATTR_OPTION_COMBO", {}).keys()
    )
    if len(aoc_to_replace) > 0:
        current_run.log_info(f"{len(aoc_to_replace)} data elements AOC will be replaced using mappings.")
        datapoints_df.loc[:, "attribute_option_combo"] = datapoints_df["attribute_option_combo"].replace(
            mappings.get("ATTR_OPTION_COMBO", {})
        )

    return datapoints_df


def apply_rate_mappings(datapoints_df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """this is a specific code to map the rate

    All matching ids (keys) will be replaced.
    Is user responsability to provide the correct UIDS.
    """
    datapoints_df = datapoints_df.copy()
    for key, value in mappings.items():
        datapoints_df.loc[((datapoints_df.dx_uid == key) & (datapoints_df.rate_type == "REPORTING_RATE")), "dx_uid"] = (
            value["REPORTING_RATE"]
        )
        datapoints_df.loc[
            ((datapoints_df.dx_uid == key) & (datapoints_df.rate_type == "REPORTING_RATE_ON_TIME")), "dx_uid"
        ] = value["REPORTING_RATE_ON_TIME"]

    return datapoints_df


def apply_indicator_mappings(datapoints_df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """this is a specific code to map the indicator acm
    All matching ids will be replaced.
    Is user responsability to provide the correct UIDS.
    """
    datapoints_df = datapoints_df.copy()
    uids_acm_to_replace = set(datapoints_df["dx_uid"]).intersection(mappings.get("UIDS", {}).keys())
    if len(uids_acm_to_replace) > 0:
        current_run.log_info(f"{len(uids_acm_to_replace)} ACM indicator(s) to be replaced using mappings.")
        datapoints_df.loc[:, "dx_uid"] = datapoints_df["dx_uid"].replace(mappings.get("UIDS", {}))

    # map category option combo default
    coc_default = mappings["CAT_OPTION_COMBO"].get("DEFAULT", None)
    if coc_default and len(uids_acm_to_replace) > 0:
        current_run.log_info(f"Using {coc_default} default COC id mapping on ACM.")
        # print(f"Using {coc_default} default COC id mapping on ACM.")
        datapoints_df.loc[datapoints_df.dx_uid.isin(mappings.get("UIDS", {}).values()), "category_option_combo"] = (
            coc_default
        )

    return datapoints_df


# log ignored datapoints in the report
def log_ignored_or_na(report_path, datapoint_list, data_type="population", is_na=False):
    if len(datapoint_list) > 0:
        current_run.log_info(
            f"{len(datapoint_list)} datapoints will be  {'updated to NA' if is_na else 'ignored'}. Please check the report for details {report_path}"
        )
        logging.warning(f"{len(datapoint_list)} {data_type} datapoints to be ignored: ")
        for i, ignored in enumerate(datapoint_list, start=1):
            logging.warning(f"{i} DataElement {'NA' if is_na else ''} ignored: {ignored}")


if __name__ == "__main__":
    dhis2_pnlp_push()
