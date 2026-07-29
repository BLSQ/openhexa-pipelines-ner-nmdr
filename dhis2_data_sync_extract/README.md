# dhis2_data_sync_extract

*(English version below)*

## Description

Extrait les extractions d'éléments de données configurées depuis l'instance DHIS2 SNIS (source) et les enregistre sous forme de fichiers Parquet. Les unités d'organisation à extraire sont résolues à partir de **groupes d'unités d'organisation** DHIS2 (plutôt qu'un niveau fixe de la pyramide), le tout piloté par un fichier de configuration JSON. L'extraction de la pyramide est actuellement désactivée pour ce pipeline car aucune étape d'alignement/envoi des unités d'organisation n'est nécessaire en aval.

## Paramètres

| Paramètre | Code | Type | Obligatoire | Valeur par défaut | Description |
|---|---|---|---|---|---|
| Date de début | `start_date` | str (`YYYYMM`) | Non | `None` | Début de la fenêtre d'extraction. Si non défini, correspond par défaut à la date actuelle moins `NUMBER_MONTHS_WINDOW` (dans la configuration). |
| Date de fin | `end_date` | str (`YYYYMM`) | Non | `None` | Fin de la fenêtre d'extraction. Si non défini, correspond par défaut à la date actuelle moins 1 mois. |
| Mode de téléchargement | `download_mode` | str, choix : `DOWNLOAD_REPLACE`, `DOWNLOAD_NEW` | Non | `DOWNLOAD_REPLACE` | `DOWNLOAD_NEW` ne télécharge que les périodes non encore extraites ; `DOWNLOAD_REPLACE` retélécharge et remplace les extractions existantes. |

## Description du processus (workflow)

1. Charger `configuration/extract_config.json` et se connecter à l'instance DHIS2 source définie dans `SETTINGS.SOURCE_DHIS2_CONNECTION`.
2. Résoudre et valider la fenêtre de période d'extraction (paramètres `start_date`/`end_date`, avec repli sur `SETTINGS.STARTDATE`/`ENDDATE`/`NUMBER_MONTHS_WINDOW` dans la configuration), produisant une liste de périodes au format `YYYYMM` à extraire.
3. Récupérer tous les groupes d'unités d'organisation de l'instance DHIS2 source.
4. Pour chaque entrée sous `DATA_ELEMENTS.EXTRACTS` dans la configuration :
   - Résoudre les unités d'organisation cibles en prenant l'**union des unités d'organisation** de tous les groupes listés dans `ORG_UNIT_GROUPS` (sans doublons).
   - Si `FREQUENCY` vaut `YEARLY`, réduire la liste des périodes aux années uniques ; sinon, utiliser la liste complète des périodes mensuelles.
   - Pour chaque période, télécharger les `UIDS` d'éléments de données configurés pour les unités d'organisation résolues et enregistrer le résultat sous forme de fichier Parquet.
   - Si le téléchargement d'une période échoue, cette extraction est ignorée pour les périodes restantes et le pipeline passe à l'extraction suivante configurée (une erreur est journalisée, le pipeline ne s'arrête pas).

Chaque entrée d'extraction dans `extract_config.json` requiert :
- `EXTRACT_UID` — identifiant utilisé pour nommer les fichiers/dossiers de sortie.
- `UIDS` — UIDs des éléments de données DHIS2 à extraire.
- `ORG_UNIT_GROUPS` — liste des UIDs de groupes d'unités d'organisation DHIS2 dont les unités doivent être incluses.
- `FREQUENCY` — `MONTHLY` ou `YEARLY`.

## Description des sorties

- **Extractions d'éléments de données** : `data/extracts/<EXTRACT_UID>/data_<EXTRACT_UID>_<period>.parquet` — un fichier Parquet par extraction configurée et par période, contenant les valeurs de données DHIS2 brutes pour les éléments de données et les unités d'organisation résolues de cette extraction.
- **Extraction de la pyramide** (actuellement désactivée) : `data/pyramid/snis_pyramid.parquet` — instantané de la hiérarchie des unités d'organisation ; l'appel d'extraction est commenté dans `pipeline.py` car ce projet ne nécessite pas d'alignement des unités d'organisation.
- **Journaux d'exécution** : la progression, les avertissements (par ex. `EXTRACT_UID`/`ORG_UNIT_GROUPS` manquants, groupes d'unités d'organisation non résolus) et les erreurs sont signalés via la console d'exécution OpenHEXA (`current_run.log_*`).

---

# dhis2_data_sync_extract (English)

## Description

Extracts configured data element extracts from the SNIS DHIS2 instance (source) and saves them as Parquet files. Org units to extract are resolved from DHIS2 **organisation unit groups** (rather than a fixed pyramid level), driven entirely by a JSON configuration file. Pyramid extraction is currently disabled for this pipeline since no org unit alignment/push step is needed downstream.

## Parameters

| Parameter | Code | Type | Required | Default | Description |
|---|---|---|---|---|---|
| Start date | `start_date` | str (`YYYYMM`) | No | `None` | Start of the extraction window. If not set, defaults to the current date minus `NUMBER_MONTHS_WINDOW` (from config). |
| End date | `end_date` | str (`YYYYMM`) | No | `None` | End of the extraction window. If not set, defaults to the current date minus 1 month. |
| Download mode | `download_mode` | str, choices: `DOWNLOAD_REPLACE`, `DOWNLOAD_NEW` | No | `DOWNLOAD_REPLACE` | `DOWNLOAD_NEW` only downloads periods not already extracted; `DOWNLOAD_REPLACE` re-downloads and overwrites existing extracts. |

## Workflow process description

1. Load `configuration/extract_config.json` and connect to the source DHIS2 instance defined in `SETTINGS.SOURCE_DHIS2_CONNECTION`.
2. Resolve and validate the extraction period window (`start_date`/`end_date` parameters, falling back to `SETTINGS.STARTDATE`/`ENDDATE`/`NUMBER_MONTHS_WINDOW` in the config), producing a list of `YYYYMM` periods to extract.
3. Retrieve all organisation unit groups from the source DHIS2 instance.
4. For each entry under `DATA_ELEMENTS.EXTRACTS` in the config:
   - Resolve the target org units by taking the **union of org units** across all group IDs listed in `ORG_UNIT_GROUPS` (deduplicated).
   - If `FREQUENCY` is `YEARLY`, collapse the period list down to unique years; otherwise use the full monthly period list.
   - For each period, download the configured data element `UIDS` for the resolved org units and save the result as a Parquet file.
   - If a period download fails, that extract is skipped for the remaining periods and the pipeline moves on to the next configured extract (an error is logged, the pipeline does not stop).

Each extract entry in `extract_config.json` requires:
- `EXTRACT_UID` — identifier used to name output files/folders.
- `UIDS` — DHIS2 data element UIDs to extract.
- `ORG_UNIT_GROUPS` — list of DHIS2 org unit group UIDs whose org units should be included.
- `FREQUENCY` — `MONTHLY` or `YEARLY`.

## Outputs description

- **Data element extracts**: `data/extracts/<EXTRACT_UID>/data_<EXTRACT_UID>_<period>.parquet` — one Parquet file per configured extract per period, containing raw DHIS2 data values for that extract's data elements and resolved org units.
- **Pyramid extract** (currently disabled): `data/pyramid/snis_pyramid.parquet` — org unit hierarchy snapshot; the extraction call is commented out in `pipeline.py` since this project does not require org unit alignment.
- **Run logs**: progress, warnings (e.g. missing `EXTRACT_UID`/`ORG_UNIT_GROUPS`, unresolved org unit groups) and errors are reported via the OpenHEXA run console (`current_run.log_*`).
