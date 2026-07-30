# dhis2_data_sync_push

*(English version below)*

## Description

Pousse (push) les extractions d'éléments de données produites par le pipeline `dhis2_data_sync_extract` vers l'instance DHIS2 NMDR (cible). Pour chaque extraction configurée, les valeurs de données brutes sont réassignées vers les UID/COC/AOC cibles selon un fichier de configuration JSON de correspondance (mapping), puis envoyées par lots à l'API DHIS2 cible.

## Paramètres

| Paramètre | Code | Type | Obligatoire | Valeur par défaut | Description |
|---|---|---|---|---|---|
| Date de début | `start_date` | str (`YYYYMM`) | Non | `201701` | Début de la fenêtre de push. |
| Date de fin | `end_date` | str (`YYYYMM`) | Non | `202112` | Fin de la fenêtre de push. |

## Description du processus (workflow)

1. Charger `configuration/push_config.json` et se connecter à l'instance DHIS2 cible définie dans `SETTINGS.TARGET_DHIS2_CONNECTION`.
2. Lire les paramètres d'envoi (`IMPORT_STRATEGY`, `DRY_RUN`, `MAX_POST`) et initialiser le client de push (`DHIS2Pusher`).
3. Pour chaque entrée sous `DATA_ELEMENTS.EXTRACTS` dans la configuration :
   - Résoudre la liste des périodes à pousser (fenêtre `start_date`/`end_date`, réduite aux années uniques si `FREQUENCY` vaut `YEARLY`).
   - Pour chaque période, charger le fichier Parquet correspondant produit par `dhis2_data_sync_extract` (`data/extracts/<EXTRACT_UID>/data_<EXTRACT_UID>_<period>.parquet`) ; si le fichier est absent, la période est ignorée (avertissement journalisé). Toute valeur enregistrée comme la chaîne littérale `"None"` (artefact connu de l'extraction) est convertie en valeur nulle réelle (avertissement journalisé).
   - Appliquer les correspondances (`MAPPINGS`) : pour chaque UID source configuré, filtrer les lignes correspondantes, ré-assigner la combinaison d'options de catégorie (`CATEGORY_OPTION_COMBO`) et d'attribut (`ATTRIBUTE_OPTION_COMBO`) vers leurs valeurs cibles (ou filtrer sur les clés fournies si ce ne sont pas des correspondances `"ALL"`), puis agréger les valeurs et renommer l'UID source vers l'UID cible. Si cette étape échoue, l'erreur est considérée comme un problème de configuration : elle est journalisée puis **le pipeline s'arrête immédiatement** (un mapping invalide échouerait de la même façon pour toutes les périodes restantes de cette extraction).
   - Appliquer le typage des valeurs selon les métadonnées DHIS2 (`valueType` de l'élément de données cible) : `NUMBER` est converti en nombre décimal, `INTEGER`/`INTEGER_ZERO_OR_POSITIVE` en entier (décimales tronquées, ex. `"10.7"` → `10`) ; toute valeur non numérique, ou négative pour `INTEGER_ZERO_OR_POSITIVE`, est remplacée par une valeur nulle (avertissement journalisé).
   - Envoyer les données mappées à l'instance cible. Si l'envoi échoue pour une période, l'erreur est journalisée et **le pipeline continue** avec la période suivante (comportement best-effort, contrairement aux erreurs de mapping).
   - Après chaque période (succès, échec d'envoi, ou fichier manquant), les journaux sont copiés dans `logs/` sous le dossier du pipeline.

Chaque entrée d'extraction dans `push_config.json` requiert :
- `EXTRACT_UID` — doit correspondre à l'UID d'extraction produit par `dhis2_data_sync_extract`, utilisé pour localiser les fichiers Parquet sources.
- `FREQUENCY` — `MONTHLY` ou `YEARLY`.
- `MAPPINGS` — dictionnaire indexé par UID source, où chaque entrée fournit :
  - `UID` — UID de l'élément de données cible.
  - `CATEGORY_OPTION_COMBO` — correspondance COC source → cible, ou `{"ALL": "<uid>"}` pour forcer une seule COC cible pour toutes les lignes.
  - `ATTRIBUTE_OPTION_COMBO` — correspondance AOC source → cible.

## Description des sorties

- **Aucun fichier de données n'est produit** : ce pipeline consomme les extractions Parquet de `dhis2_data_sync_extract` et envoie les données à l'instance DHIS2 cible ; il ne persiste pas les résultats de l'envoi sous forme de fichier.
- **Journaux d'exécution** : écrits sous `/home/jovyan/tmp/logs/push_data_<horodatage>.log`, puis copiés après chaque période dans `<pipeline_path>/logs/` ; la progression, les avertissements (fichier manquant, UID d'extraction manquant, valeurs invalides) et les erreurs sont également signalés via la console d'exécution OpenHEXA (`current_run.log_*`).

---

# dhis2_data_sync_push (English)

## Description

Pushes the data element extracts produced by the `dhis2_data_sync_extract` pipeline to the NMDR (target) DHIS2 instance. For each configured extract, raw data values are remapped to target UIDs/COCs/AOCs according to a JSON mapping configuration, then sent in batches to the target DHIS2 API.

## Parameters

| Parameter | Code | Type | Required | Default | Description |
|---|---|---|---|---|---|
| Start date | `start_date` | str (`YYYYMM`) | No | `201701` | Start of the push window. |
| End date | `end_date` | str (`YYYYMM`) | No | `202112` | End of the push window. |

## Workflow process description

1. Load `configuration/push_config.json` and connect to the target DHIS2 instance defined in `SETTINGS.TARGET_DHIS2_CONNECTION`.
2. Read the push settings (`IMPORT_STRATEGY`, `DRY_RUN`, `MAX_POST`) and initialize the push client (`DHIS2Pusher`).
3. For each entry under `DATA_ELEMENTS.EXTRACTS` in the config:
   - Resolve the list of periods to push (`start_date`/`end_date` window, collapsed to unique years if `FREQUENCY` is `YEARLY`).
   - For each period, load the matching Parquet file produced by `dhis2_data_sync_extract` (`data/extracts/<EXTRACT_UID>/data_<EXTRACT_UID>_<period>.parquet`); if the file is missing, the period is skipped (a warning is logged). Any value stored as the literal string `"None"` (a known extraction artifact) is converted to a real null (a warning is logged).
   - Apply the configured mappings (`MAPPINGS`): for each configured source UID, filter the matching rows, remap the category option combo (`CATEGORY_OPTION_COMBO`) and attribute option combo (`ATTRIBUTE_OPTION_COMBO`) to their target values (or filter down to the provided keys unless it's an `"ALL"` mapping), then aggregate values and rename the source UID to the target UID. If this step fails, it's treated as a configuration problem: the error is logged and **the pipeline stops immediately** (an invalid mapping would fail identically for every remaining period of that extract).
   - Apply value typing based on DHIS2 metadata (the target data element's `valueType`): `NUMBER` is cast to a decimal number, `INTEGER`/`INTEGER_ZERO_OR_POSITIVE` to an integer (decimals truncated, e.g. `"10.7"` → `10`); any non-numeric value, or negative value for `INTEGER_ZERO_OR_POSITIVE`, is replaced with a null (a warning is logged).
   - Push the mapped data to the target instance. If the push fails for a period, the error is logged and **the pipeline keeps going** with the next period (best-effort, unlike mapping errors).
   - After each period (success, push failure, or missing file), logs are copied into `logs/` under the pipeline folder.

Each extract entry in `push_config.json` requires:
- `EXTRACT_UID` — must match the extract UID produced by `dhis2_data_sync_extract`, used to locate the source Parquet files.
- `FREQUENCY` — `MONTHLY` or `YEARLY`.
- `MAPPINGS` — dictionary keyed by source UID, where each entry provides:
  - `UID` — target data element UID.
  - `CATEGORY_OPTION_COMBO` — source → target COC mapping, or `{"ALL": "<uid>"}` to force a single target COC for all rows.
  - `ATTRIBUTE_OPTION_COMBO` — source → target AOC mapping.

## Outputs description

- **No data files are produced**: this pipeline consumes the Parquet extracts from `dhis2_data_sync_extract` and sends data to the target DHIS2 instance; it does not persist push results as a file.
- **Run logs**: written to `/home/jovyan/tmp/logs/push_data_<timestamp>.log`, then copied after each period into `<pipeline_path>/logs/`; progress, warnings (missing file, missing extract UID, invalid values) and errors are also reported via the OpenHEXA run console (`current_run.log_*`).
