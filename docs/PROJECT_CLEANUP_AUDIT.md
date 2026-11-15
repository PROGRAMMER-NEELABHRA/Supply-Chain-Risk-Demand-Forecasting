# Project Cleanup Audit

**Audit basis:** `run_pipeline.zip` uploaded by the user.\
**Audit mode:** Read-only archive inspection. No project files were
changed or deleted.\
**Important limitation:** The ZIP does not include `.git`, so Git
history, staged changes, and live working-tree state could not be
inspected. File-use conclusions are based on archive contents, static
imports, visible path references, and project documentation;
dynamic/runtime use may require local validation.

## 1. Archive inventory

  Item                                       Count
  ---------------------------------------- -------
  ZIP entries including directories            480
  Files excluding directory entries            413
  Python source files                          120
  Python bytecode/cache files (`.pyc`)         130
  CSV files                                    126
  SQL files                                     11
  Markdown files                                 9
  PNG files                                      6
  Power BI files (`.pbix`)                       1
  Other files (TOML, TXT, CSS, DB, tags)         9

Top-level file distribution:

  Area                        Files
  ------------------------- -------
  `src/`                        132
  `outputs/`                    100
  `tests/`                       62
  `dashboard/`                   51
  `dataset/`                     19
  `data/`                        11
  `sql/`                         11
  `Supply_Chain_PowerBI/`         7
  `docs/`                         5

Counts include Python cache files where present.

## 2. Essential application and source components

  -----------------------------------------------------------------------------------------------------------------------------------------------
  Path                                                   Type           Approx. size /       Initial          Evidence / action
                                                                        contents             classification
  ------------------------------------------------------ -------------- -------------------- ---------------- -----------------------------------
  `dashboard/app.py`                                     Python         Streamlit entry      A --- Essential  Preserve; documented launch command
                                                                        point                                 is
                                                                                                              `streamlit run dashboard/app.py`.

  `dashboard/pages/`                                     Python         8 page modules       A --- Essential  Preserve all eight pages; Streamlit
                                                                                                              multipage files may be discovered
                                                                                                              dynamically.

  `dashboard/operational_data.py`                        Python         8.9 KB               A --- Essential  Preserve; operational pages/tests
                                                                                                              may depend on it.

  `dashboard/user_demand_forecasting.py`                 Python         38.5 KB              A --- Essential  Preserve; user CSV workflow module.

  `dashboard/pages/08_user_demand_forecasting.py`        Python         39.2 KB              A --- Essential  Preserve; user forecasting page.

  `src/`                                                 Python         Analysis, data,      A --- Essential  Preserve pending detailed
                                                                        forecasting,         / D --- Research per-module reference tracing.
                                                                        inventory, risk,
                                                                        scenarios,
                                                                        visualization

  `run_pipeline.py`                                      Python         1.6 KB               A --- Essential  Orchestrates four pipeline stages
                                                                                                              through `python -m` commands.

  `config.py`                                            Python         678 B                A --- Essential  Defines project and data/output
                                                                                                              paths; imported by multiple modules
                                                                                                              according to source search.

  `main.py`                                              Python         1.1 KB               A --- Utility    CLI dataset inspection entry point;
                                                                                                              preserve unless user explicitly
                                                                                                              retires it.

  `tests/`                                               Python         25 test modules plus A --- Essential  Preserve all tests.
                                                                        cache

  `data/raw/integrated_supply_chain.csv`                 CSV            58.1 MB              B --- Essential  Original source dataset; preserve
                                                                                             source           unchanged.

  `data/raw/integrated_supply_chain_until_aug2017.csv`   CSV            53.9 MB              B --- Essential  Configured DataCo source in
                                                                                             source           `config.py` and README; preserve
                                                                                                              unchanged.

  `data/processed/`                                      CSV            Monthly demand and   C --- Generated  Preserve until all consumers and
                                                                        shortlist            data             regeneration paths are verified.

  `data/quality_review/`                                 CSV/MD         7 quality-review     D --- Important  Preserve; contains data-quality
                                                                        files                research         evidence and labelled imputation.

  `dataset/processed/processed/dataco/`                  CSV/DB         19 files including   B/C ---          Preserve pending investigation of
                                                                        \~51 MB SQLite DB    Important        database and CSV consumers.
                                                                                             source/derived
                                                                                             data

  `sql/`                                                 SQL            11 scripts           D --- Important  Preserve; SQL may be used manually
                                                                                             research         and is not expected to have Python
                                                                                                              imports.

  `Supply_Chain_PowerBI/Documentation/dashboard.pbix`    Power BI       \~195 KB             A --- Essential  Preserve.
                                                                                             deliverable

  `Supply_Chain_PowerBI/Data/`                           CSV            6 source/export CSVs A/C --- Power BI Preserve; identical copies
                                                                                             inputs           elsewhere do not prove Power BI can
                                                                                                              use a different path.

  `docs/`                                                Markdown       5 documents          D --- Important  Preserve and reconcile claims with
                                                                                             documentation    current code.

  `requirements.txt`, `pyproject.toml`                   Config         Dependencies/build   A --- Essential  Preserve; optimize only after
                                                                        config               config           complete optional-workflow review.

  `.streamlit/config.toml`                               TOML           Streamlit config     A --- Essential  Preserve.
                                                                                             config
  -----------------------------------------------------------------------------------------------------------------------------------------------

## 3. Safe generated cache candidates

  --------------------------------------------------------------------------
  Candidate                          Count Classification   Recommended
                                                            action
  ------------------ --------------------- ---------------- ----------------
  `__pycache__/`                  Multiple F ---            Safe to remove
  directories                              Regenerable      from the working
                                           cache            copy after
                                                            confirming no
                                                            process is using
                                                            them.

  `*.pyc` files                        130 F ---            Safe to remove;
                                           Regenerable      Python
                                           cache            regenerates
                                                            them.

  `.pytest_cache/`     5 files / directory F ---            Safe to remove;
                                      tree Regenerable      pytest
                                           cache            regenerates it.
  --------------------------------------------------------------------------

These cache files should not be committed to source control.
`.gitignore` already excludes `__pycache__/`, `*.py[cod]`, and
`.pytest_cache/`.

## 4. Empty Python modules --- do not delete yet

Sixteen zero-byte Python files were found:

-   `src/analysis/demand_analysis.py`
-   `src/analysis/__init__.py`
-   `src/data/integration.py`
-   `src/data/__init__.py`
-   `src/forecasting/__init__.py`
-   `src/inventory/__init__.py`
-   `src/preprocessing/cleaning.py`
-   `src/preprocessing/__init__.py`
-   `src/risk/inventory_risk.py`
-   `src/risk/logistics_risk.py`
-   `src/risk/supplier_risk.py`
-   `src/risk/__init__.py`
-   `src/scenarios/__init__.py`
-   `src/visualization/charts.py`
-   `src/visualization/__init__.py`
-   `src/__init__.py`

**Recommendation:** Keep all for now. Empty `__init__.py` files can
define regular Python packages; empty feature modules may be
placeholders or planned extension points. Confirm references and package
behavior before considering removal.

## 5. Duplicate hash findings

SHA-256 comparison found 36 groups of byte-identical files. Important
examples:

  -------------------------------------------------------------------------------------------
  Identical artifact                                       Copies found Recommendation
  ---------------------------------------- ---------------------------- ---------------------
  `forecast_warning_summary.csv`                                      3 Keep the current
                                                                        output and Power BI
                                                                        copy; retain snapshot
                                                                        until archive policy
                                                                        is approved.

  `forecast_method_summary.csv`                                       3 Same as above.

  `forecast_data_warnings.csv`                                        3 Same as above.

  `raw_processed_monthly_comparison.csv`                              3 Keep current output
                                                                        and Power BI copy;
                                                                        preserve snapshot.

  `random_forest_model_comparison.csv`                                3 Keep current output
                                                                        and Power BI copy;
                                                                        preserve snapshot.

  `future_demand_forecasts.csv`                                       2 Keep both
                                                                        current/Power BI
                                                                        consumer paths until
                                                                        PBIX source
                                                                        connection is
                                                                        checked.

  `skipped_products.csv` and other           2 each in identified cases Snapshot copies may
  historical analysis files                                             be useful as
                                                                        historical evidence;
                                                                        do not delete yet.
  -------------------------------------------------------------------------------------------

The 36 groups include many empty files with the same zero-byte hash.
Those are not necessarily duplicate modules: different module paths may
have different import/package meaning.

**Important:** Byte-identical content does not mean interchangeable
paths. Power BI, scripts, documentation, or manual workflows may depend
on the exact path.

## 6. Historical snapshot

`outputs/handoff/dataco_pre_aug2017_refresh_20261114_180124/` contains a
preserved dataset/quality-review/forecast-output snapshot, including
experiment predictions and summaries.

**Classification:** D --- Important historical research / G --- Archive
policy requires approval.

**Recommendation:** Do not delete. First determine whether this is the
only preserved record of a previous DataCo run, then document its source
cutoff and purpose. If approved later, move it to a clearly named
archive only after verifying references and recording its provenance.

## 7. Dependency review

Declared runtime/test packages in `requirements.txt`:

-   `pandas`
-   `numpy`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `openpyxl`
-   `pytest`
-   `streamlit>=1.51.0`
-   `plotly>=5.22.0`

Static Python import scan found usage for pandas, NumPy, Streamlit,
Plotly, scikit-learn, matplotlib, and pytest. No direct `seaborn` or
`openpyxl` import/API usage was found in the inspected Python source and
text files.

**Recommendation:** Mark `seaborn` and `openpyxl` as dependency review
candidates only. Do not remove yet: Excel support, optional scripts,
indirect package integrations, and environment checks must be examined.
`pytest` is a development/test dependency and could be separated into an
optional development group, but only if the installation instructions
are updated and validated. `pyproject.toml` currently defines project
metadata/build settings but does not declare project dependencies.

## 8. Documentation discrepancies

The README says the dashboard navigation includes seven pages, but the
archive contains eight page modules, including
`08_user_demand_forecasting.py`. The README should be corrected to list
the eighth page.

README states the configured DataCo source is
`data/raw/integrated_supply_chain_until_aug2017.csv`, while existing
outputs may belong to an earlier source. It explicitly warns that those
outputs have not yet been rebuilt from the selected source. Preserve
this warning unless current pipeline outputs are regenerated and
verified.

README describes Inventory Management, Supply Chain Risk, and Scenario
Simulator as prototypes with limitations. Keep these caveats unless
implementation evidence supports updating them.

## 9. Directory structure assessment

The existing top-level structure is already reasonably separated:

-   `dashboard/`
-   `src/`
-   `data/raw/`, `data/processed/`, `data/quality_review/`
-   `outputs/forecasts/`, `outputs/handoff/`
-   `tests/`
-   `docs/`
-   `sql/`
-   `Supply_Chain_PowerBI/`
-   `dataset/processed/processed/dataco/`

Do not perform a broad folder migration yet. The nested
`dataset/processed/processed/dataco/` path is awkward, but it may be
referenced by the database/CSV workflows. Review its consumers before
renaming it.

## 10. Recommended next actions

1.  Remove regenerable Python and pytest caches from the local working
    copy.
2.  Trace every path used by `dashboard/`, `src/`, tests, SQL docs, and
    Power BI source configuration.
3.  Decide whether `outputs/handoff/` is a required historical snapshot
    or can be compressed/archived.
4.  Reconcile the README's seven-page claim with the eight page modules.
5.  Review `seaborn` and `openpyxl` dependency necessity.
6.  Inspect the DataCo SQLite database and nested dataset paths before
    reorganizing.
7.  Run the test suite in the actual project environment after any
    cleanup.

## 11. Validation status

No tests were run as part of this archive-only audit. The ZIP does not
include the `.git` directory or a guaranteed matching local virtual
environment. The previously reported 327-passed result has not been
independently verified in this audit.

No source datasets, application modules, tests, outputs, Power BI files,
or historical research artifacts were modified or deleted.
