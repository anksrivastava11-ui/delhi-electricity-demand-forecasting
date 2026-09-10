# Delhi Electricity Demand Forecasting

One-year-ahead, daily-resolution forecasting of Delhi's unrestricted electricity
demand — the demand that would exist before supply-side load shedding, which is
the quantity used for procurement and capacity planning.

Nine forecasters, from a seasonal-naive baseline to gradient-boosted trees and
an LSTM, are trained on historical daily load, weather, calendar and annual
socio-economic indicators, then scored on a completely held-out final year.

The organising rule of the whole project: **every input a model receives must be
knowable on the day the forecast is issued.** A planner contracting capacity for
next year does not know next year's demand, and does not know next year's
weather either. Anything the pipeline cannot know at that moment is
reconstructed, projected or replaced — never taken from the future.

---

## Contents

1. [Where the data comes from](#1-where-the-data-comes-from)
2. [What the input Excel files must contain](#2-what-the-input-excel-files-must-contain)
3. [Install](#3-install)
4. [Running the pipeline](#4-running-the-pipeline)
5. [What the code does, step by step](#5-what-the-code-does-step-by-step)
6. [The models](#6-the-models)
7. [Scoring](#7-scoring)
8. [Every file the run produces](#8-every-file-the-run-produces)
9. [The charts and how to read them](#9-the-charts-and-how-to-read-them)
10. [Manuscript figures](#10-manuscript-figures)
11. [Tests, layout, licence](#11-tests-layout-licence)

---

## 1. Where the data comes from

Two independent sources are combined. Neither is distributed with this
repository — you supply both as Excel workbooks in `data/raw/`.

### Daily electricity load and weather

Compiled from the historical operational publications and records of **Delhi
Transco Limited (DTL)**, the state transmission utility, which is associated
with the **State Load Dispatch Centre (SLDC)** that monitors the Delhi power
system through an integrated SCADA system. The relevant material sits in DTL's
document archives, including the Operation Coordination Committee (OCC) meeting
archive, which records the operational data utilities submit to the SLDC.

A note on sourcing: DTL's *annual reports* publish aggregate annual statistics,
not daily tables. They are useful only as a coarse plausibility check on the
magnitude of daily values. The daily observations themselves come from the
operational records.

For each year, the daily observations were ordered chronologically, assigned a
standard calendar date, and combined into one annual dataset — 366 rows in a
leap year, 365 otherwise. **No interpolation, smoothing or averaging was applied
at compilation time**, so fluctuations driven by temperature, weekday/weekend
patterns, holidays and exceptional events survive as recorded. Compilation-time
validation covered chronological continuity (no duplicate or omitted dates,
correct leap-year handling), consistent MW units, and checks for blank,
non-numeric, duplicate or malformed values.

Daily maximum and minimum temperature and the season flags were compiled
alongside the load records, using the official Government of Delhi date-range
definitions for each season, for the region served by the India Meteorological
Department's Regional Meteorological Centre, New Delhi.

### Annual socio-economic indicators

Drawn from the **Reserve Bank of India's *Handbook of Statistics on Indian
States***, a multi-sector state-level statistical compilation. The workbook is
not one pre-existing RBI table — it is a researcher-constructed dataset built by
locating Delhi's observations across several separate Handbook tables and
merging them on the time period.

Per indicator, the procedure was: identify the RBI table, select Delhi from the
list of states and union territories, extract Delhi's observations, and assign
each to its reference year. Before merging, geography was fixed to Delhi only,
the reporting period was normalised to a fiscal year, and original units were
preserved as reported (percentages, persons, rupees, area, establishment counts,
power capacity). Missing values were kept missing rather than estimated, since
the source series have different collection frequencies.

The Handbook is a compilation platform rather than the originator of every
statistic — the underlying indicators come from various sector-specific
government agencies. Cite the RBI Handbook as the direct source, noting that
distinction where precision matters. **NITI Aayog** maintains a parallel,
overlapping state-statistics ecosystem (its 2017 *Handbook of State Statistics*,
and the NITI-NCAER States Economic Forum), and has itself cited the RBI Handbook
as a database reference; treat it as corroborating context for India's
state-statistics landscape, not as a source of these particular numbers.

---

## 2. What the input Excel files must contain

### `data/raw/<year>*.xlsx` — one workbook per year

**Filename.** Must contain the four-digit year somewhere; anything else is free
(`2016.xlsx`, `2016 Manav.xlsx`, `Delhi load 2016 final.xlsx` all work). Files
beginning `~$` — Excel lock files — are ignored.

**Sheets.** One sheet per month, named so that the **first three letters**
match `JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC` (case-insensitive, so
`January`, `jan`, `JAN 2016` are all fine). Sheets that don't match, or that
lack the day column, are skipped silently — a workbook where *no* sheet
qualifies raises an error.

**Columns**, per month sheet:

| Column | Required | Meaning |
|---|---|---|
| `Date` | yes | **Day of the month only** (1–31), not a full date. The year comes from the filename and the month from the sheet name; the three are combined into a real calendar date. |
| `Unrestricted demand` | yes | The target, in MW. The name is configurable via `data.target`. |
| `Max_Temperature` | recommended | Daily maximum temperature. |
| `Min_Temperature` | recommended | Daily minimum temperature. |
| `Summer`, `Winter`, `Monsoon`, `Pre Monsoon` | recommended | One-hot season flags, exactly one set to 1 per row. |

Numeric fields are parsed leniently — the first number in a cell is extracted by
regular expression, so `"4512 MW"` or `" 4,512"` still reads as a number.

If two workbooks exist for the same year, the pipeline does **not** pick
alphabetically. It scores each candidate on the number of usable month sheets,
then on how many required columns they carry, then on total column count, and
keeps the richest. This exists because alphabetical selection silently chose an
abbreviated workbook for one year and discarded its seasonal fields. The choice
is recorded in `selected_workbooks.csv` so it is auditable.

> **A naming caveat worth knowing.** The column `Pre Monsoon` covers
> October–November, which is really the *post*-monsoon season. The name is wrong
> but it is what the raw workbooks use, so the loader keeps it. Rename it in
> your data and the loader will break. Write "Post Monsoon" in any prose or
> figure labels you produce.

### `data/raw/Delhi.xlsx` — annual socio-economic indicators

**Layout: indicators as rows, years as columns.** The loader transposes it. The
first column holds the indicator names; the header row holds fiscal years in
`YYYY-YY` form (`2004-05`, `2005-06`, …).

Fiscal years are converted to their **ending calendar year** — `2004-05` becomes
2005 — and then merged onto every daily row sharing that calendar year.

Nineteen indicators are recognised by name (see `SOCIAL_FEATURES` in
`sensitivity.py`) across six categories:

| Category | Indicators |
|---|---|
| Social / demographic | Population in Rural Area, Population in Urban Area, Density of Population, Life Expactancy, Unemployement rate (Adjusted) (Rural), Unemployement rate (Adjusted) (Urban), Poverty Rate |
| Economic output | Per Capita Net State Domestic Product (Current Prices), Gross State Domestic Product (Current Prices) |
| Agriculture / land use | Net Sown Area |
| Inflation | Average Inflation (CPI) - General |
| Industrial activity | State-wise Number of Factories, State-wise Total Persons Engaged, State-wise Medium & Small Scale Industries - Total Number of Units |
| Power infrastructure | Per Capita Availability of Power, Installed Capacity of Power, Electricity Transmission & Distribution Losses, Power Requirement |

The misspellings `Life Expactancy` and `Unemployement rate` are in the source
workbook and are matched literally. Correct them and those columns stop being
recognised as social features.

The source workbook carries **two rows both labelled** `Average Inflation (CPI) -
General`. Both are kept; the second is renamed `Average Inflation (CPI) -
General__2` so that scikit-learn receives unique column names. That is why the
nineteen indicators listed above include a duplicate.

---

## 3. Install

Python 3.10 or newer. The base install carries only the statistical stack
(numpy, pandas, scikit-learn, statsmodels, matplotlib, scipy, PyYAML, openpyxl).
The gradient-boosting, Prophet, LSTM and SHAP dependencies are optional extras —
but the shipped `config/default.yaml` runs all nine models, so install
everything unless you trim the `models` list:

```powershell
pip install -e .[boosting,prophet,deep-learning,shap,dev]
```

| Extra | Pulls in | Needed for |
|---|---|---|
| `boosting` | xgboost, lightgbm | the `xgboost` and `lightgbm` models |
| `prophet` | prophet | the `prophet` model |
| `deep-learning` | torch | the `lstm` model, its sensitivity and its SHAP |
| `shap` | shap | the SHAP attribution stage and its figures |
| `dev` | pytest | running the test suite |

Install a parquet engine (`pyarrow` or `fastparquet`) if you want the processed
dataset stored as parquet; without one it silently falls back to CSV.

---

## 4. Running the pipeline

```powershell
electricity-forecast run --config config/default.yaml
```

| Command | Does |
|---|---|
| `prepare` | Builds the dataset from the raw workbooks, writes the processed file, the audit artifacts and the EDA charts. |
| `train` | Fits and scores the models against an already-prepared dataset. |
| `run` | Both, in order. |

`--config` defaults to `config/default.yaml`. Every stage prints a `[pipeline]`
progress line, so a long run is followable.

### Configuration

```yaml
paths:
  raw_dir: data/raw                                   # where the workbooks live
  processed_file: data/processed/daily_demand.parquet # prepare writes here
  artifacts_dir: artifacts                            # everything else lands here
data:
  target: Unrestricted demand
  date_column: Date
  temperature_columns: [Max_Temperature, Min_Temperature]
  social_file: data/raw/Delhi.xlsx                    # omit to skip social features
  years: [2014, ..., 2023]                            # one workbook required per year
features:
  lags: [1, 2, 3, 7, 14, 28, 365]
  rolling_windows: [7, 14, 28]
models: [seasonal_naive, holt_winters, arima, sarimax, prophet,
         random_forest, xgboost, lightgbm, lstm]
seed: 42
```

Nothing about the horizon is configurable: the split is always "hold out the
final year".

---

## 5. What the code does, step by step

### `prepare`

**5.1 Pick one workbook per year** — `_find_year_file`, `_workbook_score`.
Globs `*<year>*.xls*`, skips Excel lock files, and scores each candidate as
described above. `selected_workbook_manifest` writes the decision to disk.

**5.2 Read the month sheets** — `read_year_workbook`. Every sheet is loaded,
those whose first three letters name a month are kept, and a genuine `Date` is
built from `{year from filename, month from sheet name, day from the Date
column}`. Days that will not parse become `NaT`. All months are concatenated.

**5.3 Clean the daily series** — `clean_daily_data`. In order:

1. Coerce `Date` to datetime; pull the first number out of the target and
   temperature columns by regex.
2. Drop rows with no date or no target.
3. Collapse duplicate dates by averaging.
4. **Reindex onto a continuous daily calendar** from first to last date, so a
   day the source never recorded becomes an explicit missing row rather than an
   invisible gap.
5. Interpolate the target and temperatures across those gaps in both directions.
6. Fill any season flags left blank by step 4 — `fill_season_flags_by_calendar_month`
   assigns them from the row's calendar month (Winter Dec–Feb, Summer Mar–Jun,
   Monsoon Jul–Sep, Pre Monsoon Oct–Nov), which is the mapping every populated
   row in the dataset already follows. Without this, rows with `NaN` flags are
   dropped by the models' `dropna` and silently vanish from training.
7. Raise if any target value is still missing.

Step 4 is the one that creates gap-days. They are an artefact of reindexing, not
of the original compilation — typically month-end dates the source month sheet
had no row for.

**5.4 Merge the socio-economic indicators** — `merge_notebook_delhi_social_features`.
Transposes `Delhi.xlsx`, de-duplicates the repeated inflation label, converts
fiscal years to ending calendar years, then left-joins onto every daily row by
year with `validate="many_to_one"`. If any social column is still null after the
join, it raises and names the offending years rather than training on holes.

**5.5 Project indicators past the end of the source data** —
`extend_social_features_by_trend`, `forecast_social_variable_damped`. The
workbook's coverage ends before the modelled years do, and the uncovered years
typically include the entire test horizon. Rather than repeat the last known
year forward — which would leave the whole forecast period staring at frozen
stale values — each indicator is projected independently with **Holt's
damped-trend exponential smoothing**, the damping parameter fitted from the data
rather than imposed.

A damped trend mathematically tapers over a multi-step horizon, which is what
makes it appropriate here: an undamped linear or exponential fit on a handful of
annual points produces implausible swings on bounded quantities (one rate
variable's raw extrapolation exceeded 100%). Each column is forecast **once**, as
a single multi-step horizon, so the damping tapers across consecutive years as
intended. A series with fewer than four points, or no variation at all, falls
back to holding its last value. The method chosen per indicator per year is
logged to `social_feature_forecast_methods_used.csv`.

**5.6 Write the processed dataset and audit artifacts** — parquet, or CSV if no
parquet engine is installed. Alongside it: the workbook manifest, a per-year
per-column completeness report, and a separate five-year annual extension
(`build_v3_social_extension`) that compares linear against exponential fits by
in-sample RMSE. That last one is an exported reference table; it is not what
feeds the models.

**5.7 Generate the EDA charts** — `generate_eda_plots`, into `artifacts/eda/`.

### `train`

**5.8 Build the supervised matrix** — `make_supervised`. Adds, for every row:

- **Calendar**: day of week, day of month, month, day of year, ISO week, weekend
  flag, and sine/cosine encodings of month and day-of-year — the cyclical pairs
  so that December is adjacent to January rather than eleven units away.
- **Lags** of the target at 1, 2, 3, 7, 14, 28 and 365 days.
- **Rolling mean and standard deviation** over 7-, 14- and 28-day windows,
  computed on `target.shift(1)` so a row's own value can never enter its own
  rolling statistic.

The first year is consumed as burn-in for the 365-day lag.

**5.9 Split** — `last_year_train_test_split`. Everything on or before
(last date − 1 year) trains; everything after is the test horizon. Chronological,
never randomised, and no validation partition is carved out.

**5.10 Replace test-period weather with climatology** —
`climatological_weather`. This is the step that makes the evaluation honest.
Realised temperature for a future year is not knowable when a one-year-ahead
forecast is issued, so each test row's `Max_Temperature` and `Min_Temperature`
are overwritten with the **mean of that calendar day across the training years** —
the "normal" a planner actually holds in advance. Day 366 falls back to day 365;
a day-of-year absent from training falls back to the overall training mean.
Non-positive readings are excluded from the averages, because a few rows carry a
sentinel `0` rather than a real observation, which would otherwise drag the
normal down.

**5.11 Fit each model** on the training window only, once, with a fixed seed.

**5.12 Forecast recursively** — `_recursive_predict`, `resolve_recursive_features`.
Models that consume the feature matrix predict **one day at a time**. For each
step, the lag and rolling columns are rebuilt from a running history that starts
as real observations and is extended only by the model's own predictions;
calendar and socio-economic columns are read straight from the future frame,
since they are known in advance. The prediction is appended to the history and
the loop advances. The statistical models, which forecast their own horizon
natively, are called directly.

One shared routine serves all four feature-consuming models, so no model family
gets a more permissive rule than another.

**5.13 Score, attribute and export** — see [Scoring](#7-scoring) and
[Every file the run produces](#8-every-file-the-run-produces).

---

## 6. The models

Each lives in its own module under `forecasters/` behind one `fit`/`predict`
interface, so adding a tenth means writing one file and one line in
`models.create_model`.

| Model | Configuration | Why |
|---|---|---|
| Seasonal-naive | Period 365: today's forecast is the value 365 days ago | Reference baseline carrying the annual cycle and nothing else |
| Holt-Winters | Additive trend and seasonality, period 365, damped, ML-optimised | Seasonal amplitude doesn't scale with level; damping bounds a 365-step extrapolation |
| ARIMA | Order (3,1,3), deterministic linear drift | Differencing removes the level; drift sets the long-run direction |
| SARIMAX | Order (3,1,3), seasonal (2,0,1,7), stationarity and invertibility enforced, 300 iterations, 3 annual Fourier harmonic pairs as exogenous regressors | See below |
| Prophet | Yearly and weekly seasonality on, daily off | Daily seasonality is meaningless on daily aggregates |
| Random Forest | 500 trees, min 5 samples/leaf, sqrt features per split | Per-split subsampling decorrelates trees, preserving the variance reduction a recursive forecast leans on |
| XGBoost | 500 rounds, depth 6, learning rate 0.05 | Moderate depth and low rate limit overfitting on a few thousand rows |
| LightGBM | 500 rounds, learning rate 0.05 | Matched to XGBoost's budget, so the contrast is leaf-wise against depth-wise growth |
| LSTM | 1 layer, hidden 64, 28-day lookback, Adam 1e-3, MSE, batch 64, 150 epochs | Lookback spans four weekly cycles and matches the longest rolling window |

**Why SARIMAX carries Fourier terms.** The dominant cycle in daily electricity
demand is annual, but a `seasonal_order` period of 365 is computationally
impractical for statsmodels' state-space SARIMAX on a multi-year daily series —
the state vector scales with the period. So the weekly period is used for the
seasonal ARMA component, and the annual cycle is supplied instead as **three
sine/cosine harmonic pairs of day-of-year**, entering as a linear term in the
mean with the ARMA structure fitted to the residual. These depend only on the
calendar, so they are known for the entire horizon in advance — the same class of
information every other model already gets from its calendar features.

Two constraints, both learned the hard way and both worth preserving: AR/MA
orders that are multiples of 7 collide with `seasonal_period=7` and statsmodels
rejects them; and seasonal *differencing* combined with these exogenous terms
destabilises the fit into large oscillations, which is why `seasonal_order` has
`D=0`.

**The LSTM is not univariate.** It consumes the same supervised matrix as the
tree models — calendar, lag, rolling and socio-economic columns — as a 28-day
window of feature vectors, standardised on training statistics. Its multi-step
forecast is recursive in exactly the same way.

---

## 7. Scoring

All metrics are computed on the held-out year alone. Models are fit on the full
training history, but **no in-sample training-period forecast or metric is
produced** — only the held-out year is scored.

Nine metrics span three questions:

| Question | Metrics |
|---|---|
| How large are the errors? | MAE, MSE, RMSE (absolute, in MW) · MAPE, sMAPE, WAPE (scale-free percentages) · MASE (scaled by the mean day-over-day change) |
| Are the errors systematically signed? | MBE — the only metric that separates errors which cancel from errors of constant sign |
| How much variance is explained? | R² |

### The composite score

Each metric is min-max normalised across models within the split, mapped to
[0, 1] with 1 always best. R² is higher-is-better, the error metrics are
lower-is-better, and MBE enters as its absolute value since zero is optimal from
either side. The composite is a weighted sum of those normalised scores.

The weighting is a **two-level taxonomy**, not a flat average, and this matters.
Seven of the nine metrics — MAE, MSE, RMSE, MAPE, sMAPE, WAPE, MASE — are
correlated transforms of the same underlying question. A flat mean would hand
"how big is the error" roughly seven times the influence of bias or fit, purely
because it was measured seven ways.

Instead:

- **Error magnitude, bias and fit each get one third**, as top-level concepts.
- Within error magnitude, the **absolute**, **percentage** and
  **relative-to-naive** normalisations each get an equal share of that third —
  so no single way of expressing error size dominates the other two either.
- Within a normalisation, the share splits evenly across the metrics present.

Instantiated on the nine metrics, that gives MAE = MSE = RMSE = MAPE = sMAPE =
WAPE = 3.70% each, MASE = 11.11%, MBE = 33.33%, R² = 33.33%. The weights follow
mechanically from the taxonomy and are fixed before any scoring happens — they
are derived, not tuned. If you add or remove a metric, keep the taxonomy;
`default_metric_weights` recomputes the shares for you. Pass `metric_weights`
explicitly to override.

Giving bias a full third is a statement about the application: for medium-term
planning, a forecast that is persistently low in one direction across 365 days
compounds into a capacity shortfall, whereas errors that cancel leave the annual
position roughly right. A dispatch application could justify a different split.

### Feature attribution

Two complementary methods run for the four models that consume features:

- **Permutation importance** — shuffle one column, measure how much training-set
  RMSE rises. Global, error-based, model-agnostic. 10 repeats.
- **SHAP** — exact `TreeExplainer` for the ensembles, `GradientExplainer` for
  the LSTM, sampled on up to 1,000 training rows (200 windows for the LSTM). For
  the LSTM, each window's per-timestep attributions are summed across the
  lookback so every feature gets one value per sample.

These answer different questions — one is global and RMSE-based, the other
per-prediction, signed and additive — so both are kept rather than one treated
as redundant. Each is written twice: once over all features, once restricted to
the socio-economic subset, so the social signal isn't buried under the lag
columns that dominate any short-horizon ranking.

All attribution runs on **training rows only**. Nothing touches the test year.

---

## 8. Every file the run produces

Everything lands in `artifacts/` (git-ignored).

**Data audit** — `selected_workbooks.csv` (which workbook was chosen per year
and its score), `yearly_data_quality.csv` (rows, missing count and missing
percent per column per year), `social_feature_forecast_methods_used.csv` (how
each indicator was projected into uncovered years),
`social_features_extended_v3.csv` and `social_feature_forecast_methods_v3.csv`
(the separate five-year reference extension).

**Forecasts** — `forecast_<model>.csv` (date, actual, prediction) and
`forecast_<model>.png` for each of the nine.

**Metrics** — `model_metrics_test.csv` and `model_comparison_test.csv` (all nine
metrics per model), `model_comparison.xlsx` (Methodology sheet plus raw metrics
and normalised scores), `run_metadata.json` (target, train and test date ranges,
row counts, model list).

**Attribution** — per model: `sensitivity_<model>.csv/.png`,
`sensitivity_<model>_interpretation.txt` (a one-line plain-English summary of
the top five), the `_social` variants of each,
`feature_importance_<model>_social.png`, `shap_values_<model>.csv`,
`shap_summary_<model>.png` and their `_social` variants.

When a model has no social features in its input, the pipeline writes an
explicit "unavailable" chart and CSV stating why, rather than omitting the
artifact silently.

**Charts** — `artifacts/eda/` and `artifacts/model_comparison_charts/`,
described next.

---

## 9. The charts and how to read them

What each chart looks like, and what it is for. Nothing below describes any
particular dataset's outcome — these are the questions each chart answers.

### Data quality (`artifacts/eda/`)

**`eda_missing_values.png`** — bar chart, one bar per column with missing values,
tallest first. On a clean run it prints "No missing values remain in the cleaned
dataset" instead of bars. Read it as a check that cleaning did what it claimed;
any surviving bar points at a column the loader could not fill.

**`eda_missing_heatmap.png`** — every numeric column across the x-axis, every row
of the dataset down the y-axis, coloured where a value is absent. Bars of colour
mean a contiguous outage; scattered speckle means isolated bad cells. The two
tell you very different things about the source.

### The demand series itself

**`eda_daily_demand_over_time.png`** — the whole daily series as one line. The
chart to look at first. It shows whether the annual cycle repeats with a stable
shape, whether the level drifts up or down across years, and whether any period
looks discontinuous in a way that suggests a data problem rather than real
demand.

**`eda_target_distribution.png`** — histogram beside a boxplot. Whether demand is
roughly symmetric or skewed, whether it is bimodal (which for a cooling-driven
grid would suggest two distinct regimes rather than one continuum), and how far
the extreme days sit from the body of the distribution.

**`eda_monthly_average_demand.png`** and **`eda_seasonal_average_demand.png`** —
mean demand by calendar month and by season. Where the annual peak and trough
fall, and how large the swing is.

**`eda_unrestricted_demand_by_month.png`** and **`eda_target_by_season.png`** —
the same cuts as boxplots rather than bars, so you also see *spread*. A month
with a moderate mean but a wide box is a hard month to forecast; a narrow box
means the level is dependable. This distinction is invisible in the bar charts.

### Weather and drivers

**`eda_unrestricted_demand_vs_max_temperature.png`** and
**`eda_unrestricted_demand_vs_min_temperature.png`** — scatter of demand against
temperature, one point per day. The shape is
the finding: a flat left portion turning upward past a threshold indicates a
cooling-load regime, a U shape indicates both heating and cooling. The threshold
where the curve turns is the temperature at which cooling load begins to bite.
Vertical spread at a fixed temperature tells you how much demand is explained by
something *other* than temperature — which is precisely why the calendar and
season features are retained alongside temperature rather than treated as
redundant with it.

**`eda_max_temperature_by_month.png`** and
**`eda_min_temperature_by_month.png`** — monthly temperature boxplots, the
weather counterpart to the demand boxplots.

**`eda_correlation_heatmap.png`** — a square matrix of every numeric column
against every other, blue through red for −1 to +1. Two uses: find what moves
with demand, and spot blocks of mutually near-identical predictors. A solid
block of dark red among the socio-economic indicators means those columns carry
largely the same information, so importance attributed to any one of them can be
shared with its neighbours.

**`eda_target_correlations.png`** — one bar per feature, its correlation with
demand, sorted. A quick ranking, but a linear one: a feature with a strong
non-linear relationship to demand (temperature is the obvious candidate) can
show a modest bar here while mattering a great deal to the models.

**`eda_yearly_social_feature_trends.png`** — one line per socio-economic
indicator, annual averages. Since these are raw units on a shared axis, the
large-magnitude series dominate visually; read it for *shape* — which indicators
trend steadily, which are flat, which turn — rather than for level comparison.
The projected years appear as continuations of these lines.

> **A wrinkle in the EDA season labels.** These charts derive their own
> five-way `Season` column (Winter Dec–Jan, Spring Feb–Mar, Summer Apr–Jun,
> Monsoon Jul–Sep, Autumn Oct–Nov). That is *not* the four-way scheme in the
> workbook flags that the models consume (Winter Dec–Feb, Summer Mar–Jun,
> Monsoon Jul–Sep, Pre Monsoon Oct–Nov). The EDA charts and the model features
> therefore slice the year slightly differently — worth remembering before
> reading one as evidence about the other.

### Forecasts

**`forecast_<model>.png`** — actual demand as a solid black line, the model's
test-year forecast as a dashed orange line, over the held-out year only. What to
look for, in order: does the forecast reproduce the *shape* of the year at all
(a near-flat line means the model has no way to represent the annual cycle);
does it sit above or below the actual line for long stretches (visible bias); and
does it capture the peaks, or track the middle and clip the extremes? For a
recursive forecast, also check whether error grows with horizon or simply tracks
the seasonally difficult periods — those look different and imply different
fixes.

### Model comparison (`artifacts/model_comparison_charts/`)

**`test_<metric>.png`** — ten charts, one per metric plus the composite. Each is
a bar chart of all nine models, sorted best-first for that metric. Comparing
`test_rmse.png` against `test_mae.png` shows who is hurt by a few large misses,
since RMSE squares errors and MAE does not. `test_mbe.png` is the one to read
differently: bars extend both ways from zero, and the sign matters more than the
size — a model near zero has errors that cancel; a large bar in either direction
means a forecast that is persistently high or persistently low all year.

**`test_composite_score.png`** — the taxonomy-weighted ranking, fixed to a 0–1
axis, best model at 1.0 by construction. Read alongside `test_rmse.png`: where
the two orderings disagree, a model is being rewarded for low bias or good fit
while being penalised on raw error size, or the reverse. Those disagreements are
the informative part — a model can be too erratic day-to-day to schedule against
while still estimating the annual position well, and the two rankings separate
exactly that.

**`model_comparison_test.png`** — a plain RMSE bar chart written by the CLI, the
quick single-number view.

### Feature attribution

**`sensitivity_<model>.png`** — horizontal bars, top 20 features by the RMSE
increase caused by shuffling them. Longer bar, more the model depends on it.
Expect lag and rolling features to dominate any such ranking, which is why the
social-only variant exists.

**`sensitivity_<model>_social.png`** and **`feature_importance_<model>_social.png`** —
the same ranking restricted to the nineteen socio-economic indicators, by
permutation sensitivity and by the estimator's built-in importance respectively.
Agreement between the two methods is the signal worth trusting; disagreement
usually means correlated indicators sharing credit.

**`shap_summary_<model>.png`** — a beeswarm. One row per feature, ordered by mean
absolute impact. One dot per sampled day, positioned horizontally by how much
that feature pushed that day's prediction (right = pushed up, left = pushed
down), coloured by the feature's own value (blue low, red high). Three things to
read from it:

- **Row order** — overall influence ranking.
- **Row width** — a wide row means the feature swings predictions a lot on some
  days; a tight cluster at zero means it rarely matters.
- **The colour pattern** — this is what a bar chart cannot show. Red on one side
  and blue on the other means a monotone relationship, and which side tells you
  the direction. Red and blue interleaved means the effect depends on context,
  or the relationship is non-monotone.

Attributions are in the target's units (MW) for the tree models, but the LSTM
operates on standardised inputs, so its values are in standardised units — only
the *ordering* is comparable between the LSTM panel and a tree panel, never the
magnitudes.

---

## 10. Manuscript figures

`scripts/` holds the figure generators for the accompanying paper, deliberately
kept outside the pipeline. They read artifacts the pipeline has already written,
so run the pipeline first, and they write to `artifacts/paper_figures/`.

```powershell
python scripts/make_paper_figures.py          # EDA, ablation and comparison charts
python scripts/make_forecast_grid_figure.py   # all nine forecasts as one panel grid
python scripts/make_shap_grid_figure.py       # 2x2 SHAP beeswarm grid
python scripts/refit_univariate_sarimax.py    # SARIMAX-without-exogenous ablation series
```

`refit_univariate_sarimax.py` refits SARIMAX with the Fourier regressors removed
and everything else identical, exporting its test-year series so the two
variants can be plotted together — an ablation that isolates what the annual
harmonic representation contributes. Run it before
`make_paper_figures.py`, which reads its output.

`make_shap_grid_figure.py` refits the four feature-consuming models to recompute
SHAP values (several minutes, mostly the LSTM) and caches them to
`artifacts/shap_grid_cache.pkl`, so re-running to adjust layout is instant.
Delete the cache to force a genuine refit.

---

## 11. Tests, layout, licence

```powershell
pytest
```

```
config/default.yaml            canonical configuration
src/electricity_forecasting/
    config.py                  loads config/default.yaml into a typed ForecastConfig
    data.py                    workbook discovery, loading, cleaning, social merge and projection
    features.py                calendar/lag/rolling features, climatology, recursive resolution
    split.py                   chronological hold-out
    models.py                  Forecaster interface, sklearn adapter, factory
    forecasters/               one module per forecasting method
    metrics.py                 the nine metrics
    model_comparison.py        normalisation, taxonomy weights, composite, comparison charts
    sensitivity.py             permutation importance and SHAP, overall and social-only
    plots.py                   EDA, forecast, importance and SHAP charts
    pipeline.py                prepare / train orchestration
    cli.py                     command-line entry point
    intervals.py               conformal prediction intervals — see note below
scripts/                       manuscript figure generation
tests/                         test suite
CLAUDE.md                      data provenance and design decisions in detail
```

**A note on `intervals.py`.** `ConformalInterval` and `metrics.interval_metrics`
implement residual-based prediction intervals and are covered by the tests, but
they are **not currently wired into the pipeline** — no run produces interval
artifacts. They are available for use, not part of the documented flow.

### Licence

No licence file is included yet. Without one, default copyright applies and
others may not reuse the code; add a `LICENSE` before making the repository
public if that is not the intent.
