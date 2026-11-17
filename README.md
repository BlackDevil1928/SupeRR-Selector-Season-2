# Royals Pressure Lab

An interactive cricket analytics dashboard for the Rajasthan Royals SupeRR Selector Season 2 submission. Explore the ranked batting and bowling shortlists, compare players under pressure, and reproduce individual ratings in the browser.

## What you can do

- Explore the ranked shortlists and the evidence behind each player’s position.
- Compare players using rating bars, an exposure scatter plot, and a radar chart. The comparison table shows the underlying values.
- Load a player into the score calculator and change the residuals, sample sizes, or model parameters to test a scenario.
- Switch to the historical IPL cross-check for a separate view of the 2018–2021 Cricsheet sample.

## Run the dashboard

Open [`index.html`](index.html) in a modern browser. The dashboard is static and requires no build step or server. The graphs and calculator run in the browser from embedded player-level aggregates.

To publish the dashboard, drag the folder containing `index.html` onto [Netlify Drop](https://app.netlify.com/drop). Keep `external_ipl_results.csv` and `model/` alongside it so the download links work. If you already have a Netlify project, upload the folder from that project’s **Deploys** page to keep its URL.

## How the score works

The model compares observed performance against a matched peer baseline for each delivery. It combines the resulting residuals separately for pressure and ordinary situations. The report rating is expressed in run-equivalents per 100 events.

```text
Rating = 100 × (
    pressure residual sum / (pressure events + 60)
    − 0.35 × ordinary residual sum / (ordinary events + 60)
)
```

The primary evidence tier requires at least **36 pressure events, 72 ordinary events, and 8 distinct pressure matches**. Players in this tier rank first, with ratings determining order within each evidence tier. The bowling shortlist includes one unscored fallback for the required tenth slot. Calculator scenarios are exploratory and do not change the submission rankings.

## Reproduce the rankings

Use Python 3.12 and run these commands from the repository root. Replace the example paths with the actual ZIP paths.

```bash
python -m pip install -r model/requirements.txt
python model/rank_dropdown.py "/path/to/organizer_data.zip" --out reproduced
python model/validate_cricsheet.py "/path/to/ipl_json.zip" --out reproduced/External_IPL_Results.csv
```

The ranking script writes batting and bowling ranking CSVs and phase summaries into `reproduced/`. The validation script reproduces the historical IPL cross-check, which covers first innings from 2018–2021 and is separate from the submitted competition model. Get the source ZIP from the [Cricsheet IPL JSON archive](https://cricsheet.org/downloads/ipl_json.zip).

## Repository files

| File | Purpose |
| --- | --- |
| [`index.html`](index.html) | Interactive dashboard and browser calculator |
| [`external_ipl_results.csv`](external_ipl_results.csv) | Public IPL cross-check output |
| [`model/rank_dropdown.py`](model/rank_dropdown.py) | Rebuild the submission shortlists |
| [`model/rr_pressure_model.py`](model/rr_pressure_model.py) | Match-state and pressure-rating model |
| [`model/validate_cricsheet.py`](model/validate_cricsheet.py) | Reproduce the historical IPL check |
| [`model/requirements.txt`](model/requirements.txt) | Pinned Python dependencies |

## Data sources

The dashboard displays derived player-level competition results. Obtain the original organizer data pack separately and keep it out of the public repository. The historical IPL cross-check uses [Cricsheet match data](https://cricsheet.org/matches/) under the Open Data Commons Attribution License.

**Prepared by Lavesh Hasani** for the Rajasthan Royals SupeRR Selector Season 2 case study.
