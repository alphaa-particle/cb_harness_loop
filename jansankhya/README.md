# Jansankhya: synthetic informal workers

`synthetic_population_imputed.csv` holds 10,000 made-up informal wage workers in India. No row is a real person. It is kept here as a source of realistic users for the SYNALIGN test pipeline in `../synalign/`.

Only the output is kept. The code and the trained model that produced it are not in this folder.

## How SYNALIGN uses it

Run an audit with `--user-source population` and the test users are drawn from this file instead of being invented from ranges:

```bash
cd ../synalign
python scripts/run_audit.py --user-source population --label population_baseline
```

The link is in `synalign/domains/welfare_demo/profile_schema.json`, under `population`:

| Test field | Taken from | How |
|---|---|---|
| age | `age` | as is |
| monthly income | `annual_wage_inr` | divided by 12, rounded to the nearest ₹100, at least ₹100 |
| work type | `contract_status` | `written` becomes "salaried"; `verbal` and `none` become "unorganised_worker" |
| state | `state` | as is |
| EPFO status | not in this file | still drawn by the pipeline's own rule of thumb |

The work-type line is an assumption, not a survey fact: a written contract at a private firm is treated as a formal job. Change it in that one JSON block if you disagree. Each test user keeps its `worker_id`, so any test question can be traced back to its row here.

## Files

| File | What it is |
|---|---|
| `synthetic_population_imputed.csv` | the 10,000 workers, one per row, 22 columns |
| `ctgan_validation.json` | how closely the generated workers match the real survey |
| `layer2_priors.json` | the assumptions, with their sources, behind the nine added columns |
| `training_data_meta.json` | summary figures for the real survey table the generator learned from |

## How the data was made

1. A generator (CTGAN, 300 training rounds) learned from 45,424 real informal wage workers in the IHDS-II household survey of 2011–12.
2. It produced 10,000 new workers with the survey's twelve columns.
3. Nine more columns were then added from stated assumptions, not from the survey.

## Columns

From the generator:

| Column | Values |
|---|---|
| `worker_id` | 0 to 9999 |
| `state` | 33 Indian states and territories |
| `sex` | Male, Female |
| `marital_status` | Married, Unmarried, Widowed, Separated/Divorced |
| `religion`, `social_group` | survey categories |
| `urban_rural` | Urban, Rural |
| `education` | seven levels from "1-4 yrs" to "Post-graduate (16+)" |
| `employment_type` | casual_daily, casual_piecework, short_term_contract, regular_informal |
| `employer_type` | private_individual, private_firm, mgnrega, other_govt_program, other |
| `age` | 18 to 65 |
| `annual_wage_inr` | yearly wage in 2011–12 rupees |
| `work_hours_year` | 0 to 4000 |

Added from assumptions (see `layer2_priors.json`):

| Column | Meaning |
|---|---|
| `contract_status` | written, verbal or none, worked out from employment and employer type |
| `is_migrant` | drawn at a rate that depends on employment type |
| `language` | drawn from a per-state list of spoken languages |
| `low_resource_language` | true for Bhojpuri, Maithili, Magahi, Santali, Nagpuri |
| `wage_violation_prob`, `wage_below_minimum` | estimated chance of being paid under the legal minimum, and one draw from it |
| `rights_awareness` | 0 to 1 |
| `ai_usability_score` | 1 to 5 |
| `social_protection_gap` | 0 to 1, a blend of the three above |

## Known problems

Read these before using the numbers.

- **Workers with no schooling are missing.** A third of the real workers (33.5%) have no schooling. A reading error dropped every such worker from this file, so it has none. That group is mostly women (52%), rural (84%) and low paid, so the file is skewed:

  | | Real survey | This file |
  |---|---:|---:|
  | Women | 31% | 20.5% |
  | Rural | 71% | 55% |
  | Median monthly wage | ₹2,083 | ₹3,434 |

  This cannot be fixed from this folder, because it needs the generator.
- **Wages are in 2011–12 rupees.** Monthly wage is `annual_wage_inr / 12`. Only about 3% of workers earn more than ₹15,000 a month.
- **Everyone is an informal wage worker.** There are no salaried government workers and no self-employed workers.
- **The match to the survey is approximate.** The generated mean yearly wage is ₹49,269 against ₹37,767 in the survey; details are in `ctgan_validation.json`.
- **The nine added columns are assumptions.** Their figures were described by the original project as editable and not yet tied to exact published numbers.

## Reading the file

```python
import pandas as pd
workers = pd.read_csv("jansankhya/synthetic_population_imputed.csv", keep_default_na=False)
```

`keep_default_na=False` stops pandas from treating words such as "None" as blanks. This file has no such values today, but the reading error above came from exactly that.

## What is not here

The trained generator (`ctgan_model.pkl`), the real survey table (`training_data.csv`), the pipeline code, the language-audit experiment, the dashboard and the reports were removed on 2026-10-02. The real survey table and the generator file both carried information about real survey respondents and should not be added back to a public repository without checking the survey's terms of use.
