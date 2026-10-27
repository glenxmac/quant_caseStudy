**validation (10 days)** (error = MAE as % of mean true rate)

| Model | +5 min | +10 min | +15 min | overall |
|---|---|---|---|---|
| Raw count carried forward (persistence) | 18.5% | 19.5% | 20.6% | 19.5% |
| Hour-of-day profile (entry x weekday/weekend x 15 min) | 11.4% | 11.1% | 11.0% | 11.2% |
| + weather (profile x rain) | 10.9% | 10.6% | 10.6% | 10.7% |
| + live entrance counts: Poisson GLM + guard (main) | 6.8% | 7.3% | 8.0% | 7.3% |
| GBM + live counts + guard (check) | 6.8% | 7.3% | 7.8% | 7.3% |
| Oracle: true rate now carried forward (unreachable) | 4.5% | 7.2% | 9.5% | 7.1% |

**test (10 days)** (error = MAE as % of mean true rate)

| Model | +5 min | +10 min | +15 min | overall |
|---|---|---|---|---|
| Raw count carried forward (persistence) | 25.3% | 26.3% | 27.4% | 26.3% |
| Hour-of-day profile (entry x weekday/weekend x 15 min) | 10.3% | 10.2% | 10.1% | 10.2% |
| + weather and event calendar (profile x rain x event) | 9.8% | 9.6% | 9.6% | 9.7% |
| + live entrance counts: Poisson GLM + guard (main) | 6.6% | 7.0% | 7.6% | 7.0% |
| GBM + live counts + guard (check) | 7.1% | 7.5% | 7.9% | 7.5% |
| Oracle: true rate now carried forward (unreachable) | 4.7% | 7.7% | 10.3% | 7.6% |

**fresh (10 days)** (error = MAE as % of mean true rate)

| Model | +5 min | +10 min | +15 min | overall |
|---|---|---|---|---|
| Raw count carried forward (persistence) | 20.6% | 21.6% | 22.7% | 21.7% |
| Hour-of-day profile (entry x weekday/weekend x 15 min) | 10.8% | 10.6% | 10.4% | 10.6% |
| + weather and event calendar (profile x rain x event) | 10.0% | 9.8% | 9.7% | 9.8% |
| + live entrance counts: Poisson GLM + guard (main) | 6.8% | 7.0% | 7.6% | 7.1% |
| GBM + live counts + guard (check) | 6.6% | 7.0% | 7.5% | 7.0% |
| Oracle: true rate now carried forward (unreachable) | 4.7% | 7.7% | 10.2% | 7.5% |

**held-out: test + fresh (20 days)** (error = MAE as % of mean true rate)

| Model | +5 min | +10 min | +15 min | overall |
|---|---|---|---|---|
| Raw count carried forward (persistence) | 23.0% | 23.9% | 25.1% | 24.0% |
| Hour-of-day profile (entry x weekday/weekend x 15 min) | 10.5% | 10.4% | 10.2% | 10.4% |
| + weather and event calendar (profile x rain x event) | 9.9% | 9.7% | 9.6% | 9.7% |
| + live entrance counts: Poisson GLM + guard (main) | 6.7% | 7.0% | 7.6% | 7.1% |
| GBM + live counts + guard (check) | 6.8% | 7.3% | 7.7% | 7.3% |
| Oracle: true rate now carried forward (unreachable) | 4.7% | 7.7% | 10.2% | 7.6% |
