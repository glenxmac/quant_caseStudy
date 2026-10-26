**validation (10 days)** (error = MAE as % of mean true rate)

| Model | +5 min | +10 min | +15 min | overall |
|---|---|---|---|---|
| Raw count carried forward (persistence) | 18.5% | 19.5% | 20.4% | 19.5% |
| Hour-of-day profile (entry x weekday/weekend x 15 min) | 10.9% | 10.5% | 10.5% | 10.6% |
| + weather (profile x rain) | 10.8% | 10.5% | 10.4% | 10.6% |
| + live entrance counts: Poisson GLM + guard (main) | 6.5% | 7.0% | 7.7% | 7.0% |
| GBM + live counts + guard (check) | 6.4% | 7.0% | 7.5% | 7.0% |
| Oracle: true rate now carried forward (unreachable) | 4.4% | 7.0% | 9.1% | 6.8% |

**test (10 days)** (error = MAE as % of mean true rate)

| Model | +5 min | +10 min | +15 min | overall |
|---|---|---|---|---|
| Raw count carried forward (persistence) | 25.0% | 25.9% | 27.0% | 26.0% |
| Hour-of-day profile (entry x weekday/weekend x 15 min) | 9.9% | 9.8% | 9.7% | 9.8% |
| + weather (profile x rain) | 9.6% | 9.5% | 9.4% | 9.5% |
| + live entrance counts: Poisson GLM + guard (main) | 6.4% | 6.7% | 7.3% | 6.8% |
| GBM + live counts + guard (check) | 6.8% | 7.2% | 7.6% | 7.2% |
| Oracle: true rate now carried forward (unreachable) | 4.6% | 7.4% | 9.8% | 7.3% |

**fresh (10 days)** (error = MAE as % of mean true rate)

| Model | +5 min | +10 min | +15 min | overall |
|---|---|---|---|---|
| Raw count carried forward (persistence) | 20.4% | 21.3% | 22.3% | 21.4% |
| Hour-of-day profile (entry x weekday/weekend x 15 min) | 10.1% | 9.9% | 9.7% | 9.9% |
| + weather (profile x rain) | 9.8% | 9.7% | 9.6% | 9.7% |
| + live entrance counts: Poisson GLM + guard (main) | 6.5% | 6.7% | 7.3% | 6.8% |
| GBM + live counts + guard (check) | 6.2% | 6.6% | 7.0% | 6.6% |
| Oracle: true rate now carried forward (unreachable) | 4.5% | 7.3% | 9.6% | 7.1% |

**held-out: test + fresh (20 days)** (error = MAE as % of mean true rate)

| Model | +5 min | +10 min | +15 min | overall |
|---|---|---|---|---|
| Raw count carried forward (persistence) | 22.8% | 23.6% | 24.7% | 23.7% |
| Hour-of-day profile (entry x weekday/weekend x 15 min) | 10.0% | 9.8% | 9.7% | 9.8% |
| + weather (profile x rain) | 9.7% | 9.6% | 9.5% | 9.6% |
| + live entrance counts: Poisson GLM + guard (main) | 6.4% | 6.7% | 7.3% | 6.8% |
| GBM + live counts + guard (check) | 6.5% | 6.9% | 7.3% | 6.9% |
| Oracle: true rate now carried forward (unreachable) | 4.5% | 7.4% | 9.7% | 7.2% |
