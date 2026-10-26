# Offline A/B testing

The A/B module is an **offline simulation**, not a live-user experiment. It hashes
`(user_id, experiment_name)` to assign a stable control or treatment group. In
historical data, each rating in the supplied test period is treated as an
impression, and ratings at or above `rating_click_threshold` are counted as
proxy clicks. This does not model recommendation exposure or establish the
causal impact of either recommender.

Before a real experiment, calculate the planning sample size with
`required_sample_size_per_group`, passing the alpha, power, minimum detectable
effect, and baseline click rate from `configs/experiments.yaml`. The configured
`baseline_click_rate` is a planning assumption and should be replaced by an
appropriate historical baseline before using the sample-size estimate.

`simulate_experiment` reports user counts, proxy impressions and clicks, group
CTR, the Welch t-test by default (or Mann-Whitney U), a bootstrap interval, the
required per-group sample size, and whether CUPED was applied. An optional
pre-period ratings frame supplies the CUPED covariate. Passing a SQLAlchemy
engine stores each result in the app database's `experiments` table; apply
migrations with `alembic upgrade head` first.

`aa_false_positive_rate` estimates the false-positive rate on equal-rate
synthetic groups. Its expected rate is approximately alpha (5% at the default
alpha of 0.05), subject to simulation variability. This check is synthetic and
does not use MovieLens data.
