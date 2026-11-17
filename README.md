# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/adelbudi88-crypto/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/adelbudi88-crypto/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-by-hour Twitch CCU forecast for GTA VI launch day (2026-11-19 UTC), as five quantiles per hour, from the first hour's print plus 38–42 comparable big-launch curves. Python 3.13, standard library only, deterministic, offline.

## Model

For each comparable launch with a complete release day, take the log-ratio curve `r_h = log(viewers_h / viewers_1)`. The curve's shape depends strongly on log(hour-1 viewers) — big hour-1 prints stay flat or dip, tiny ones explode later (R² ≈ 0.87 for hour 24). The forecast conditions on log(hour-1) two ways, blended 70/30 in log space: a local-linear regression of `r_h` on log(hour-1) (Gaussian kernel, bandwidth 1.5, console-only comparables up-weighted ×2, predictions clipped to the fitted range), and a kernel-weighted median of `r_h` (bandwidth 1.0). Each component's spread is its kernel-weighted residual sd per hour; quantiles are `pred + z_tau · sd`. With fewer than 5 usable comparables it falls back to the house baseline (persistence, σ = 0.15·√(h−1)).

Honest validation — leave-one-out over the 37 usable practice comparables (baseline in brackets): mean pinball loss **0.379** (0.930) overall, **0.249** (0.321) for launches with hour-1 ≥ 20k (the band holding all four practice cases), **0.208** (0.261) for hour-1 ≥ 100k (the band GTA VI's hour-1 is expected in); typical miss 77% (213%), 90% coverage 0.893 (0.624).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=hynfsjfx&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
