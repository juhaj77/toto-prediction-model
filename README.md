# Toto Prediction Model

A two-model TensorFlow.js system for predicting Toto race outcomes. The project now trains and serves two complementary neural networks:

- Runner Model (per-horse): learns the win/placing likelihood of each runner from its own history and static features.
- Race Model (field-aware): refines per-runner scores by modeling interactions within a race field (which horses matter for each other) and outputs a score per runner.

Training data is scraped from [Veikkaus](https://www.veikkaus.fi).

---

## Architecture Overview

- Runner Model
  - Inputs: per-runner time series (historical performances) + static features (runner/race).
  - Core: LSTM over history + Dense layers for static features, then merged and scored per runner.
  - Output: probability/score for each runner independently.

- Race Model
  - Inputs: per-runner sequences and static features for all runners in a race, plus a mask for padding.
  - Core: TimeDistributed encoders → Multi-Head Self-Attention across runners (field-level interactions) → residual projection + LayerNorm → output head.
  - Output: per-runner score that accounts for race context; padding positions are masked out.

These models can be trained and used independently or together (Runner → features → Race).

---

## Project Structure

- `model-training/` – Data scraping and model training
- `front-end/` – Prediction user interface

---

## Install
Run `npm install` in both `model-training` and `front-end`.

---

## Data Collection

Race data is appended to:

`training_data.json`

⚠️ The Veikkaus API retains race data for approximately 10 days only. To build a sufficiently large training dataset, run the scraper periodically.

### Run the scraper

From the `model-training` directory:

```bash
node scraper
```

This appends new race data to `training_data.json`.

---

## Model Training

From the `model-training` directory:

- Train Runner Model
```bash
node model_runner
```

- Train Race Model
```bash
node model_race
```

Training scripts will save `model.json` plus weight shards and write feature mapping files used by the front-end.

---

## Deploying the Trained Models to the Front-End

After training, copy artifacts as follows:

1. Copy `mappings_runner.json`
   → to `front-end/public`

2. Copy `model.json`
   from `model-runner/`
   → to `front-end/public/model-runner`

3. Copy `mappings_race.json`
   → to `front-end/public`

4. Copy `model.json`
   from `model-race/`
   → to `front-end/public/model-race`

---

## Running the Front-End

From the `front-end` directory:

```bash
npm run dev
```

---

## Live Version

A working deployment is available here:
👉 [toto prediction system](https://toto-kesm.onrender.com)
