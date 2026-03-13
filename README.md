# Toto Prediction Model

A multi-model TensorFlow.js system for predicting Toto race outcomes. The project features three complementary neural network architectures to capture different aspects of horse racing data.

Training data is scraped from [Veikkaus](https://www.veikkaus.fi). A working deployment of this project is available here:
👉 [toto prediction system](https://toto-kesm.onrender.com)

---

## Architecture Overview

The system consists of three distinct model types, each designed for a specific level of abstraction:

### 1. Runner Model (Per-Horse)
- **Focus:** Individual horse performance and potential.
- **Inputs:** Per-runner time series (historical performances) + static features (horse/race).
- **Core:** LSTM layers process the history, while Dense layers handle static features. These branches are merged to score each runner independently.
- **Output:** A raw probability/score for each runner without considering the strength of the rest of the field.

### 2. Race Model (Field-Aware)
- **Focus:** Field dynamics and relative strength.
- **Inputs:** Static features for all runners in a race simultaneously.
- **Core:** Uses **Multi-Head Self-Attention** to model interactions between all horses in the field. It learns to "rank" horses by looking at who else is in the same race.
- **Output:** A per-runner score that accounts for the competition context; padding positions for empty stalls are masked out.

### 3. Mixed Model (Hybrid Ensemble)
- **Focus:** The most comprehensive view, combining deep history with field-level context.
- **Inputs:** - `history_input`: Full historical sequences for all runners.
    - `static_input`: Static features for all runners.
    - `mask_input`: Binary mask to handle varying field sizes.
- **Core Architecture:**
    - **Runner Encoder:** A TimeDistributed LSTM stack that encodes the history of every horse in the race into a compact embedding.
    - **Static Branch:** Processes horse/race-specific features with LayerNormalization for stability.
    - **Attention Block:** A 64-dimensional Multi-Head Attention layer (8 heads) that allows the model to compare the encoded "strength" of each horse against all others in the field.
    - **FFN Residual Block:** A feed-forward network with residual connections and LayerNorm to refine the attended representations.
- **Output:** Sigmoid probabilities for each runner slot, optimized using a custom combined race loss (or ListNet for ranking).

---

## Key Features

- **Advanced Masking:** All field-aware models use sophisticated masking to ensure that empty runner slots (e.g., in a race with only 10 horses vs 16) do not affect the attention weights or gradients.
- **Feature Reliability:** Implements "Known-flags" for unreliable data (e.g., missing records or uncertain breed info), allowing the model to learn when to ignore specific input features.
- **Residual Learning:** The Mixed and Race models utilize residual connections (Add/Concatenate) to prevent vanishing gradients and maintain information flow from the raw features to the final decision.
- **Multi-Head Attention:** Enables the model to focus on different aspects of the competition simultaneously (e.g., one head focusing on speed records, another on recent form).

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
```
node -e "require('./model_runner').runTraining({ runId: 'runner-ln', epochs: 80, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 768, bestBy: 'val_auc', dropout: 0.15, l2: 1e-4, swapBNtoLN: true, useLayerNormInStatic: true })" -- --backend=gpu
```
or
```
node -e "require('./model_runner').runTraining({ runId: 'runner-lr1e-3-reg', epochs: 80, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 768, bestBy: 'val_auc', dropout: 0.25, l2: 2e-4 })" -- --backend=gpu
```

- Train Race Model
```
node -e "require('./model_race').runTraining({ runId: 'race-ln2', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 8, learningRate: 2e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_auc', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 230, dropout: 0.2, outDropout: 0.3, l2: 1e-4, swapBNtoLN: true, useLayerNormInStatic: true })" -- --backend=cpu
```
- Train Mixed Model

```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-preenc16-auxanneal2', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 10, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 768, bestBy: 'val_ndcg3', auxLossWeight: 0.6, auxSchedule: { from: 0.6, to: 0.3, startEpoch: 25, endEpoch: 45 }, useListNet: false, swapBNtoLN: true, useLayerNormInStatic: true, preEncodeHistory: { units: 16, useLayerNorm: true }, attnHeads: 12, embedDim: 116, ffnDim: 230, runnerProjDim: 64, runnerLstm2Units: 48, dropout: 0.20, outDropout: 0.30, l2: 1e-4, labelSmoothing: 0.05 })" -- --backend=gpu
```

Training scripts will save `model.json` and write feature mapping files used by the front-end.

---

## Deploying the Trained Models to the Front-End

After training, copy artifacts as follows:

1. Copy `mappings_runner.json`
   → to `front-end/public`

2. Copy `model.json`
   from `model-runner/runs/[runId]]/`
   → to `front-end/public/model-runner`

3. Copy `mappings_race.json`
   → to `front-end/public`

4. Copy `model.json`
   from `model-race/runs/[runId]]/`
   → to `front-end/public/model-race`

3. Copy `mappings_mixed.json`
   → to `front-end/public`

4. Copy `model.json`
   from `model-mixed/runs/[runId]]/`
   → to `front-end/public/model-mixed`

---

## Running the Front-End

From the `front-end` directory:

```bash
npm run dev
```
