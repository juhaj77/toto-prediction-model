# Toto Prediction Model

A mixed neural network architecture combining **LSTM** and **Dense** layers.

- **LSTM** processes each runner’s historical race data (time-series input).
- **Dense** processes static race and runner features.
- The outputs are merged into a single predictive model.

Training data is scraped from [Veikkaus](https://www.veikkaus.fi).

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

`training.json`

⚠️ The Veikkaus API retains race data for approximately 10 days only.  
To build a sufficiently large training dataset, run the scraper periodically.

### Run the scraper

From the `model-training` directory:

```bash
node scraper
```

This appends new race data to `training_data.json`.

---

## Model Training

From the `model-training` directory:

```bash
node model_runner
```
```bash
node model_race
```

---

## Deploying the Trained Model to the Front-End

After training:

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
