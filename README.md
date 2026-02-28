# Toto Prediction Model

A mixed neural network architecture combining **LSTM** and **Dense** layers.

- **LSTM** processes each runner’s historical race data (time-series input).
- **Dense** processes static race and runner features.
- The outputs are merged into a single predictive model.

Training data is scraped from [Veikkaus](https://www.veikkaus.fi).

---

## Project Structure

- `model-learning/` – Data scraping and model training  
- `front-end/` – Prediction user interface  

---

## Install
Run `npm install` in both `model-learning` and `front-end`.

---

## Data Collection

Race data is appended to:

`ravit_opetusdata.json`

⚠️ The Veikkaus API retains race data for approximately 10 days only.  
To build a sufficiently large training dataset, run the scraper periodically.

### Run the scraper

From the `model-learning` directory:

```bash
node scraper
```

This appends new race data to `ravit_opetusdata.json`.

---

## Model Training

From the `model-learning` directory:

```bash
node model
```

This trains the mixed LSTM + Dense model.

---

## Deploying the Trained Model to the Front-End

After training:

1. Copy `mappings.json`  
   → to `front-end/public`

2. Copy `model_full.json`  
   from `ravimalli-mixed/`  
   → to `front-end/public/ravimalli-mixed`

---

## Running the Front-End

From the `front-end` directory:

```bash
npm run dev
```

## TODO

### Migrating from runner-based to race-based model architecture

#### Current state: runner-based model

The model evaluates each horse **independently**, without any knowledge of the other runners in the same race. Each horse-race pair forms its own row in the tensor.

**Current tensors:**

```
History tensor:  [n_runners, 8, 25]
                  │           │   └─ features per historical start
                  │           └───── history sequence (max 8 prior starts, most recent first)
                  └─────────────────  all horse-race pairs (e.g. 17 877)

Static tensor:   [n_runners, 27]
                  │           └─ static features per horse
                  └─────────────  all horse-race pairs
```

The limitation: the model has no awareness of the other horses in the race when making a prediction — it scores each horse in isolation and cannot learn race-relative relationships ("this horse is the fastest *in this particular field*").

---

#### Target state: race-based model

The model receives all runners in a race **as a single input** and evaluates them simultaneously. This enables learning of within-race relationships between horses.

**New tensors:**

```
History tensor:  [n_races, MAX_RUNNERS, 8, 25]
                  │         │            │   └─ features per historical start
                  │         │            └───── history sequence (max 8)
                  │         └─────────────────  runners per race (padded to MAX_RUNNERS)
                  └───────────────────────────  all races (e.g. ~10 000)

Static tensor:   [n_races, MAX_RUNNERS, 27]
                  │         │            └─ static features per horse
                  │         └──────────────  runners per race (padded)
                  └────────────────────────  all races

Output:          [n_races, MAX_RUNNERS, 1]
                  │         │            └─ top-3 probability per horse
                  │         └──────────────  runners per race
                  └────────────────────────  all races
```

Missing runners (padding slots) are filled with `-1`, consistent with the existing history sequence padding convention.

---

#### Planned `buildModel` architecture

The architecture uses `TimeDistributed` layers to process each horse independently first (same as the current model, but now nested inside the race dimension), followed by **Multi-Head Attention** which learns relationships between runners within the race.

```javascript
function buildModel(maxRunners, timeSteps, histFeatures, staticFeatures) {
    // Inputs — race-based: outermost runner dimension groups horses per race
    const histInput   = tf.input({ shape: [maxRunners, timeSteps, histFeatures], name: 'history_input' });
    const staticInput = tf.input({ shape: [maxRunners, staticFeatures],          name: 'static_input'  });

    // ── History branch ────────────────────────────────────────────────────────
    // TimeDistributed runs the same LSTM stack independently for each runner.
    // Masking before the LSTM ignores -1-padded history slots.

    let h = tf.layers.timeDistributed({
        layer: tf.layers.masking({ maskValue: -1 }),
        name: 'hist_masking',
    }).apply(histInput);

    h = tf.layers.timeDistributed({
        layer: tf.layers.lstm({
            units: 64,
            returnSequences: true,
            recurrentDropout: 0.1,
            kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        }),
        name: 'hist_lstm1',
    }).apply(h);
    // Output: [n_races, maxRunners, timeSteps, 64]

    h = tf.layers.timeDistributed({
        layer: tf.layers.lstm({
            units: 32,
            returnSequences: false,
            recurrentDropout: 0.1,
            kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        }),
        name: 'hist_lstm2',
    }).apply(h);
    // Output: [n_races, maxRunners, 32]

    h = tf.layers.timeDistributed({
        layer: tf.layers.batchNormalization(),
        name: 'hist_bn',
    }).apply(h);

    h = tf.layers.timeDistributed({
        layer: tf.layers.dropout({ rate: 0.3 }),
        name: 'hist_dropout',
    }).apply(h);

    // ── Static branch ─────────────────────────────────────────────────────────
    // TimeDistributed Dense processes each runner's static features independently.

    let s = tf.layers.timeDistributed({
        layer: tf.layers.dense({
            units: 48,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        }),
        name: 'static_dense1',
    }).apply(staticInput);
    // Output: [n_races, maxRunners, 48]

    s = tf.layers.timeDistributed({
        layer: tf.layers.batchNormalization(),
        name: 'static_bn1',
    }).apply(s);

    s = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 32, activation: 'relu' }),
        name: 'static_dense2',
    }).apply(s);

    s = tf.layers.timeDistributed({
        layer: tf.layers.dropout({ rate: 0.3 }),
        name: 'static_dropout',
    }).apply(s);

    // ── Merge branches ────────────────────────────────────────────────────────
    // Concatenate history and static representations per runner.
    // Output: [n_races, maxRunners, 64]

    let combined = tf.layers.concatenate({ axis: -1, name: 'combine' }).apply([h, s]);

    combined = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 64, activation: 'relu' }),
        name: 'combined_dense',
    }).apply(combined);

    combined = tf.layers.timeDistributed({
        layer: tf.layers.batchNormalization(),
        name: 'combined_bn',
    }).apply(combined);
    // Output: [n_races, maxRunners, 64]

    // ── Multi-Head Attention ──────────────────────────────────────────────────
    // Self-attention over the runner dimension: the model learns which other
    // horses in the race are relevant when scoring each individual runner.
    // Query = Key = Value = combined  →  numHeads=4, keyDim=16 → 64-dim output

    const attended = tf.layers.multiHeadAttention({
        numHeads: 4,
        keyDim:   16,
        dropout:  0.1,
        name:     'runner_attention',
    }).apply([combined, combined]);
    // Output: [n_races, maxRunners, 64]

    // Residual connection + Layer Normalization (standard post-attention pattern)
    const residual = tf.layers.add({ name: 'attention_residual' }).apply([combined, attended]);
    const normed   = tf.layers.layerNormalization({ name: 'attention_ln' }).apply(residual);

    // ── Output head ───────────────────────────────────────────────────────────
    // Each runner gets its own top-3 probability via TimeDistributed sigmoid.

    let out = tf.layers.timeDistributed({
        layer: tf.layers.dropout({ rate: 0.25 }),
        name: 'out_dropout',
    }).apply(normed);

    out = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 24, activation: 'relu' }),
        name: 'out_dense',
    }).apply(out);

    out = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 1, activation: 'sigmoid' }),
        name: 'output',
    }).apply(out);
    // Output: [n_races, maxRunners, 1]

    const model = tf.model({ inputs: [histInput, staticInput], outputs: out });
    model.compile({
        optimizer: tf.train.adam(0.0003),
        loss:      'binaryCrossentropy',   // padding runners must be masked from loss
        metrics:   ['accuracy'],
    });
    return model;
}
```

#### Implementation notes

- **MAX_RUNNERS** — set to the largest field size in the dataset (typically 16). Padded runner slots are filled with zeros and must be excluded from loss computation using a separate runner mask tensor.
- **Loss masking** — padded runners must be masked out of the loss calculation, otherwise the model learns to predict zeros for padding slots. In TensorFlow.js this can be done via a `sample_weight` tensor or a custom loss function.
- **`multiHeadAttention`** — available in TensorFlow.js from version 4.x onwards. Verify the installed version before implementing.
- **Memory requirements** — a 4D tensor of shape `[10000, 16, 8, 25]` contains ~128M float32 values (~512 MB). Reduce batch size (e.g. 16–32) to keep memory pressure manageable during training.

---

## Live Version

A working deployment is available here:  
👉 [toto prediction system](https://toto-kesm.onrender.com)
