### Toteutetut muutokset (Mixed‑malli)
- BCE label smoothing:
    - Laajensin `bceLoss(yTrue, yPred, labelSmoothing)` ja johdin sen läpi yhdistettyyn loss:iin.
    - Uusi optio: `labelSmoothing` (oletus 0.0, turvaraja 0.2).
- Aux‑lossin annealaus (0.6 → 0.3 tms.):
    - Lisäsin dynaamisen lossin (`dynamicCombinedRaceLoss`) ja globaalit säätimet `__auxWeight` ja `__labelSmoothing`.
    - Treeniloopin `onEpochBegin` säätää aux‑painoa `options.auxSchedule` mukaan (lineaarinen rampitus `startEpoch` → `endEpoch`).
- Pieni pre‑enkooderi historiahaaralle ennen LSTM:ää:
    - Uusi optio: `preEncodeHistory: { units: 16, useLayerNorm: true }`.
    - Toteutettu `TimeDistributed(Dense(units, relu))` + valinnainen `TimeDistributed(LayerNormalization)` ennen `hist_lstm1`.
- Loss‑valinta ja koonti:
    - Jos `useListNet: true` → loss = `topKSoftAuxLoss` (puhtaan listanetiin), muuten → dynaaminen yhdistelmä (BCE + aux SoftTopK).
    - Label smoothing vaikuttaa vain BCE‑osaan.
- Yhteensopivuus säilyy:
    - Vanhoilla asetuksilla käytös ennallaan (ellei uusia optioita anneta).

### Mistä löydät muutokset
- `model-training/model_mixed_runner_and_race.js`:
    - BCE smoothing: rivit ~90–110.
    - Dynamic loss + säätimet: rivit ~92–106, 137–155.
    - Pre‑encoder ennen LSTM:ää: rivit ~658–667.
    - Lossin valinta `dynamicCombinedRaceLoss()`: rivit ~745–751.
    - Annealaus callbackissa: rivit ~898–918.
    - `buildMixedModel` tukee `preEncodeHistory`a; `runTraining` plumbattu (`labelSmoothing`, `auxSchedule`, `preEncodeHistory`).

### Valmiit ajokäskyt (kopioi & aja)
Huom: käytä `-- --backend=gpu` GPU:lle tai `-- --backend=cpu` CPU:lle. `embedDim` ja `attnHeads` sovitetaan automaattisesti yhteen.

#### 1) Baseline + label smoothing (s=0.05)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-s005', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 10, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, swapBNtoLN: true, useLayerNormInStatic: true, attnHeads: 12, embedDim: 116, ffnDim: 230, runnerProjDim: 64, runnerLstm2Units: 48, dropout: 0.20, outDropout: 0.30, l2: 1e-4, labelSmoothing: 0.05 })" -- --backend=cpu
```

#### 2) Korkeampi aux SoftTopK paino (0.6)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-aux06', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 10, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.6, useListNet: false, swapBNtoLN: true, useLayerNormInStatic: true, attnHeads: 12, embedDim: 116, ffnDim: 230, runnerProjDim: 64, runnerLstm2Units: 48, dropout: 0.20, outDropout: 0.30, l2: 1e-4, labelSmoothing: 0.05 })" -- --backend=cpu
```

#### 3) Aux‑annealaus 0.6 → 0.3 epokin 40 jälkeen (lineaarinen 40–70)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-aux06to03-e40-70', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 10, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.6, auxSchedule: { from: 0.6, to: 0.3, startEpoch: 40, endEpoch: 70 }, useListNet: false, swapBNtoLN: true, useLayerNormInStatic: true, attnHeads: 12, embedDim: 116, ffnDim: 230, runnerProjDim: 64, runnerLstm2Units: 48, dropout: 0.20, outDropout: 0.30, l2: 1e-4, labelSmoothing: 0.05 })" -- --backend=cpu
```

#### 4) Pre‑enkooderi ennen LSTM:ää (TD Dense16 + LN)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-preenc16', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 10, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, swapBNtoLN: true, useLayerNormInStatic: true, preEncodeHistory: { units: 16, useLayerNorm: true }, attnHeads: 12, embedDim: 116, ffnDim: 230, runnerProjDim: 64, runnerLstm2Units: 48, dropout: 0.20, outDropout: 0.30, l2: 1e-4, labelSmoothing: 0.05 })" -- --backend=cpu
```

#### 5) Pre‑enkooderi + annealaus (0.6→0.3 @40–70)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-preenc16-auxanneal', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 10, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.6, auxSchedule: { from: 0.6, to: 0.3, startEpoch: 40, endEpoch: 70 }, useListNet: false, swapBNtoLN: true, useLayerNormInStatic: true, preEncodeHistory: { units: 16, useLayerNorm: true }, attnHeads: 12, embedDim: 116, ffnDim: 230, runnerProjDim: 64, runnerLstm2Units: 48, dropout: 0.20, outDropout: 0.30, l2: 1e-4, labelSmoothing: 0.05 })" -- --backend=cpu
```

#### 6) ListNet‑versio (ei BCE:tä; smoothing ei vaikuta)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-listnet', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 10, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', useListNet: true, auxLossWeight: 0.0, swapBNtoLN: true, useLayerNormInStatic: true, attnHeads: 12, embedDim: 116, ffnDim: 230, runnerProjDim: 64, runnerLstm2Units: 48, dropout: 0.20, outDropout: 0.30, l2: 1e-4 })" -- --backend=cpu
```

#### 7) GPU‑suositus (sama kuin 5), mutta GPU:lle)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-preenc16-auxanneal-gpu', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 10, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.6, auxSchedule: { from: 0.6, to: 0.3, startEpoch: 40, endEpoch: 70 }, useListNet: false, swapBNtoLN: true, useLayerNormInStatic: true, preEncodeHistory: { units: 16, useLayerNorm: true }, attnHeads: 12, embedDim: 116, ffnDim: 230, runnerProjDim: 64, runnerLstm2Units: 48, dropout: 0.20, outDropout: 0.30, l2: 1e-4, labelSmoothing: 0.05 })" -- --backend=gpu
```

### Huomioita
- `labelSmoothing` vaikuttaa vain BCE‑osaan. ListNet‑ajossa se ohitetaan.
- `auxSchedule` käyttää lineaarista rampitusta välillä `[startEpoch, endEpoch]`. Jätä pois, jos et halua annealausta.
- Pre‑enkooderi auttaa oppimaan `kmTime×distance/isCarStart` ‑vuorovaikutuksia ilman käsin määriteltyjä kaavoja.
- `embedDim` ja `attnHeads`: tiedosto korjaa pienen epäsuhdan automaattisesti (varoituslogi), jotta MHA toimii.

Palaa asiaan, jos haluat saman paketin myös Race‑malliin; voin peilata muutokset sinne 1:1 ja kirjoittaa vastaavat komentoesimerkit