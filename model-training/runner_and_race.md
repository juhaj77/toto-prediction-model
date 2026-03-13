### Suunnitelma: tee ensin samat CLI‑parannukset myös race/runner ‑malleihin
Jotta voit varioida parametreja yhtä joustavasti kuin mixed‑mallissa, suosittelen tekemään seuraavat muutokset tiedostoihin `model-training/model_race.js` ja `model-training/model_runner.js` (nykyarvot jäävät oletuksiksi):
- Lisää dynaaminen TFJS‑backendin valinta (cpu/gpu) samaan tapaan kuin mixedissä. 
- Siirrä `buildModel(…, options)` ja `runTraining(opts)` käyttämään option‑objektia, jossa on:
  - Optimointi: `learningRate`, `scheduler` ('cosine'|'plateau'), `warmupEpochs`, `minLearningRate`, `plateauPatience`, `plateauFactor`, `earlyStopPatience`.
  - Arkkitehtuuri: `attnHeads`, `embedDim`, `ffnDim`, `dropout`, `outDropout`, `runnerLstm2Units` (runner/mixed), `runnerProjDim` (runner/mixed), `useLayerNormInStatic`, `swapBNtoLN`.
  - Lossit: `auxLossWeight`, `useListNet` (race: kyllä, runner: ei tarvetta oletuksena).
  - Treeni: `epochs`, `batchSize`, `valFraction`, `temporalSplit`, `bestBy`.
- Talleta `trainingInfo.run_options` kaikkiin tallennettaviin JSON‑tiedostoihin sekä `tfjs_backend`.
- Lisää `autoFixHeads` (oletus true) varmistamaan, että `embedDim % attnHeads == 0` (race‑mallin MHA tarvitsee tämän samalla tavalla).

Voin toteuttaa nämä muokkaukset puolestasi, jos haluat. Alla sillä välin valmiit testikomennot — toimivat heti, jos `runTraining(opts)` ja optionit ovat käytettävissä race/runner‑tiedostoissa kuten mixedissä. Jos ei ole, toteutan päivityksen pyynnöstäsi.

—

### Testiparametrilistat: model_race (aja kansiosta model-training)
Päämittari: `val_ndcg3` (sekundäärinä `val_hit1`, `val_ap_macro`, `val_auc`). Backendin määrittely: lisää loppuun ` -- --backend=gpu` tai ` -- --backend=cpu`.

1) Baseline (nykyisiä arvoja vastaava, vertailukohta)
```
node -e "require('./model_race').runTraining({ runId: 'race-baseline-ndcg3', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, dropout: 0.15, outDropout: 0.25, l2: 1e-4, swapBNtoLN: false, useLayerNormInStatic: false })" -- --backend=gpu
```

2) LayerNorm‑painotus (sinulla tämä toimi mixedissä parhaiten)
```
node -e "require('./model_race').runTraining({ runId: 'race-ln', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, dropout: 0.15, outDropout: 0.25, l2: 1e-4, swapBNtoLN: true, useLayerNormInStatic: true })" -- --backend=gpu
```
2B)
```
node -e "require('./model_race').runTraining({ runId: 'race-ln2', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 8, learningRate: 2e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_auc', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 230, dropout: 0.2, outDropout: 0.3, l2: 1e-4, swapBNtoLN: true, useLayerNormInStatic: true })" -- --backend=cpu
```

3) Korkea LR 1e‑3 (kuten paras mixedissä) + regulaatioa
```
node -e "require('./model_race').runTraining({ runId: 'race-lr1e-3-reg', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, dropout: 0.20, outDropout: 0.30, l2: 2e-4 })" -- --backend=gpu
```

4) Puhdas ListNet (usein nostaa ranking‑mittareita)
```
node -e "require('./model_race').runTraining({ runId: 'race-listnet', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', useListNet: true, auxLossWeight: 0.0, attnHeads: 12, embedDim: 96, ffnDim: 192, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" -- --backend=gpu
```

5) Pidempi treeni + pidempi warmup
```
node -e "require('./model_race').runTraining({ runId: 'race-long120', epochs: 120, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 8, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 448, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, dropout: 0.15, outDropout: 0.25, l2: 1e-4, earlyStopPatience: 12 })" -- --backend=gpu
```

6) Batch‑koon ablaatio (enemmän kohinaa, usein parempi ndcg)
```
node -e "require('./model_race').runTraining({ runId: 'race-batch320', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 320, bestBy: 'val_ndcg3', auxLossWeight: 0.5 })" -- --backend=gpu
```

7) Kapasiteettinosto (embed 120, heads 12 kelvollinen)
```
node -e "require('./model_race').runTraining({ runId: 'race-cap120', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 120, ffnDim: 240, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" -- --backend=gpu
```

—

### Testiparametrilistat: model_runner (aja kansiosta model-training)
Päämittarit: `val_auc`, `val_ap` (sekä halutessa `val_ndcg3`, `val_hit1` jos implementoitu runnerille). Runner‑mallissa ei ole per‑kisa attentionia, joten ranking‑aux ei yleensä ole käytössä; keskity optimointiin ja kapasiteettiin.

1) Baseline (vertailukohta)
```
node -e "require('./model_runner').runTraining({ runId: 'runner-baseline-auc', epochs: 80, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 512, bestBy: 'val_auc', dropout: 0.2, l2: 1e-4 })" -- --backend=gpu
```

2) LayerNorm‑painotus (jos BN/LN‑vaihto on tuettu kuten mixedissä)
```
node -e "require('./model_runner').runTraining({ runId: 'runner-ln', epochs: 80, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 768, bestBy: 'val_auc', dropout: 0.15, l2: 1e-4, swapBNtoLN: true, useLayerNormInStatic: true })" -- --backend=gpu
```

3) Korkea LR 1e‑3 + regulaatio
```
node -e "require('./model_runner').runTraining({ runId: 'runner-lr1e-3-reg', epochs: 80, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 768, bestBy: 'val_auc', dropout: 0.25, l2: 2e-4 })" -- --backend=gpu
```

4) Pienempi batch (enemmän kohinaa → usein parempi AUC/AP)
```
node -e "require('./model_runner').runTraining({ runId: 'runner-batch384', epochs: 80, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 384, bestBy: 'val_auc', dropout: 0.2, l2: 1e-4 })" -- --backend=gpu
```

5) Kapasiteettinosto (historia‑LSTM 64→96, projisoitu 32→48)
```
node -e "require('./model_runner').runTraining({ runId: 'runner-cap', epochs: 80, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 640, bestBy: 'val_auc', runnerLstm2Units: 64, runnerProjDim: 48, dropout: 0.2, l2: 1e-4 })" -- --backend=gpu
```

6) LR‑sweep matalampi (jos sahausta/epävakautta)
```
node -e "require('./model_runner').runTraining({ runId: 'runner-lr2e-4', epochs: 80, scheduler: 'cosine', warmupEpochs: 4, learningRate: 2e-4, minLearningRate: 3e-5, batchSize: 512, bestBy: 'val_auc', dropout: 0.2, l2: 1e-4 })" -- --backend=gpu
```

7) Pitkä ajo + EarlyStop (jos paranee hitaasti)
```
node -e "require('./model_runner').runTraining({ runId: 'runner-long120', epochs: 120, scheduler: 'cosine', warmupEpochs: 8, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 640, bestBy: 'val_auc', dropout: 0.2, l2: 1e-4, earlyStopPatience: 12 })" -- --backend=gpu
```

—

### Huomioita ja vinkkejä
- Jos `model_race.js` käyttää vielä suoraa `require('@tensorflow/tfjs-node-gpu')`, CPU‑ajo (`--backend=cpu`) ei toimi ennen kuin lisätään dynaaminen backend‑lataus.
- Race‑mallissa varmista, että maski/pehmuste ei päädy metriikoihin (sinulla on jo per‑race eval — hyvä). Runner‑mallissa päämittarina AUC/AP, koska luokkajakauma on epätasapainoinen.
- Voit yhdistellä LN + korkea LR (+reg) ‑reseptin, koska se toimi hyvin mixedissä.

Haluatko, että teen koodimuutokset race/runner ‑tiedostoihin (CLI‑backend, `runTraining(opts)`, LN‑vivut, autoFixHeads, optioiden talletus `trainingInfo.run_options`) samaan tyyliin kuin mixedissä? Teen ne niin, että nykyiset arvot jäävät oletuksiksi ja vanhat komennot toimivat edelleen.