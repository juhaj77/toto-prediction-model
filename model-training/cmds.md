### Päivitetyt “node -e” ‑komennot (aja kansiosta model-training)

- Aja nämä komennot hakemistosta `model-training`.
- Backendin vaihto per ajo: komennon loppuun `--backend=gpu` → vaihda `gpu` → `cpu` kun haluat CPU‑ajon.
- Polku on päivitetty muotoon `require('./model_mixed_runner_and_race')` (koska ajat `model-training`‑kansiosta).
- Huom: kapasiteettinostossa (`exp-cap128`) päivitin `attnHeads: 16`, jotta `embedDim: 128` jakautuu tasan päiden kesken (välttää auto‑korjauksen).

#### 1) Baseline (päivitetty) — vertailukohta ndcg@3:lle
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-baseline-ndcg3', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

#### 2) Nosta ranking‑apuhäviön painoa — usein halpa voitto top‑3:een
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-aux06', epochs: 90, auxLossWeight: 0.6, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

#### 3) Puhdas listnet/soft top‑k — joskus paras rankingille
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-listnet', epochs: 90, useListNet: true, auxLossWeight: 0.0, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

#### 4) Pidempi treeni + pidempi warmup — anna schedulerille aikaa
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-long120', epochs: 120, warmupEpochs: 6, temporalSplit: true, scheduler: 'cosine', learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

#### 5) Kapasiteettinosto (attention/embedit) — jos dataa riittää
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-cap128', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 352, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 16, embedDim: 128, ffnDim: 256, runnerProjDim: 64, runnerLstm2Units: 64, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

#### 6) Kevyempi/regularisoidumpi malli — jos ylireg/valossa hyödyt
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-compact64', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 448, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 8, embedDim: 64, ffnDim: 128, runnerProjDim: 32, runnerLstm2Units: 32, dropout: 0.2, outDropout: 0.3, l2: 2e-4 })" --backend=gpu
```

#### 7) LR‑sweep korkeampi — nopeampi eteneminen jos stagnaatio
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

#### 8) LR‑sweep matalampi — vakaampi, jos kohina/epävakaus
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1p5e-4', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 1.5e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

#### 9) LayerNorm painotus — joskus parempi pienehköillä batcheilla
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-ln', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, swapBNtoLN: true, useLayerNormInStatic: true, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

#### 10) Batch‑koon ablaatio — voi vaikuttaa sekä optimointiin että regularisaatioon
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-batch256', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 256, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

#### 11) Runner‑historian vahvistus — jos historia on tärkein signaali
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-runner-enc-strong', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 352, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 64, runnerLstm2Units: 64, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

#### 12) Scheduler‑ablaatio: Plateau vs. Cosine
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-plateau', epochs: 90, temporalSplit: true, scheduler: 'plateau', plateauPatience: 6, plateauFactor: 0.5, learningRate: 3e-4, minLearningRate: 1e-5, warmupEpochs: 0, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" --backend=gpu
```

### Keskitaso/ablaatiot

#### 13) Validaatiojako 15% — vakaammat mittarit, hitaampi treeni
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'abl-val15', epochs: 90, valFraction: 0.15, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5 })" --backend=gpu
```

#### 14) Ilman temporal split ‑ablaatiossa (vuotoriski!)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'abl-no-temporal', epochs: 90, temporalSplit: false, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5 })" --backend=gpu
```

#### 15) bestBy‑metriikan vaihto AUC:iin (pelkkä ablaatio)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'abl-bestby-auc', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_auc', auxLossWeight: 0.5 })" --backend=gpu
```

### Vinkit ajamiseen
- Voit ajaa kahta instanssia rinnakkain ilman ylikuormaa: toinen `--backend=gpu`, toinen `--backend=cpu`.
- Jos `@tensorflow/tfjs-node` (CPU) ei ole asennettu, asenna `model-training`‑kansiossa: `npm i @tensorflow/tfjs-node@^4.22.0`.
- Jos GPU‑muisti loppuu, pienennä `batchSize` tai aja CPU:lla.

Haluatko, että teen lisäksi pienen `.cmd`‑tiedoston, joka sisältää nämä komennot valmiina ja kysyy sinulta vain backendin (cpu/gpu)?