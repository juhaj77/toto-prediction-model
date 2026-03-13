### Hieno havainto: korkea LR (1e-3) toimii — näin jatkat kehitystä
Kun `learningRate=1e-3` tuottaa parhaan ndcg@3:n, se viittaa siihen että optimointi hyötyy rohkeammasta askelluksesta. Alla priorisoitu jatkosarja, joka nojaa tuohon asetukseen ja kokeilee turvallisia vakautuskeinoja (regularisaatio, LayerNorm), sekä kapasiteettia. Olen lisännyt jokaiselle kohteelle valmiin komennon. Aja komennot kansiosta `model-training`. Jos haluat määrittää backendin, lisää loppuun ` -- --backend=gpu` tai ` -- --backend=cpu`.

Huom: Jos ajat `node -e`, muista kaksoisviiva `--` ennen backend‑lippua.

---

#### 1) LR=1e-3 + vahvempi regularisaatio (vakauttaa, usein parantaa top‑k rankingia)
Muutokset vs. sinun paras: `l2: 2e-4` ja lievästi suurempi dropout.
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-reg', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.20, outDropout: 0.30, l2: 2e-4 })" -- --backend=gpu
```
Miksi: korkea LR + hieman isompi painonhävikkö ja dropout hillitsevät ylireagointia ja usein tuovat parempaa ndcg@3:aa.

#### 2) LR=1e-3 + LayerNorm‑painotus (robustimpi pienillä/epästabiileilla batch‑ko’oilla)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-ln', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, swapBNtoLN: true, useLayerNormInStatic: true, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" -- --backend=gpu
```
Miksi: LN on usein vakaampi korkean LR:n kanssa ja auttaa optimoitumaan tasaisemmin.

#### 3) LR=1e-3 + pidempi warmup (10% epokkeja) — pehmeämpi startti
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-wu10', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 9, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" -- --backend=gpu
```
Miksi: jos alku on kohiseva, pidempi warmup auttaa pääsemään parempaan uraan.

#### 4) LR=1e-3 + pienempi batch (enemmän päivityskohinaa → parempi yleistys)
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-b320', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 320, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" -- --backend=gpu
```
Miksi: hieman pienempi batch voi parantaa ranking‑mittareita vaikka val_loss nousisi.

#### 5) LR=1e-3 + kapasiteettinosto maltillisesti (embed 120) — heads 12 pysyy validina
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-cap120', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 120, ffnDim: 240, runnerProjDim: 64, runnerLstm2Units: 64, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" -- --backend=gpu
```
Miksi: hieman suurempi representaatio voi nostaa ndcg@3/Hit@1 ilman että muisti räjähtää.

#### 6) LR=1e-3 + ListNet‑painotus (puhdas listnet tai hybrid)
- Puhdas ListNet:
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-listnet', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', useListNet: true, auxLossWeight: 0.0, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" -- --backend=gpu
```
- Hybridi (BCE + SoftTopK w=0.6):
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-hybrid06', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', useListNet: false, auxLossWeight: 0.6, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" -- --backend=gpu
```
Miksi: korkea LR toimii usein hyvin yhdessä eksplisiittisen listalossin kanssa.

#### 7) LR=1e-3 + pidempi treeni (120 ep) — anna kosinelle aikaa konvergoida
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-long120', epochs: 120, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 8, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 448, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4, earlyStopPatience: 12 })" -- --backend=gpu
```
Miksi: aiemmat käyrät viittasivat hitaaseen paranemiseen; 120 ep + korkea LR voi tuoda vielä pari sadasosaa ndcg@3:een.

---

### Muita kehitysideoita (voit pyytää niin toteutan koodiin)
- Gradient clipping: globaali normi 1.0–1.5 stabiloi korkean LR:n (toteutetaan treenikierrossa ennen optimizer.step). Ei vielä projektissa — voin lisätä.
- Aux‑lossin annealaus: aloita 0.6 → laske 0.3:een epokin 40 jälkeen (helpottaa loppukonvergenssia). Tarvitsee callback‑tuen — voin tehdä optionin `auxSchedule`.
- Label smoothing BCE: pieni s=0.05 auttaa kalibrointia ilman listnettiä.
- Weight Averaging (EMA/SWA): pitkillä ajoilla vakauttaa val_ndcg3:ea.
- Kapasiteetti 128/16 (jos muisti sallii):
```
node -e "require('./model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-cap128h16', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 352, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 16, embedDim: 128, ffnDim: 256, runnerProjDim: 64, runnerLstm2Units: 64, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })" -- --backend=gpu
```

### Seuranta ja valinta
- Pidä päämittarina `val_ndcg3`. Katso myös `val_hit1` ja `val_ap_macro` — korkea LR + lisäregularisaatio usein nostaa juuri näitä.
- Hyödynnä `tools/summarize_runs.js` järjestääksesi ajot ndcg@3:n mukaan; saat samalla valmiit komentojen rekonstruoinnit.

Haluatko, että toteutan seuraavaksi gradient clippingin ja aux‑annealoinnin (takautuvasti taaksepäin yhteensopivina optioina), jotta voit ajaa yllä olevat kokeet vielä turvallisemmin korkealla oppimisnopeudella?