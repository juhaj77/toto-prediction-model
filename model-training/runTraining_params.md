### Suunnitelma kokeista (priorisoitu)
Alla on selkeä, käytännöllinen koeputki mixed‑mallille (Y=1 jos top‑3), jossa etenet halvasta → kalliimpaan ja pidät jokaisessa kokeessa saman `runId`‑nimityskäytännön. Kunkin kohdan alla on valmis ajokomento. Suosittelen ajamaan 1–3 rinnakkain (eri prosesseissa), mutta aina eri `runId`:llä.

- Peruslähtökohta: temporal split päällä, cosine‑scheduler + warmup, bestBy=val_ndcg3, pidempi treeni. Ajele 90–120 epochia, patience ~12.
- Tärkein vipu top‑3‑tehtävälle: ranking‑apuhäviön (soft Top‑K) paino tai vaihtoehtoisesti puhdas listnet (useListNet=true). 
- Seuraavaksi: kapasiteetti (embedDim/ffnDim/attnHeads, runnerLstm2Units) ja pienet optimointisäädöt (lr, batchSize, l2, dropout). 

Huom: alla olevat komennot on tarkoitettu ajettaviksi projektin juuresta. Käytän `require('./model-training/model_mixed_runner_and_race')` jotta polku on varmasti oikein.

---

### Korkean prioriteetin kokeet (1 = tärkein)
1) Baseline (päivitetty) — vertailukohta ndcg@3:lle
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-baseline-ndcg3', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: kiinteä referenssi ennen suurempia muutoksia.

2) Nosta ranking‑apuhäviön painoa — usein halpa voitto top‑3:een
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-aux06', epochs: 90, auxLossWeight: 0.6, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: tyypillisesti +0.02…+0.05 ndcg@3.

3) Puhdas listnet/soft top‑k — joskus paras rankingille
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-listnet', epochs: 90, useListNet: true, auxLossWeight: 0.0, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: eliminoi BCE:n mahdollisen ristiriidan per‑kisa listatavoitteen kanssa.

4) Pidempi treeni + pidempi warmup — anna schedulerille aikaa
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-long120', epochs: 120, warmupEpochs: 6, temporalSplit: true, scheduler: 'cosine', learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: aiemmat käyrät viittasivat hitaaseen paranemiseen.

5) Kapasiteettinosto (attention/embedit) — jos dataa riittää
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-cap128', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 352, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 128, ffnDim: 256, runnerProjDim: 64, runnerLstm2Units: 64, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: lisää esityskykyä rankingille; pienensin batchia hieman muistin takia.

6) Kevyempi/regularisoidumpi malli — jos ylireg/valossa hyödyt
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-compact64', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 448, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 8, embedDim: 64, ffnDim: 128, runnerProjDim: 32, runnerLstm2Units: 32, dropout: 0.2, outDropout: 0.3, l2: 2e-4 })"
```
Miksi: helpottaa optimointia jos datamäärä/featuret eivät tue isoa kapasiteettia.

7) LR‑sweep korkeampi — nopeampi eteneminen jos stagnaatio
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 6, learningRate: 1e-3, minLearningRate: 1e-4, batchSize: 512, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: jos grad‑normit jäävät isoiksi eikä loss putoa tarpeeksi.

8) LR‑sweep matalampi — vakaampi, jos kohina/epävakaus
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1p5e-4', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 1.5e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: jos validi‑mittarit sahaavat ja paras malli osuu varhain.

9) LayerNorm painotus — joskus parempi pienehköillä batcheilla
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-ln', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, swapBNtoLN: true, useLayerNormInStatic: true, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: LN on robustimpi pienillä/muuttuvilla batch‑ko’oilla.

10) Batch‑koon ablaatio — voi vaikuttaa sekä optimointiin että regularisaatioon
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-batch256', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 256, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: suurempi päivitys‑kohina → usein parempi yleistys.

11) Runner‑historian vahvistus — jos historia on tärkein signaali
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-runner-enc-strong', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 352, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 64, runnerLstm2Units: 64, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: kasvattaa juoksija‑upotteen laatua ennen kisa‑attentionia.

12) Scheduler‑ablaatio: Plateau vs. Cosine
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-plateau', epochs: 90, temporalSplit: true, scheduler: 'plateau', plateauPatience: 6, plateauFactor: 0.5, learningRate: 3e-4, minLearningRate: 1e-5, warmupEpochs: 0, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5, useListNet: false, attnHeads: 12, embedDim: 96, ffnDim: 192, runnerProjDim: 48, runnerLstm2Units: 48, dropout: 0.15, outDropout: 0.25, l2: 1e-4 })"
```
Miksi: joissain datasarjoissa plateau laskee lr:n “oikeaan aikaan”.

---

### Keskitaso/ablaatiot (ajo vain jos haluat vahvistaa valintoja)
13) Validaatiojako 15% — vakaammat mittarit, hitaampi treeni
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'abl-val15', epochs: 90, valFraction: 0.15, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5 })"
```

14) Ilman temporal split ‑ablaatiossa (vuotoriski!)
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'abl-no-temporal', epochs: 90, temporalSplit: false, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_ndcg3', auxLossWeight: 0.5 })"
```
Tarkoitus: varmistaa että temporal split ei yksinään paina metriikoita alas.

15) bestBy‑metriikan vaihto AUC:iin (pelkkä ablaatio)
```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'abl-bestby-auc', epochs: 90, temporalSplit: true, scheduler: 'cosine', warmupEpochs: 4, learningRate: 3e-4, minLearningRate: 3e-5, batchSize: 384, bestBy: 'val_auc', auxLossWeight: 0.5 })"
```

---

### Käyttövinkit
- Aja komennot projektin juuresta (polku `./model-training/...`). 
- Kaikki ajot menevät nyt omaan kansioon `model-training/model-mixed/runs/<runId>/` — valitse kuvaava `runId` aina. 
- Seuraa ensisijaisesti `val_ndcg3`, `val_hit1`, `val_ap_macro`; sekundäärinä `val_auc`. 
- Jos GPU‑muisti loppuu kapasiteettinostoissa: pienennä batchSize: 384→320→288. 
- Jos treeni on liian hidas mutta metriikat hyvät: lyhennä `epochs` 90→70 ja nosta `warmupEpochs` 4→5.

Haluatko, että teen sinulle pienen .cmd/.sh “kokeiluajurin”, joka ajaa nämä sarjana ja kerää parhaat ndcg@3‑tulokset yhteen taulukkoon? Se helpottaa tulosten vertailua jälkeenpäin.