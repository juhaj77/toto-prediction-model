### Mixed-model runs — summary

- Scanned: \C:/Users/juhaj/WebstormProjects/toto-prediction-model/model-training/model-mixed/runs
- Sorted by: `val_ndcg3` (top 50)

| # | runId | file | epoch | val_ndcg@3 | val_hit@1 | val_r@3 | val_p@3 | val_auc | backend |
|:-:|:------|:-----|------:|-----------:|----------:|--------:|--------:|-------:|:-------|
| 1 | exp-lr1e-3-ln | model_best_ndcg3.json | 40 | 0.5398 | 0.5696 | 0.5302 | 0.5302 | 0.7454 | tfjs-node |
| 2 | exp-ln-s005 | model_best_ndcg3.json | 41 | 0.5388 | 0.5833 | 0.5231 | 0.5198 | 0.7487 | tfjs-node |
| 3 | exp-ln-preenc16 | model_best_ndcg3.json | 37 | 0.5344 | 0.5696 | 0.5218 | 0.5218 | 0.7425 | tfjs-node |
| 4 | exp-listnet | model_best_ndcg3.json | 42 | 0.5343 | 0.5738 | 0.5246 | 0.5246 | 0.7254 | tfjs-node |
| 5 | exp-lr1e-3-ln-mod | model_best_ndcg3.json | 24 | 0.5309 | 0.6034 | 0.5077 | 0.5077 | 0.7132 | tfjs-node-gpu |
| 6 | exp-ln-aux06 | model_best_ndcg3.json | 48 | 0.5295 | 0.5738 | 0.5134 | 0.5134 | 0.7433 | tfjs-node |
| 7 | exp-lr1e-3-reg | model_best_ndcg3.json | 20 | 0.5241 | 0.6076 | 0.5007 | 0.5007 | 0.7103 | tfjs-node-gpu |
| 8 | exp-ln-preenc16-auxanneal-gpu | model_best_ndcg3.json | 47 | 0.5236 | 0.5738 | 0.5091 | 0.5091 | 0.7332 | tfjs-node-gpu |

---

#### exp-lr1e-3-ln

- Folder: `C:/Users/juhaj/WebstormProjects/toto-prediction-model/model-training/model-mixed/runs/exp-lr1e-3-ln`
- Source file: `model_best_ndcg3.json`
- Saved at: `2026-03-09T22:31:08.937Z`
- Metric snapshot: ndcg@3=0.5398, hit@1=0.5696, r@3=0.5302, p@3=0.5302, auc=0.7454, val_loss=1.3695
- Data: races=2379, runners=24734, 2026-02-01 → 2026-03-07
- Backend: tfjs-node (# recorded with CPU backend: use --backend=cpu or set $env:TFJS_BACKEND='cpu')

Ready-to-copy command (programmatic):

```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-ln',  })"
```

PowerShell (with backend env override):

```powershell
$env:TFJS_BACKEND='cpu'; node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-ln',  })"
```

#### exp-ln-s005

- Folder: `C:/Users/juhaj/WebstormProjects/toto-prediction-model/model-training/model-mixed/runs/exp-ln-s005`
- Source file: `model_best_ndcg3.json`
- Saved at: `2026-03-10T08:33:22.984Z`
- Metric snapshot: ndcg@3=0.5388, hit@1=0.5833, r@3=0.5231, p@3=0.5198, auc=0.7487, val_loss=1.5356
- Data: races=2520, runners=26178, 2026-02-01 → 2026-03-09
- Backend: tfjs-node (# recorded with CPU backend: use --backend=cpu or set $env:TFJS_BACKEND='cpu')

Ready-to-copy command (programmatic):

```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-s005',  })"
```

PowerShell (with backend env override):

```powershell
$env:TFJS_BACKEND='cpu'; node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-s005',  })"
```

#### exp-ln-preenc16

- Folder: `C:/Users/juhaj/WebstormProjects/toto-prediction-model/model-training/model-mixed/runs/exp-ln-preenc16`
- Source file: `model_best_ndcg3.json`
- Saved at: `2026-03-10T03:19:51.003Z`
- Metric snapshot: ndcg@3=0.5344, hit@1=0.5696, r@3=0.5218, p@3=0.5218, auc=0.7425, val_loss=1.5516
- Data: races=2379, runners=24734, 2026-02-01 → 2026-03-07
- Backend: tfjs-node (# recorded with CPU backend: use --backend=cpu or set $env:TFJS_BACKEND='cpu')

Ready-to-copy command (programmatic):

```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-preenc16',  })"
```

PowerShell (with backend env override):

```powershell
$env:TFJS_BACKEND='cpu'; node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-preenc16',  })"
```

#### exp-listnet

- Folder: `C:/Users/juhaj/WebstormProjects/toto-prediction-model/model-training/model-mixed/runs/exp-listnet`
- Source file: `model_best_ndcg3.json`
- Saved at: `2026-03-10T04:17:28.498Z`
- Metric snapshot: ndcg@3=0.5343, hit@1=0.5738, r@3=0.5246, p@3=0.5246, auc=0.7254, val_loss=2.1402
- Data: races=2379, runners=24734, 2026-02-01 → 2026-03-07
- Backend: tfjs-node (# recorded with CPU backend: use --backend=cpu or set $env:TFJS_BACKEND='cpu')

Ready-to-copy command (programmatic):

```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-listnet',  })"
```

PowerShell (with backend env override):

```powershell
$env:TFJS_BACKEND='cpu'; node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-listnet',  })"
```

#### exp-lr1e-3-ln-mod

- Folder: `C:/Users/juhaj/WebstormProjects/toto-prediction-model/model-training/model-mixed/runs/exp-lr1e-3-ln-mod`
- Source file: `model_best_ndcg3.json`
- Saved at: `2026-03-08T17:34:56.932Z`
- Metric snapshot: ndcg@3=0.5309, hit@1=0.6034, r@3=0.5077, p@3=0.5077, auc=0.7132, val_loss=1.3946
- Data: races=2379, runners=24734, 2026-02-01 → 2026-03-07
- Backend: tfjs-node-gpu (# runs with GPU backend (default if installed): add --backend=gpu when using CLI entrypoint)

Ready-to-copy command (programmatic):

```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-ln-mod',  })"
```

PowerShell (with backend env override):

```powershell
$env:TFJS_BACKEND='gpu'; node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-ln-mod',  })"
```

#### exp-ln-aux06

- Folder: `C:/Users/juhaj/WebstormProjects/toto-prediction-model/model-training/model-mixed/runs/exp-ln-aux06`
- Source file: `model_best_ndcg3.json`
- Saved at: `2026-03-10T01:57:15.525Z`
- Metric snapshot: ndcg@3=0.5295, hit@1=0.5738, r@3=0.5134, p@3=0.5134, auc=0.7433, val_loss=1.7614
- Data: races=2379, runners=24734, 2026-02-01 → 2026-03-07
- Backend: tfjs-node (# recorded with CPU backend: use --backend=cpu or set $env:TFJS_BACKEND='cpu')

Ready-to-copy command (programmatic):

```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-aux06',  })"
```

PowerShell (with backend env override):

```powershell
$env:TFJS_BACKEND='cpu'; node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-aux06',  })"
```

#### exp-lr1e-3-reg

- Folder: `C:/Users/juhaj/WebstormProjects/toto-prediction-model/model-training/model-mixed/runs/exp-lr1e-3-reg`
- Source file: `model_best_ndcg3.json`
- Saved at: `2026-03-08T10:49:35.298Z`
- Metric snapshot: ndcg@3=0.5241, hit@1=0.6076, r@3=0.5007, p@3=0.5007, auc=0.7103, val_loss=1.4163
- Data: races=2379, runners=24734, 2026-02-01 → 2026-03-07
- Backend: tfjs-node-gpu (# runs with GPU backend (default if installed): add --backend=gpu when using CLI entrypoint)

Ready-to-copy command (programmatic):

```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-reg',  })"
```

PowerShell (with backend env override):

```powershell
$env:TFJS_BACKEND='gpu'; node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-lr1e-3-reg',  })"
```

#### exp-ln-preenc16-auxanneal-gpu

- Folder: `C:/Users/juhaj/WebstormProjects/toto-prediction-model/model-training/model-mixed/runs/exp-ln-preenc16-auxanneal-gpu`
- Source file: `model_best_ndcg3.json`
- Saved at: `2026-03-10T07:55:26.790Z`
- Metric snapshot: ndcg@3=0.5236, hit@1=0.5738, r@3=0.5091, p@3=0.5091, auc=0.7332, val_loss=1.6261
- Data: races=2379, runners=24734, 2026-02-01 → 2026-03-07
- Backend: tfjs-node-gpu (# runs with GPU backend (default if installed): add --backend=gpu when using CLI entrypoint)

Ready-to-copy command (programmatic):

```
node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-preenc16-auxanneal-gpu',  })"
```

PowerShell (with backend env override):

```powershell
$env:TFJS_BACKEND='gpu'; node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: 'exp-ln-preenc16-auxanneal-gpu',  })"
```
