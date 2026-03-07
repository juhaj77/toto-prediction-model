// ═══════════════════════════════════════════════════════════════════════════════
// TOTO PREDICTION MODEL — model_mixed_runner_and_race.js
//
// Purpose: Combine Runner encoder representations with Race model context.
//
// Architecture overview (per race, per runner slot):
//   1) Runner sequence encoder (per runner): Masking → LSTM(64, seq) → LSTM(32) →
//      Dense projection (32) → Dropout. This yields a compact runner embedding.
//   2) Static branch (per runner): Dense(48) → BN → Dense(32) → Dropout.
//   3) Concatenate [RunnerEmbedding, StaticProc] → TD Dense(64) → BN.
//   4) Multi‑Head Self‑Attention over runner dimension (numHeads=8, embedDim=64)
//      with pre/post LayerNorm and a small FFN residual block for stability.
//   5) Output: TD Dropout → TD Dense(24, relu) → TD Dense(1, sigmoid), masked.
//
// Inputs (batch implicit):
//   history_input: [MAX_RUNNERS, MAX_HISTORY, histFeatures]  // -1 padded + Masking
//   static_input:  [MAX_RUNNERS, staticFeatures]
//   mask_input:    [MAX_RUNNERS, 1]                          // 1 for valid runner, 0 padded
// Output:
//   [MAX_RUNNERS, 1]  — top‑3 probability per runner slot
//
// Notes:
// - This is a light‑weight integration: the Runner encoder is embedded directly
//   into the TimeDistributed branch of the Race model.
// - Designed to lift ranking quality (NDCG@3/Hit@1) by providing richer
//   individual history representation to the within‑race attention.
// - Keep the decision rule per race as Top‑K (K=3 for sijoitus, K=1 for voittaja)
//   and avoid any global 0.5 thresholds.
//
// Training:
// - This file only builds the model. Reuse the training loop from model_race.js,
//   or compile here via options.compile=true. Loss defaults to BCE plus optional
//   soft Top‑K auxiliary loss (weight 0.3).
//
// ═══════════════════════════════════════════════════════════════════════════════

'use strict';

let __running = false; // simple in-process concurrency guard

const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const { MultiHeadAttention } = require('./MultiHeadAttention.js');

// Paths and constants (distinct for mixed model)
const TRAINING_DATA = './training_data.json';
const MODEL_FOLDER  = './model-mixed';
const MAPPINGS_FILE = './mappings_mixed.json';
const MODEL_MAIN    = 'model.json';
const MAX_HISTORY   = 8;    // expected history steps in data (can be inferred from loadData when needed)
const MAX_RUNNERS   = 18;   // padding up to max field size

// ─── LOSSES & METRICS (minimal subset reused locally) ──────────────────────────

// Binary cross-entropy, numerically stable
function bceLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const one = tf.scalar(1);
    const p = yPred.clipByValue(1e-7, 1 - 1e-7);
    const term1 = yTrue.mul(p.log());
    const term2 = one.sub(yTrue).mul(one.sub(p).log());
    return term1.add(term2).neg().mean();
  });
}

// Differentiable soft Top‑K auxiliary loss (K=3 by default)
function topKSoftAuxLoss(yTrue, yPred, K = 3, smooth = 0.1) {
  return tf.tidy(() => {
    const yT = yTrue.squeeze([-1]);                             // [b, r]
    const p  = yPred.squeeze([-1]).clipByValue(1e-7, 1 - 1e-7); // [b, r]
    const logits = p.log().sub(tf.scalar(1).sub(p).log());      // logit(p)
    const q = tf.softmax(logits, 1);                            // [b, r]

    const pos = yT;                                             // [b, r]
    const posCount = pos.sum(1, true);                          // [b, 1]
    const safeDen = posCount.add(posCount.equal(0).toFloat());  // avoid /0
    let t = pos.div(safeDen);                                   // normalize positives per race

    if (smooth > 0) {
      const smoothBase = q.stopGradient ? q.stopGradient() : q; // soft uniform-ish mass
      t = t.mul(tf.scalar(1 - smooth)).add(smoothBase.mul(tf.scalar(smooth)));
      const z = t.sum(1, true);
      t = t.div(z);
    }

    const ce = t.mul(q.add(1e-7).log()).sum(1).neg().mean();
    return ce;
  });
}

// Combined race loss: BCE + auxWeight * soft Top‑K
function combinedRaceLoss(auxWeight = 0.3) {
  return (yTrue, yPred) => tf.tidy(() => {
    const bce = bceLoss(yTrue, yPred);
    if (!auxWeight || auxWeight <= 0) return bce;
    const aux = topKSoftAuxLoss(yTrue, yPred, 3, 0.1);
    return bce.add(aux.mul(tf.scalar(auxWeight)));
  });
}

// Per‑race Recall@3
function recallAtThree(yTrue, yPred) {
  return tf.tidy(() => {
    const yT = yTrue.squeeze([-1]);
    const yP = yPred.squeeze([-1]);
    const topk = tf.topk(yP, 3, true);
    const idx  = topk.indices;
    const numRunners = yP.shape[1];
    const oneHot = tf.oneHot(idx, numRunners);
    const top3Mask = oneHot.sum(1);
    const captured = yT.mul(top3Mask).sum(1);
    const denom = yT.sum(1);
    const safeDenom = denom.add(denom.equal(0).toFloat());
    return captured.div(safeDenom).mean();
  });
}

// Per‑race Precision@3
function precisionAtThree(yTrue, yPred) {
  return tf.tidy(() => {
    const yT = yTrue.squeeze([-1]);
    const yP = yPred.squeeze([-1]);
    const topk = tf.topk(yP, 3, true);
    const idx  = topk.indices;
    const numRunners = yP.shape[1];
    const oneHot = tf.oneHot(idx, numRunners);
    const top3Mask = oneHot.sum(1);
    const truePos = yT.mul(top3Mask).sum(1);
    return truePos.div(tf.scalar(3)).mean();
  });
}

// ─── EVAL/UTIL HELPERS ───────────────────────────────────────────────────────
function seededRng(seed = 42) {
  let s = Math.floor(seed) >>> 0;
  return function () {
    s = (1664525 * s + 1013904223) >>> 0;
    return (s & 0xffffffff) / 0x100000000;
  };
}

function seededShuffle(arr, seed = 42) {
  const rng = seededRng(seed);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
  }
  return arr;
}

function splitTrainValRaces(hist, stat, mask, y, valFraction = 0.1, seed = 42, dates = null, temporal = false) {
  const n = y.shape[0];
  let trainIdxArr = [], valIdxArr = [];
  if (temporal && Array.isArray(dates) && dates.length === n) {
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => new Date(dates[a]) - new Date(dates[b]));
    const valCount = Math.max(1, Math.floor(n * valFraction));
    valIdxArr = order.slice(-valCount);
    trainIdxArr = order.slice(0, n - valCount);
  } else {
    const idx = Array.from({ length: n }, (_, i) => i);
    seededShuffle(idx, seed);
    const valCount = Math.max(1, Math.floor(n * valFraction));
    valIdxArr = idx.slice(0, valCount);
    trainIdxArr = idx.slice(valCount);
  }

  const idxTrain = tf.tensor1d(trainIdxArr, 'int32');
  const idxVal   = tf.tensor1d(valIdxArr, 'int32');

  const histTrain = tf.gather(hist, idxTrain);
  const statTrain = tf.gather(stat, idxTrain);
  const maskTrain = tf.gather(mask, idxTrain);
  const yTrain    = tf.gather(y, idxTrain);

  const histVal = tf.gather(hist, idxVal);
  const statVal = tf.gather(stat, idxVal);
  const maskVal = tf.gather(mask, idxVal);
  const yVal    = tf.gather(y, idxVal);

  idxTrain.dispose(); idxVal.dispose();
  return { histTrain, statTrain, maskTrain, yTrain, histVal, statVal, maskVal, yVal, trainIdxArr, valIdxArr };
}

function computeRocAuc(yTrueArr, yPredArr, step = 0.01) {
  const points = [];
  for (let t = 0.0; t <= 1.000001; t += step) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < yTrueArr.length; i++) {
      const y = yTrueArr[i] >= 0.5 ? 1 : 0;
      const yhat = yPredArr[i] >= t ? 1 : 0;
      if (yhat === 1 && y === 1) tp++;
      else if (yhat === 1 && y === 0) fp++;
      else if (yhat === 0 && y === 0) tn++;
      else fn++;
    }
    const tpr = tp / Math.max(1, tp + fn);
    const fpr = fp / Math.max(1, fp + tn);
    points.push({ fpr, tpr });
  }
  points.push({ fpr: 0, tpr: 0 });
  points.push({ fpr: 1, tpr: 1 });
  points.sort((a, b) => a.fpr - b.fpr);
  let auc = 0;
  for (let i = 1; i < points.length; i++) {
    const x0 = points[i - 1].fpr, x1 = points[i].fpr;
    const y0 = points[i - 1].tpr, y1 = points[i].tpr;
    auc += Math.max(0, x1 - x0) * (y0 + y1) / 2;
  }
  return { auc: Math.max(0, Math.min(1, auc)), rocCurve: points };
}

// Per-race metrics (Precision/Recall/NDCG@k, Hit@1, AP macro, per-race logloss)
function computePerRaceEval(yTrue, yPred, mask, k = 3) {
  const nRaces = yTrue.shape[0];
  const maxR = yTrue.shape[1];
  const yT = Array.from(yTrue.dataSync());
  const yP = Array.from(yPred.dataSync());
  const mK = Array.from(mask.dataSync());
  let sumPAtK = 0, sumRAtK = 0, sumNDCG = 0, sumHit1 = 0, sumAP = 0, sumLogLoss = 0, counted = 0;
  const eps = 1e-7;

  for (let r = 0; r < nRaces; r++) {
    const labs = [];
    const scs  = [];
    for (let j = 0; j < maxR; j++) {
      const idx = (r * maxR + j) * 1; // last dim is 1
      const valid = mK[idx] >= 0.5;
      if (!valid) continue;
      labs.push(yT[idx]);
      scs.push(yP[idx]);
    }
    const n = scs.length;
    if (n === 0) continue;
    const posCount = labs.reduce((a,b)=>a+(b>=0.5?1:0),0);
    if (posCount === 0) continue;

    const order = Array.from({length:n}, (_,i)=>i).sort((a,b)=>scs[b]-scs[a]);
    const sortedLabs = order.map(i=>labs[i]);

    const kk = Math.min(k, n);
    let tpAtK = 0;
    for (let i = 0; i < kk; i++) tpAtK += (sortedLabs[i] >= 0.5 ? 1 : 0);
    const pAtK = tpAtK / kk;
    const rAtK = tpAtK / Math.max(1, posCount);

    const hit1 = (sortedLabs[0] >= 0.5) ? 1 : 0;

    const dcg = (()=>{ let s=0; const log2=(x)=>Math.log(x)/Math.log(2); for(let i=0;i<kk;i++){ const rel=(sortedLabs[i] >= 0.5)?1:0; s += (Math.pow(2, rel) - 1)/log2(i+2);} return s; })();
    const ideal = (()=>{ let s=0; const log2=(x)=>Math.log(x)/Math.log(2); const kk2=Math.min(kk, posCount); for (let i=0;i<kk2;i++) s += (Math.pow(2,1)-1)/log2(i+2); return Math.max(eps, s); })();
    const ndcg = dcg / ideal;

    // Macro-AP per race (binary labels)
    let apSum=0, hits=0;
    for (let i=0;i<n;i++){
      if (sortedLabs[i] >= 0.5){ hits++; apSum += hits/(i+1); }
    }
    const ap = hits>0 ? apSum/Math.max(1, posCount) : 0;

    // Per-race average logloss over valid slots
    let ll=0; for (let i=0;i<n;i++){ const p=scs[order[i]]; const y=(sortedLabs[i]>=0.5?1:0); const pp=Math.min(1-1e-7, Math.max(1e-7, p)); ll += -(y*Math.log(pp) + (1-y)*Math.log(1-pp)); }
    ll /= Math.max(1, n);

    sumPAtK += pAtK; sumRAtK += rAtK; sumNDCG += ndcg; sumHit1 += hit1; sumAP += ap; sumLogLoss += ll; counted++;
  }

  const denom = Math.max(1, counted);
  return {
    precisionK: sumPAtK/denom,
    recallK:    sumRAtK/denom,
    ndcgK:      sumNDCG/denom,
    hit1:       sumHit1/denom,
    apMacro:    sumAP/denom,
    loglossRace:sumLogLoss/denom,
  };
}

// Global Average Precision (PR-AUC) over valid slots
function computeAveragePrecision(yTrueArr, yPredArr){
  const pairs = [];
  for (let i=0;i<yTrueArr.length;i++) pairs.push({y: yTrueArr[i] >= 0.5 ? 1:0, s: yPredArr[i]});
  pairs.sort((a,b)=> b.s - a.s);
  let tp=0, fp=0; let apSum=0; let pos= pairs.reduce((a,b)=>a+b.y,0);
  for (let i=0;i<pairs.length;i++){
    if (pairs[i].y===1){ tp++; apSum += tp/(tp+fp); } else { fp++; }
  }
  return pos>0 ? apSum/pos : 0;
}

// ─── DATA LOADING (mixed) ─────────────────────────────────────────────────────
function extractSurname(fullName){ if(!fullName) return 'unknown'; return fullName.trim().toLowerCase().split(' ').at(-1); }
function parseDate(str){ if(!str) return null; if(str.includes('-')) return new Date(str); if(str.includes('.')){ const [d,m,y]=str.split('.'); const year = parseInt(y)<100 ? parseInt(y)+2000 : parseInt(y); return new Date(year, parseInt(m)-1, parseInt(d)); } return null; }
function normaliseKmTime(km, distance){ if(km===null||km===undefined||isNaN(km)||km<=0) return null; const dist=(distance!==null&&distance!==undefined&&distance>0)?distance:2100; return km + (2100 - dist)/2000; }

// Mirror encodeTrackCondition from model_race.js so history rows can include it
function encodeTrackCondition(condition){
  switch ((String(condition ?? '')).toLowerCase().trim()) {
    case 'heavy track':
    case 'heavy':             return 0.00;
    case 'sloppy':            return 0.10;
    case 'quite heavy track': return 0.25;
    case 'good':              return 0.70;
    case 'winter track':      return 0.75;
    case 'fast':              return 0.85;
    case 'light track':       return 1.00;
    default:                  return 0.50;
  }
}

function loadData(filePath = TRAINING_DATA){
  let maps = { coaches:{}, drivers:{}, tracks:{}, counts:{ c:1, d:1, t:1 } };
  function getID(map, name, type){ if(!name||name==='Unknown'||name==='') return 0; const key = (type==='driver') ? extractSurname(name) : name.trim().toLowerCase(); if(!map[key]){ map[key] = maps.counts[type[0]]; maps.counts[type[0]]++; } return map[key]; }

  const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  const races = raw.races || [];

  // Per-breed means for imputation
  const breedStats = { SH:{ records:[], kmTimes:[] }, LV:{ records:[], kmTimes:[] } };
  for (const race of races){ const breed = race.isColdBlood ? 'SH' : 'LV'; for(const runner of (race.runners||[])) { if(runner.record>0) breedStats[breed].records.push(runner.record); for (const ps of (runner.prevStarts||[])){ const km = normaliseKmTime(ps.kmTime, ps.distance); if(km>0) breedStats[breed].kmTimes.push(km); } } }
  const avg=arr=>arr.length?arr.reduce((a,b)=>a+b,0)/arr.length:null;
  const means={ SH:{ record: avg(breedStats.SH.records) ?? 28.0, km: avg(breedStats.SH.kmTimes) ?? 29.0 }, LV:{ record: avg(breedStats.LV.records) ?? 15.0, km: avg(breedStats.LV.kmTimes) ?? 16.0 } };

  const X_hist=[], X_static=[], Y=[], X_mask=[]; // shapes per race
  let keptRaces=0, keptRunners=0, droppedRaces=0; const keptDates=[]; const raceDates=[];

  for (const race of races){
    const starters = (race.runners||[]).filter(r=>!r.scratched);
    const hasPositive = starters.some(r=> r && r.position!=null && r.position>=1 && r.position<=3);
    if(!hasPositive){ droppedRaces++; continue; }
    keptRaces++; keptRunners+=starters.length; if(race.date) { keptDates.push(race.date); raceDates.push(race.date); } else { raceDates.push(null); }

    const raceHist=[], raceStatic=[], raceY=[], raceMask=[];
    for (let slot=0; slot<MAX_RUNNERS; slot++){
      const runner = starters[slot];
      if(!runner){
        raceStatic.push(new Array(25).fill(0));
        raceHist.push(Array.from({length: MAX_HISTORY}, ()=> new Array(25).fill(-1)));
        raceY.push([0]); raceMask.push([0]);
        continue;
      }
      raceMask.push([1]);

      const frontStr=(runner.frontShoes||'').toUpperCase(); const rearStr=(runner.rearShoes||'').toUpperCase();
      const frontActive=frontStr==='HAS_SHOES'?1:0; const frontKnown=(frontStr==='HAS_SHOES'||frontStr==='NO_SHOES')?1:0;
      const rearActive=rearStr==='HAS_SHOES'?1:0; const rearKnown=(rearStr==='HAS_SHOES'||rearStr==='NO_SHOES')?1:0;
      const cartStr=(runner.specialCart||'').toUpperCase(); const cartActive=cartStr==='YES'?1:0; const cartKnown=(cartStr==='YES'||cartStr==='NO')?1:0;

      const coachId = getID(maps.coaches, runner.coachName||runner.coach||'', 'coach');
      const driverId = getID(maps.drivers, runner.driverName||runner.driver||'', 'driver');

      // Compute prevStarts quality indicators similar to model_race.js
      const prevStarts  = runner.prevStarts || [];
      const validPrev   = prevStarts.filter(ps => ps.date && ps.driver);
      const histKnown   = validPrev.length > 0 ? 1 : 0;
      let prevIndexNorm = 0;
      if (validPrev.length > 0) {
        let score = 0, count = 0;
        for (const ps of validPrev) {
          const pos = ps.position;
          if (pos == null) continue;
          count++;
          if      (pos === 1) score += 1.00;
          else if (pos === 2) score += 0.50;
          else if (pos === 3) score += 0.33;
        }
        prevIndexNorm = count > 0 ? score / count : 0;
      }

      // Static vector — align with model_race.js (25 dims, no trackId)
      const breed = race.isColdBlood ? 'SH' : 'LV';
      const record = (runner.record && runner.record>0) ? runner.record : means[breed].record;
      const startNum = Math.max(1, Number(runner.number || runner.startNumber || runner.draw || 1));
      const genderVal = (runner.gender != null) ? Number(runner.gender) : 2; // 1=mare 2=gelding 3=stallion (mapped earlier in pipeline)
      const staticVec = [
        (startNum) / 20,                              // [0]  start number normalized
        coachId/6000.0,                               // [1]  coach ID
        (record || 0) / 50,                           // [2]  record (imputed if missing)
        driverId/5000.0,                              // [3]  driver ID
        (Number(runner.age || 5)) / 15,               // [4]  age
        (genderVal) / 3,                              // [5]  gender code 1..3
        (race.isColdBlood ? 1 : 0),                   // [6]  breed (cold blood flag)
        frontActive, frontKnown,                      // [7-8]
        rearActive,  rearKnown,                       // [9-10]
        (runner.frontShoesChanged ? 1 : 0),           // [11]
        (runner.rearShoesChanged  ? 1 : 0),           // [12]
        (Number(race.distance || 2100)) / 3100,       // [13] distance norm
        (race.isCarStart ? 1 : 0),                    // [14] car start
        (Number(runner.bettingPercentage || 0)) / 100,// [15] betting %
        (Number(runner.winPercentage || 0)) / 100,    // [16] win %
        (Number(runner.winPercentage || 0) > 0 ? 1:0),// [17] win% known
        (runner.isAutoRecord ? 1 : 0),                // [18] auto record flag
        cartActive, cartKnown,                        // [19-20] special cart on/known
        prevIndexNorm,                                // [21]
        histKnown,                                    // [22]
        0, 0                                          // [23-24] reserved
      ];
      raceStatic.push(staticVec);

      // History matrix MAX_HISTORY x 25 — mirror model_race.js
      const hist = [];
      for (let i = 0; i < MAX_HISTORY; i++) {
        const ps = validPrev[i];
        if (!ps) { hist.push(new Array(25).fill(-1)); continue; }

        const raceDate  = parseDate(race.date);
        const startDate = parseDate(ps.date);
        const daysSince = (raceDate && startDate) ? Math.min(365, (raceDate - startDate) / 86400000) : 30;

        const kmNorm  = normaliseKmTime(ps.kmTime, ps.distance);
        const kmKnown = kmNorm > 0 ? 1 : 0;
        const kmFinal = kmKnown ? kmNorm / 100 : (means[breed].km / 100);

        const distKnown  = (ps.distance || 0) > 0 ? 1 : 0;
        const distFinal  = distKnown ? (ps.distance / 3100) : 0.67;

        let prizeVal = parseFloat(ps.firstPrice) || 0;
        if (prizeVal > 200000)      prizeVal = prizeVal / 10000;
        else if (prizeVal > 2000)   prizeVal = prizeVal / 100;
        const prizeKnown = prizeVal > 0 ? 1 : 0;
        const prizeFinal = prizeKnown ? Math.log1p(prizeVal) / 10 : 0.55;

        const oddKnown   = (ps.odd || 0) > 0 ? 1 : 0;
        const oddFinal   = oddKnown ? Math.log1p(ps.odd) / 5 : 0.50;

        const posKnown   = ps.position != null && ps.position > 0 ? 1 : 0;
        const posFinal   = posKnown ? ps.position / 20 : 0.5;

        const psfront = (ps.frontShoes  || '').toUpperCase();
        const psrear  = (ps.rearShoes   || '').toUpperCase();
        const pscart  = (ps.specialCart || '').toUpperCase();

        hist.push([
          kmFinal,    kmKnown,                                           // [0-1]
          distFinal,  distKnown,                                         // [2-3]
          daysSince / 365,                                               // [4]
          posFinal,   posKnown,                                          // [5-6]
          prizeFinal, prizeKnown,                                        // [7-8]
          oddFinal,   oddKnown,                                          // [9-10]
          (ps.isCarStart ? 1 : 0),                                       // [11]
          (ps.isGallop   ? 1 : 0),                                       // [12]
          (ps.number ?? 1) / 30,                                         // [13]
          getID(maps.drivers, ps.driver, 'driver') / 5000,               // [14]
          getID(maps.tracks,  ps.track,  'track')  / 600,                // [15]
          (ps.disqualified ? 1 : 0),                                     // [16]
          (ps.DNF          ? 1 : 0),                                     // [17]
          (psfront === 'HAS_SHOES' ? 1 : 0),                             // [18]
          ((psfront === 'HAS_SHOES' || psfront === 'NO_SHOES') ? 1 : 0), // [19]
          (psrear  === 'HAS_SHOES' ? 1 : 0),                             // [20]
          ((psrear  === 'HAS_SHOES' || psrear  === 'NO_SHOES') ? 1 : 0), // [21]
          (pscart  === 'YES' ? 1 : 0),                                   // [22]
          ((pscart  === 'YES' || pscart  === 'NO') ? 1 : 0),             // [23]
          encodeTrackCondition(ps.trackCondition),                       // [24]
        ]);
      }
      raceHist.push(hist);

      // Target: top-3 finish
      const y = (runner.position!=null && runner.position>=1 && runner.position<=3) ? 1 : 0;
      raceY.push([y]);
    }

    X_hist.push(raceHist); X_static.push(raceStatic); X_mask.push(raceMask); Y.push(raceY);
  }

  // Write mappings to distinct file for mixed model
  fs.writeFileSync(MAPPINGS_FILE, JSON.stringify(maps, null, 2));
  console.log(`  Mappings (mixed) saved: coaches=${Object.keys(maps.coaches).length}, drivers=${Object.keys(maps.drivers).length}, tracks=${Object.keys(maps.tracks).length}`);

  const allDates = keptDates.length ? keptDates.slice().sort() : races.map(r=>r.date).filter(Boolean).sort();
  const dataMeta = {
    totalRaces:    keptRaces>0 ? keptRaces : races.length,
    totalRunners:  keptRunners>0 ? keptRunners : races.reduce((n,r)=> n + (r.runners||[]).filter(ru=>!ru.scratched).length, 0),
    dataStartDate: allDates[0] ?? null,
    dataEndDate:   allDates[allDates.length-1] ?? null,
    droppedRacesNoTarget: droppedRaces,
  };
  console.log(`  Data (mixed): ${dataMeta.totalRunners} runners across ${dataMeta.totalRaces} races, ${dataMeta.dataStartDate} → ${dataMeta.dataEndDate}`);
  if (droppedRaces>0) console.log(`  Filter: dropped ${droppedRaces} races without any valid target.`);

  return {
    hist:               tf.tensor4d(X_hist),
    static:             tf.tensor3d(X_static),
    mask:               tf.tensor3d(X_mask),
    y:                  tf.tensor3d(Y),
    histFeatureCount:   25,
    staticFeatureCount: 25,
    dataMeta,
    raceDates,
  };
}

// ─── MODEL PERSISTENCE (mixed) ────────────────────────────────────────────────
async function saveModel(model, meta = {}, fileName = MODEL_MAIN, folder = MODEL_FOLDER){
  const outDir = folder || MODEL_FOLDER;
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  await model.save(tf.io.withSaveHandler(async (artifacts)=>{
    const payload = {
      modelTopology: artifacts.modelTopology,
      weightSpecs:   artifacts.weightSpecs,
      weightData:    Buffer.from(artifacts.weightData).toString('base64'),
      trainingInfo: {
        savedAt:       new Date().toISOString(),
        epoch:         meta.epoch         ?? null,
        loss:          meta.loss          ?? null,
        val_loss:      meta.val_loss      ?? null,
        val_acc:       meta.val_acc       ?? null,
        val_r3:        meta.val_r3        ?? null,
        val_p3:        meta.val_p3        ?? null,
        val_auc:       meta.val_auc       ?? null,
        val_ndcg3:     meta.val_ndcg3     ?? null,
        val_hit1:      meta.val_hit1      ?? null,
        val_ap_macro:  meta.val_ap_macro  ?? null,
        val_logloss_race: meta.val_logloss_race ?? null,
        val_best_threshold: meta.val_best_threshold ?? meta.recommended_threshold ?? null,
        val_f1_best:   meta.val_f1_best   ?? null,
        best_by:       meta.best_by       ?? 'val_loss',
        learningRate:  meta.learningRate  ?? null,
        dataStartDate: meta.dataStartDate ?? null,
        dataEndDate:   meta.dataEndDate   ?? null,
        totalRaces:    meta.totalRaces    ?? null,
        totalRunners:  meta.totalRunners  ?? null,
        recommended_threshold: meta.recommended_threshold ?? null,
        recommended_K: meta.recommended_K ?? 3,
        calibration:   meta.calibration   ?? null,
        metrics_at_selection: meta.metrics_at_selection ?? null,
        history:       meta.history       ?? null,
      },
    };
    fs.writeFileSync(`${outDir}/${fileName}`, JSON.stringify(payload));
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
  }));
}

// ─── MODEL BUILDER ─────────────────────────────────────────────────────────────
/**
 * Build the Mixed Runner+Race model: per‑runner history encoder feeding the race context.
 *
 * @param {number} maxRunners      number of runner slots (e.g., 18)
 * @param {number} timeSteps       history time steps (e.g., 8)
 * @param {number} histFeatures    number of history features (e.g., 25)
 * @param {number} staticFeatures  number of static features (e.g., 25)
 * @param {object} options         optional: {
 *   runnerProjDim=32, attnHeads=8, embedDim=64, ffnDim=128, dropout=0.2,
 *   outDropout=0.25, auxLossWeight=0.3, learningRate=3e-4, compile=true
 * }
 * @returns {tf.LayersModel}
 */
function buildMixedModel(maxRunners, timeSteps, histFeatures, staticFeatures, options = {}) {
  const opt = Object.assign({
    runnerProjDim: 32,
    attnHeads: 8,
    embedDim: 64,
    ffnDim: 128,
    dropout: 0.2,
    outDropout: 0.25,
    outUnits: 24,
    auxLossWeight: 0.3,
    learningRate: 3e-4,
    l2: 5e-4,
    useLayerNormInStatic: false,
    swapBNtoLN: false,
    runnerLstm2Units: 32, // set 64 to try 64→64
    compile: true,
    useListNet: false,
  }, options);

  const l2reg = tf.regularizers.l2({ l2: opt.l2 });

  const histInput   = tf.input({ shape: [maxRunners, timeSteps, histFeatures], name: 'history_input' });
  const staticInput = tf.input({ shape: [maxRunners, staticFeatures],          name: 'static_input'  });
  const maskInput   = tf.input({ shape: [maxRunners, 1],                       name: 'mask_input'    });

  // ── Runner sequence encoder (per runner via TimeDistributed) ────────────────
  let h = tf.layers.timeDistributed({ layer: tf.layers.masking({ maskValue: -1 }), name: 'hist_masking' })
    .apply(histInput);

  h = tf.layers.timeDistributed({
    layer: tf.layers.lstm({ units: 64, returnSequences: true, recurrentDropout: 0.2,
                            kernelRegularizer: l2reg }),
    name: 'hist_lstm1',
  }).apply(h);

  h = tf.layers.timeDistributed({
    layer: tf.layers.lstm({ units: opt.runnerLstm2Units, returnSequences: false, recurrentDropout: 0.2,
                            kernelRegularizer: l2reg }),
    name: 'hist_lstm2',
  }).apply(h); // [b, R, runnerLstm2Units]

  // Small projection head for runner embedding
  h = tf.layers.timeDistributed({
    layer: tf.layers.dense({ units: opt.runnerProjDim, activation: 'relu', kernelRegularizer: l2reg }),
    name: 'runner_proj',
  }).apply(h); // [b, R, runnerProjDim]

  h = tf.layers.timeDistributed({ layer: tf.layers.dropout({ rate: 0.1 }), name: 'runner_proj_dropout' }).apply(h);

  // ── Static branch ───────────────────────────────────────────────────────────
  let s = tf.layers.timeDistributed({
    layer: tf.layers.dense({ units: 48, activation: 'relu', kernelRegularizer: l2reg }),
    name: 'static_dense1',
  }).apply(staticInput);

  if (opt.useLayerNormInStatic || opt.swapBNtoLN) {
    s = tf.layers.timeDistributed({ layer: tf.layers.layerNormalization(), name: 'static_ln' }).apply(s);
  } else {
    s = tf.layers.timeDistributed({ layer: tf.layers.batchNormalization(), name: 'static_bn' }).apply(s);
  }
  s = tf.layers.timeDistributed({ layer: tf.layers.dense({ units: 32, activation: 'relu', kernelRegularizer: l2reg }), name: 'static_dense2' }).apply(s);
  s = tf.layers.timeDistributed({ layer: tf.layers.dropout({ rate: 0.2 }), name: 'static_dropout' }).apply(s);

  // ── Merge branches (concat runner embedding + static) ───────────────────────
  let combined = tf.layers.concatenate({ axis: -1, name: 'combine' }).apply([h, s]); // [b, R, runnerProjDim+32]
  combined = tf.layers.timeDistributed({ layer: tf.layers.dense({ units: opt.embedDim, activation: 'relu', kernelRegularizer: l2reg }), name: 'combined_dense' }).apply(combined);
  if (opt.swapBNtoLN) {
    combined = tf.layers.timeDistributed({ layer: tf.layers.layerNormalization(), name: 'combined_ln' }).apply(combined);
  } else {
    combined = tf.layers.timeDistributed({ layer: tf.layers.batchNormalization(), name: 'combined_bn' }).apply(combined);
  }

  // ── Multi‑Head Self‑Attention over runners ──────────────────────────────────
  const preAttn = tf.layers.layerNormalization({ name: 'pre_attention_ln' }).apply(combined);

  const attended = new MultiHeadAttention({
    numHeads: opt.attnHeads,
    embedDim: opt.embedDim,
    dropout:  0.1,
  }).apply(preAttn); // [b, R, embedDim]

  // Residual-style fusion: concat → project → LN
  const fused = tf.layers.concatenate({ axis: -1, name: 'attention_fuse_concat' }).apply([combined, attended]);
  const fusedProj = tf.layers.timeDistributed({ layer: tf.layers.dense({ units: opt.embedDim, activation: 'linear', kernelRegularizer: l2reg }), name: 'attention_fuse_proj' }).apply(fused);
  let normed = tf.layers.layerNormalization({ name: 'attention_ln' }).apply(fusedProj);

  // FFN block with residual + LayerNorm
  let ffn = tf.layers.timeDistributed({ layer: tf.layers.dense({ units: opt.ffnDim, activation: 'relu', kernelRegularizer: l2reg }), name: 'ffn_dense1' }).apply(normed);
  ffn = tf.layers.timeDistributed({ layer: tf.layers.dropout({ rate: opt.dropout }), name: 'ffn_dropout' }).apply(ffn);
  ffn = tf.layers.timeDistributed({ layer: tf.layers.dense({ units: opt.embedDim, activation: 'linear', kernelRegularizer: l2reg }), name: 'ffn_dense2' }).apply(ffn);

  const resid = tf.layers.add({ name: 'ffn_residual_add' }).apply([normed, ffn]);
  normed = tf.layers.layerNormalization({ name: 'post_ffn_ln' }).apply(resid);

  // ── Output head ─────────────────────────────────────────────────────────────
  let out = tf.layers.timeDistributed({ layer: tf.layers.dropout({ rate: opt.outDropout }), name: 'out_dropout' }).apply(normed);
  out = tf.layers.timeDistributed({ layer: tf.layers.dense({ units: opt.outUnits, activation: 'relu', kernelRegularizer: l2reg }), name: 'out_dense' }).apply(out);
  out = tf.layers.timeDistributed({ layer: tf.layers.dense({ units: 1, activation: 'sigmoid' }), name: 'output' }).apply(out);

  // Apply mask to zero out padding slots
  out = tf.layers.multiply({ name: 'mask_apply' }).apply([out, maskInput]);

  const model = tf.model({ inputs: [histInput, staticInput, maskInput], outputs: out, name: 'MixedRunnerRace' });

  if (opt.compile) {
    const optimizer = tf.train.adam(opt.learningRate);
    const lossFn = opt.useListNet ? (yTrue, yPred)=> topKSoftAuxLoss(yTrue, yPred, 3, 0.1) : combinedRaceLoss(opt.auxLossWeight);
    model.compile({
      optimizer,
      loss:      lossFn,
      metrics:   ['accuracy', recallAtThree, precisionAtThree],
    });
  }

  // Attach lightweight metadata for consumers (decision rule etc.)
  model.recommended = {
    decision: 'Top-K per race',
    K_default: 3,
    notes: 'Avoid 0.5 threshold; choose Top-1/Top-3 per race based on use-case (ROI vs sijoitus).',
    calibration: { method: 'isotonic or temperature scaling (per race)', status: 'todo' },
  };

  return model;
}

// ─── TRAINING LOOP (mixed) ───────────────────────────────────────────────────
async function runTraining(opts = {}){
  if (__running) { console.warn('runTraining is already running in this process. Ignoring second call.'); return null; }
  __running = true;
  const startedAt = new Date();
  const runId = opts.runId || `${startedAt.toISOString().replace(/[:.]/g,'-')}-${Math.random().toString(36).slice(2,7)}`;
  const runFolder = `${MODEL_FOLDER}/runs/${runId}`;
  const options = Object.assign({
    trainingFile: TRAINING_DATA,
    valFraction: 0.1,
    temporalSplit: true,
    seed: 42,
    epochs: 90,                 // pidennetty treeni
    batchSize: 384,
    learningRate: 3e-4,        // Adam (L2 toimii weight decayna)
    minLearningRate: 3e-5,
    scheduler: 'cosine',       // warmup + cosine
    warmupEpochs: 4,
    earlyStopPatience: 12,
    plateauPatience: 5,
    plateauFactor: 0.5,
    auxLossWeight: 0.5,        // painotetaan ranking-auxia enemmän top-3‑tehtävään
    useListNet: false,         // vaihtoehtona voi asettaa true → puhdas listnet/soft top‑k
    attnHeads: 12,             // hieman lisää kapasiteettia
    embedDim: 96,
    ffnDim: 192,
    runnerProjDim: 48,
    runnerLstm2Units: 48,
    dropout: 0.15,
    outDropout: 0.25,
    outUnits: 24,
    l2: 1e-4,
    bestBy: 'val_ndcg3',       // optimoi suoraan ranking‑mittaria
  }, opts);

  console.log(`▶ Loading data for mixed model... (runId=${runId})`);
  if (!fs.existsSync(runFolder)) fs.mkdirSync(runFolder, { recursive: true });
  console.log(`  Outputs will be saved under: ${runFolder}`);
  const data = loadData(options.trainingFile);
  const { hist, static: stat, mask, y, raceDates } = data;

  const maxRunners = hist.shape[1];
  const timeSteps  = hist.shape[2];
  const histFeat   = hist.shape[3];
  const statFeat   = stat.shape[2];

  console.log(`  Shapes: hist=${hist.shape}, static=${stat.shape}, mask=${mask.shape}, y=${y.shape}`);

  const model = buildMixedModel(maxRunners, timeSteps, histFeat, statFeat, {
    runnerProjDim: options.runnerProjDim,
    runnerLstm2Units: options.runnerLstm2Units,
    attnHeads: options.attnHeads,
    embedDim: options.embedDim,
    ffnDim: options.ffnDim,
    auxLossWeight: options.auxLossWeight,
    learningRate: options.learningRate,
    l2: options.l2,
    swapBNtoLN: options.swapBNtoLN || false,
    useLayerNormInStatic: options.useLayerNormInStatic || false,
    outUnits: options.outUnits || 24,
    outDropout: options.outDropout || 0.25,
    useListNet: options.useListNet || false,
    compile: true,
  });

  const { histTrain, statTrain, maskTrain, yTrain, histVal, statVal, maskVal, yVal } =
      splitTrainValRaces(hist, stat, mask, y, options.valFraction, options.seed, raceDates || null, !!options.temporalSplit);

  let bestValLoss = Number.POSITIVE_INFINITY;
  let bestValR3   = -1;
  let bestValAUC  = -1;
  let bestValNDCG = -1;
  let patienceCounter = 0, targetPatience = 0;
  // Provide safe defaults if options are undefined
  const EARLY_STOP_PATIENCE = (options.earlyStopPatience != null)
    ? options.earlyStopPatience
    : Math.max(15, Math.floor(options.epochs * 0.2));
  const PLATEAU_PATIENCE = (options.plateauPatience != null)
    ? options.plateauPatience
    : 10;
  // Early stopping tuning: require minimum epochs and minimum meaningful improvement
  const MIN_EPOCHS_BEFORE_STOP = (options.minEpochsBeforeStop != null)
    ? options.minEpochsBeforeStop
    : Math.max(30, Math.min(60, Math.floor(options.epochs * 0.5)));
  const MIN_DELTA = (options.minDelta != null)
    ? options.minDelta
    : (options.bestBy === 'val_loss' ? 1e-3 : 2e-3); // need >=0.2% AUC (or similar) improvement by default
  // Optional metric floor to prevent stopping too early (default only for AUC)
  const MIN_METRIC_FLOOR = (options.minMetricFloor != null)
    ? options.minMetricFloor
    : ((options.bestBy === 'val_auc' || options.bestBy == null) ? 0.75 : -Infinity);
  let epochStart;
  const history = [];

  // Helper: selection metric getter
  function getSelMetric(rec){
    switch (options.bestBy){
      case 'val_auc': return rec.val_auc;
      case 'val_ndcg3': return rec.val_ndcg3;
      case 'val_r3': return rec.val_r3;
      case 'val_loss': return -rec.val_loss; // invert so larger is better
      default: return rec.val_auc; // default to AUC for early stopping selection
    }
  }

  // Cosine schedule with warmup
  const baseLR = options.learningRate;
  const minLR  = options.minLearningRate;
  const totalE = options.epochs;
  const warmup = Math.max(0, Math.min(options.warmupEpochs, totalE-1));

  await model.fit([histTrain, statTrain, maskTrain], yTrain, {
    epochs: options.epochs,
    batchSize: options.batchSize,
    shuffle: true,
    validationData: [[histVal, statVal, maskVal], yVal],
    callbacks: {
      onEpochBegin: async (epoch) => {
        epochStart = Date.now();
        if (options.scheduler === 'cosine'){
          let newLR = baseLR;
          if (epoch < warmup){ newLR = baseLR * ((epoch + 1) / Math.max(1, warmup)); }
          else {
            const t = (epoch - warmup) / Math.max(1, (totalE - warmup));
            newLR = minLR + 0.5 * (baseLR - minLR) * (1 + Math.cos(Math.PI * t));
          }
          model.optimizer.learningRate = newLR;
        }
      },
      onEpochEnd: async (epoch, logs) => {
        const ms  = Date.now() - epochStart;
        const lr  = model.optimizer.learningRate;
        const acc = (logs.val_acc || logs.val_accuracy || 0).toFixed(4);

        const yPredValTensor = model.predict([histVal, statVal, maskVal], { batchSize: 16 });
        const perRace = computePerRaceEval(yVal, yPredValTensor, maskVal, 3);

        const yPredArr = Array.from(yPredValTensor.dataSync());
        yPredValTensor.dispose();
        const yTrueArrAll = Array.from(yVal.dataSync());
        const maskArr = Array.from(maskVal.dataSync());
        const yTrueArr = []; const yScoreArr = [];
        for (let i = 0; i < yTrueArrAll.length; i++){
          if (maskArr[i] >= 0.5){ yTrueArr.push(yTrueArrAll[i]); yScoreArr.push(yPredArr[i]); }
        }
        const { auc: rocAuc } = computeRocAuc(yTrueArr, yScoreArr, 0.01);
        const apGlobal = computeAveragePrecision(yTrueArr, yScoreArr);

        const r3ValNum = perRace.recallK;
        const p3ValNum = perRace.precisionK;

        const r3v = r3ValNum.toFixed(4), p3v = p3ValNum.toFixed(4), aucv = rocAuc.toFixed(4);
        const ndcg3v = perRace.ndcgK.toFixed(4);
        const hit1v  = perRace.hit1.toFixed(4);
        const apmv   = perRace.apMacro.toFixed(4);
        const apg    = apGlobal.toFixed(4);
        const llrv   = perRace.loglossRace.toFixed(4);

        const rec = {
          epoch: epoch + 1,
          loss: logs.loss,
          val_loss: logs.val_loss,
          val_acc: Number(logs.val_acc || logs.val_accuracy || 0),
          val_r3: r3ValNum,
          val_p3: p3ValNum,
          val_auc: rocAuc,
          val_pr_auc: apGlobal,
          val_ndcg3: perRace.ndcgK,
          val_hit1: perRace.hit1,
          val_ap_macro: perRace.apMacro,
          val_logloss_race: perRace.loglossRace,
          lr,
        };
        history.push(rec);

        // Compute and attach global best threshold (F1) before saving, so it appears in history/rec
        let bestThr = 0.5, bestF1 = 0;
        for (let t = 0.05; t <= 0.951; t += 0.01){
          let tp=0, fp=0, fn=0;
          for (let i=0;i<yTrueArr.length;i++){
            const y = yTrueArr[i] >= 0.5 ? 1 : 0;
            const yhat = yScoreArr[i] >= t ? 1 : 0;
            if (yhat===1 && y===1) tp++; else if (yhat===1 && y===0) fp++; else if (yhat===0 && y===1) fn++;
          }
          const prec = tp/Math.max(1,tp+fp);
          const recV  = tp/Math.max(1,tp+fn);
          const f1   = (prec+recV)>0 ? (2*prec*recV)/(prec+recV) : 0;
          if (f1 > bestF1){ bestF1 = f1; bestThr = t; }
        }
        history[history.length-1].val_best_threshold = bestThr;
        history[history.length-1].val_f1_best = bestF1;
        rec.val_best_threshold = bestThr;
        rec.val_f1_best = bestF1;

        console.log(
          `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} | ` +
          `val_loss=${logs.val_loss.toFixed(4)} | val_acc=${acc} | val_r@3=${r3v} | val_p@3=${p3v} | val_auc=${aucv} | ` +
          `PR_AUC=${apg} | ndcg@3=${ndcg3v} | hit@1=${hit1v} | AP_macro=${apmv} | race_logloss=${llrv} | ` +
          `lr=${lr.toFixed(6)} | ` +
          `${Math.floor(ms / 60000)}m ${((ms % 60000) / 1000).toFixed(0)}s`
        );

        // Save best by val_loss
        if (logs.val_loss < bestValLoss){
          bestValLoss = logs.val_loss; patienceCounter = 0;
          console.log('   ⭐ New best val_loss — saving (mixed)...');
          await saveModel(model, {
            ...rec,
            epoch: epoch + 1,
            loss: Math.round(logs.loss * 10000) / 10000,
            val_loss: Math.round(logs.val_loss * 10000) / 10000,
            learningRate: lr,
            best_by: 'val_loss',
            history,
            ...data.dataMeta,
          }, MODEL_MAIN, runFolder);
        } else { patienceCounter++; }

        // Save best by val_r3
        if (r3ValNum > bestValR3){
          bestValR3 = r3ValNum;
          console.log('   🌟 New best val_r@3 — saving (mixed)...');
          await saveModel(model, {
            ...rec,
            epoch: epoch + 1,
            learningRate: lr,
            best_by: 'val_r3',
            history,
            ...data.dataMeta,
          }, 'model_best_r3.json', runFolder);
        }

        // Save best by val_auc
        if (rocAuc > bestValAUC){
          bestValAUC = rocAuc;
          console.log('   🌟 New best val_auc — saving (mixed)...');
          await saveModel(model, {
            ...rec,
            epoch: epoch + 1,
            learningRate: lr,
            best_by: 'val_auc',
            history,
            ...data.dataMeta,
          }, 'model_best_auc.json', runFolder);
        }

        // Save best by val_ndcg3
        if (perRace.ndcgK > bestValNDCG){
          bestValNDCG = perRace.ndcgK;
          console.log('   🌟 New best val_ndcg@3 — saving (mixed)...');
          await saveModel(model, {
            ...rec,
            epoch: epoch + 1,
            learningRate: lr,
            best_by: 'val_ndcg3',
            history,
            ...data.dataMeta,
          }, 'model_best_ndcg3.json', runFolder);
        }

        // duplicate F1 threshold computation removed (computed earlier and stored in rec/history)

        // Plateau LR for non-cosine schedule
        if (options.scheduler !== 'cosine' && patienceCounter >= PLATEAU_PATIENCE){
          model.optimizer.learningRate = Math.max(minLR, lr * options.plateauFactor);
          console.log(`   📉 Reducing LR (plateau): ${model.optimizer.learningRate.toFixed(6)}`);
          patienceCounter = 0;
        }

        // Early stopping on selected metric (less aggressive)
        const currentSel = getSelMetric(rec);
        const enoughEpochs = (epoch + 1) >= MIN_EPOCHS_BEFORE_STOP;
        if (enoughEpochs && history.length >= EARLY_STOP_PATIENCE){
          const recent = history.slice(-EARLY_STOP_PATIENCE);
          const before = history.slice(0, -EARLY_STOP_PATIENCE);
          const bestRecent = Math.max(...recent.map(getSelMetric));
          const bestBefore = before.length ? Math.max(...before.map(getSelMetric)) : -Infinity;
          const improved = bestRecent > (bestBefore + MIN_DELTA);
          if (!improved){
            console.log(`   ⏹ Early stopping: no meaningful improvement (Δ<=${MIN_DELTA}) over last ${EARLY_STOP_PATIENCE} epochs`);
            model.stopTraining = true;
          }
        }
      }
    }
  });

  // Dispose splits
  histTrain.dispose(); statTrain.dispose(); maskTrain.dispose(); yTrain.dispose();
  histVal.dispose();   statVal.dispose();   maskVal.dispose();   yVal.dispose();

  // Final save with last epoch metrics (if not already best)
  // --- Calibration (temperature scaling on validation) ---
  try {
    const yPredValTensor = model.predict([hist, stat, mask], { batchSize: 16 });
    const yPredArrAll = Array.from(yPredValTensor.dataSync());
    yPredValTensor.dispose();
    const yTrueArrAll = Array.from(y.dataSync());
    const maskArrAll  = Array.from(mask.dataSync());
    const yTrueArr = []; const yScoreArr = [];
    for (let i=0;i<yTrueArrAll.length;i++) if (maskArrAll[i] >= 0.5){ yTrueArr.push(yTrueArrAll[i]); yScoreArr.push(yPredArrAll[i]); }

    // Compute logits from probabilities
    const eps = 1e-7;
    const logits = yScoreArr.map(p => Math.log(Math.min(1-eps, Math.max(eps, p)) / (1 - Math.min(1-eps, Math.max(eps, p)))));

    function bceForT(T){
      let ll=0; const Te = Math.max(0.2, Math.min(5.0, T));
      for (let i=0;i<logits.length;i++){
        const z = logits[i]/Te; const p = 1/(1+Math.exp(-z)); const yv = yTrueArr[i];
        const pp = Math.min(1-eps, Math.max(eps, p));
        ll += -( yv*Math.log(pp) + (1-yv)*Math.log(1-pp) );
      }
      return ll / Math.max(1, logits.length);
    }
    let bestT = 1.0, bestLL = Infinity;
    for (let T=0.5; T<=3.0001; T+=0.05){ const ll = bceForT(T); if (ll < bestLL){ bestLL = ll; bestT = parseFloat(T.toFixed(2)); } }

    const calibration = { type: 'temperature', T: bestT, fittedOn: 'training_all_valid_masked', loss: Math.round(bestLL*1e6)/1e6 };
    await saveModel(model, { best_by: 'last', calibration, history, ...data.dataMeta }, 'model_last.json', runFolder);
    // also drop a separate calibration file for the front-end if needed
    try { fs.writeFileSync(`${runFolder}/calibration.json`, JSON.stringify(calibration, null, 2)); } catch(e) {}
  } catch (e) {
    await saveModel(model, { best_by: 'last', history, ...data.dataMeta }, 'model_last.json', runFolder);
  }

  __running = false;
  return { model, runId, runFolder };
}

module.exports = {
  buildMixedModel,
  loadData,
  saveModel,
  runTraining,
  // expose helpers
  losses: { bceLoss, topKSoftAuxLoss, combinedRaceLoss },
  metrics: { recallAtThree, precisionAtThree },
  utils: { computePerRaceEval, computeRocAuc, splitTrainValRaces },
};

if (require.main === module) {
  // Simple CLI arg parsing: --key=value supports numbers and booleans
  const args = process.argv.slice(2);
  const opts = {};
  for (const a of args) {
    const m = a.match(/^--([^=]+)=(.+)$/);
    if (m) {
      const k = m[1]; let v = m[2];
      if (v === 'true' || v === 'false') v = v === 'true';
      else if (!isNaN(Number(v))) v = Number(v);
      opts[k] = v;
    }
  }
  runTraining(opts).then(({ runId, runFolder }) => {
    console.log(`▶ Training finished. runId=${runId} folder=${runFolder}`);
  }).catch(err => {
    console.error(err);
    process.exitCode = 1;
  });
}