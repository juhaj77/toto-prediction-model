// ═══════════════════════════════════════════════════════════════════════════════
// TOTO PREDICTION MODEL — model_runner.js
//
// Reads:  training_data.json  (training)
// Writes: model-runner/model.json  (trained model weights)
//         mappings_runner.json                    (name → integer ID maps)
//
// Architecture: Mixed LSTM (history branch) + Dense (static branch)
// Target:       binary — top-3 finish (position 1–3) vs. not
// ═══════════════════════════════════════════════════════════════════════════════

'use strict';
// runner-lr1e-3-reg
let TFJS_BACKEND_LABEL = 'tfjs-node-gpu';
let tf;
(function selectTfBackend(){
  const envPref = (process.env.TFJS_BACKEND || '').toLowerCase();
  let argPref = '';
  for (const a of process.argv || []){
    const m = a.match(/^--(?:backend|tfjs)=([^=]+)$/i);
    if (m) { argPref = String(m[1]).toLowerCase(); break; }
  }
  const pref = (argPref || envPref).trim();
  const wantsGPU = ['gpu','tfjs-node-gpu','cuda','nvidia','device:gpu'].includes(pref);
  const wantsCPU = ['cpu','tfjs-node','node','device:cpu'].includes(pref);
  function tryReq(name){ try { return require(name); } catch(e){ return null; } }
  if (wantsCPU){
    tf = tryReq('@tensorflow/tfjs-node');
    if (!tf){ tf = tryReq('@tensorflow/tfjs-node-gpu'); TFJS_BACKEND_LABEL = tf ? 'tfjs-node-gpu' : 'missing'; }
    else { TFJS_BACKEND_LABEL = 'tfjs-node'; }
  } else {
    tf = tryReq('@tensorflow/tfjs-node-gpu');
    if (!tf){ tf = tryReq('@tensorflow/tfjs-node'); TFJS_BACKEND_LABEL = tf ? 'tfjs-node' : 'missing'; }
    else { TFJS_BACKEND_LABEL = 'tfjs-node-gpu'; }
  }
  if (!tf){
    throw new Error('Unable to load TensorFlow.js backend. Install @tensorflow/tfjs-node or @tensorflow/tfjs-node-gpu.');
  }
})();
const fs = require('fs');

// References
const TRAINING_DATA  = './training_data.json';
const PREDICTION_DATA = '';
const MAPPINGS_FILE  = './mappings_runner.json';
const MODEL_FOLDER   = './model-runner';
const MAX_HISTORY    = 8;
const MODEL = 'model.json';

// Configuration toggles
const USE_FOCAL_LOSS = true; // set true to use focal loss instead of BCE
const FOCAL_GAMMA = 2.0;
const FOCAL_ALPHA = 0.25;

// ─── METRICS (runner-based) ───────────────────────────────────────────────────
// ROC AUC when available in tf.metrics; otherwise return 0 as a placeholder.
function aucRoc(yTrue, yPred) {
    if (tf.metrics && typeof tf.metrics.auc === 'function') {
        // Default curve is ROC in tfjs
        return tf.metrics.auc(yTrue, yPred);
    }
    return tf.tidy(() => tf.scalar(0));
}

// Focal loss for imbalanced binary classification
function focalLoss(gamma = FOCAL_GAMMA, alpha = FOCAL_ALPHA) {
    return (yTrue, yPred) => tf.tidy(() => {
        const eps = tf.scalar(1e-7);
        const one = tf.scalar(1);
        const p = yPred.clipByValue(1e-7, 1 - 1e-7);
        const y = yTrue;
        const pt = y.mul(p).add(one.sub(y).mul(one.sub(p))); // y ? p : 1-p
        const w = y.mul(tf.scalar(alpha)).add(one.sub(y).mul(one.sub(tf.scalar(alpha))));
        const loss = w.mul(tf.pow(one.sub(pt), tf.scalar(gamma))).mul(pt.log().neg());
        return loss.mean();
    });
}

// Precision at fixed threshold (default 0.5)
function precisionAt05(yTrue, yPred) {
    return tf.tidy(() => {
        const thresh = tf.scalar(0.5);
        const yHat = yPred.greater(thresh).toFloat();
        const tp = yHat.mul(yTrue).sum();
        const predPos = yHat.sum().add(tf.scalar(1e-7));
        return tp.div(predPos);
    });
}

// Recall at fixed threshold (default 0.5)
function recallAt05(yTrue, yPred) {
    return tf.tidy(() => {
        const thresh = tf.scalar(0.5);
        const yHat = yPred.greater(thresh).toFloat();
        const tp = yHat.mul(yTrue).sum();
        const pos = yTrue.sum().add(tf.scalar(1e-7));
        return tp.div(pos);
    });
}


// ─── MODEL PERSISTENCE ────────────────────────────────────────────────────────

async function saveModel(model, meta = {}, fileName = MODEL, folder = MODEL_FOLDER) {
    const outDir = folder || MODEL_FOLDER;
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
    await model.save(tf.io.withSaveHandler(async (artifacts) => {
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
                val_auc:       meta.val_auc       ?? null,
                val_ap:        meta.val_ap        ?? null,
                // Runner model: optional race-style metrics when available
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
                calibration:   meta.calibration   ?? null,
                metrics_at_selection: meta.metrics_at_selection ?? null,
                history:       meta.history       ?? null,
                tfjs_backend:  meta.tfjs_backend  ?? TFJS_BACKEND_LABEL,
                run_options:   meta.run_options   ?? null,
            },
        };
        fs.writeFileSync(`${outDir}/${fileName}`, JSON.stringify(payload));
        return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));
}

async function loadModel() {
    const filePath = `${MODEL_FOLDER}/model.json`;
    if (!fs.existsSync(filePath)) return null;
    const saved  = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const buffer = Buffer.from(saved.weightData, 'base64');
    return tf.loadLayersModel(tf.io.fromMemory({
        modelTopology: saved.modelTopology,
        weightSpecs:   saved.weightSpecs,
        weightData:    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
    }));
}

// ─── FEATURE HELPERS ──────────────────────────────────────────────────────────

// Map track condition string → 0–1 scale (0 = heaviest, 1 = lightest)
function encodeTrackCondition(condition) {
    switch ((String(condition ?? '')).toLowerCase().trim()) {
        case 'heavy track':       return 0.00;
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

// Use surname only so "J Mäkinen" and "Juhani Mäkinen" map to the same ID
function extractSurname(fullName) {
    if (!fullName) return 'unknown';
    return fullName.trim().toLowerCase().split(' ').at(-1);
}

function parseDate(str) {
    if (!str) return null;
    if (str.includes('-')) return new Date(str);
    if (str.includes('.')) {
        const [d, m, y] = str.split('.');
        const year = parseInt(y) < 100 ? parseInt(y) + 2000 : parseInt(y);
        return new Date(year, parseInt(m) - 1, parseInt(d));
    }
    return null;
}

// Normalise km time to a 2100 m base distance.
// Correlation with finishing position: r = 0.031 → 0.090 after normalisation.
function normaliseKmTime(km, distance) {
    if (km === null || km === undefined || isNaN(km) || km <= 0) return null;
    const dist = (distance !== null && distance !== undefined && distance > 0) ? distance : 2100;
    return km + (2100 - dist) / 2000;
}

// ─── DATA LOADING ─────────────────────────────────────────────────────────────

function loadData(filePath, isTraining = true) {
    // Name → integer ID maps. Persisted to mappings.json so prediction uses
    // the same IDs as training.
    let maps = {
        coaches: {}, drivers: {}, tracks: {},
        counts: { c: 1, d: 1, t: 1 },
    };

    if (!isTraining && fs.existsSync(MAPPINGS_FILE)) {
        maps = JSON.parse(fs.readFileSync(MAPPINGS_FILE, 'utf8'));
        console.log(
            `  Loaded mappings: ${Object.keys(maps.coaches).length} coaches, ` +
            `${Object.keys(maps.drivers).length} drivers, ` +
            `${Object.keys(maps.tracks).length} tracks`
        );
    }

    // Assign a stable integer ID for each unique name.
    // In prediction mode unknown names fall back to 0.
    function getID(map, name, type) {
        if (!name || name === 'Unknown' || name === '') return 0;
        const key = (type === 'driver') ? extractSurname(name) : name.trim().toLowerCase();
        if (isTraining) {
            if (!map[key]) { map[key] = maps.counts[type[0]]; maps.counts[type[0]]++; }
            return map[key];
        }
        return map[key] || 0;
    }

    // ── Parse JSON ────────────────────────────────────────────────────────────
    const raw   = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const races = raw.races || [];

    // ── Per-breed mean imputation values ─────────────────────────────────────
    // Used when a runner's record or a historical km time is missing.

    const breedStats = { SH: { records: [], kmTimes: [] }, LV: { records: [], kmTimes: [] } };
    for (const race of races) {
        const breed = race.isColdBlood ? 'SH' : 'LV';
        for (const runner of (race.runners || [])) {
            if (runner.record > 0) breedStats[breed].records.push(runner.record);
            for (const ps of (runner.prevStarts || [])) {
                const km = normaliseKmTime(ps.kmTime, ps.distance);
                if (km > 0) breedStats[breed].kmTimes.push(km);
            }
        }
    }
    const avg = arr => arr.length ? arr.reduce((a, b) => a + b) / arr.length : null;
    const means = {
        SH: { record: avg(breedStats.SH.records) ?? 28.0, km: avg(breedStats.SH.kmTimes) ?? 29.0 },
        LV: { record: avg(breedStats.LV.records) ?? 15.0, km: avg(breedStats.LV.kmTimes) ?? 16.0 },
    };

    // ── Per-race betting/win ranking maps ─────────────────────────────────────
    // Relative rank within a race is more informative than raw percentage.
    const betRankMap  = {};   // raceId_startNumber → rank (1 = favourite)
    const winRankMap  = {};
    const rankMetaMap = {};   // → { betKnown, winKnown }

    for (const race of races) {
        const starters = (race.runners || []).filter(r => !r.scratched);
        const hasBet   = starters.some(r => (r.bettingPercentage || 0) > 0);
        const hasWin   = starters.some(r => (r.winPercentage     || 0) > 0);
        const neutral  = (starters.length + 1) / 2;

        const byBet = [...starters].sort((a, b) => (b.bettingPercentage || 0) - (a.bettingPercentage || 0));
        const byWin = [...starters].sort((a, b) => (b.winPercentage     || 0) - (a.winPercentage     || 0));

        for (const runner of starters) {
            const key      = `${race.trackID}_${runner.number}`;
            const betKnown = hasBet && (runner.bettingPercentage || 0) > 0 ? 1 : 0;
            const winKnown = hasWin && (runner.winPercentage     || 0) > 0 ? 1 : 0;
            betRankMap[key]  = betKnown ? byBet.findIndex(r => r.number === runner.number) + 1 : neutral;
            winRankMap[key]  = winKnown ? byWin.findIndex(r => r.number === runner.number) + 1 : neutral;
            rankMetaMap[key] = { betKnown, winKnown };
        }
    }

    // ── Build feature tensors ─────────────────────────────────────────────────
    const X_hist   = [];
    const X_static = [];
    const Y        = [];
    const metadata = [];
    const RACE_IDS = [];
    const raceIdMap = new Map();
    let raceSeq = 0;

    // Filter out races with no positive target (no runner finished 1–3)
    let keptRaces = 0;
    let droppedRaces = 0;
    const keptDates = [];

    for (const race of races) {
        const breed     = race.isColdBlood ? 'SH' : 'LV';
        const starters  = (race.runners || []).filter(r => !r.scratched);
        const neutral   = (starters.length + 1) / 2;

        // Skip races where none of the starters has a positive target (position 1–3)
        if (isTraining) {
            const hasPositive = starters.some(r => r && r.position != null && r.position >= 1 && r.position <= 3);
            if (!hasPositive) { droppedRaces++; continue; }
        }
        keptRaces++;
        if (race.date) keptDates.push(race.date);

        // Stable race identifier for grouping (track+date+raceNo)
        const ridKey = `${race.trackID || 'trk'}_${race.date || 'nodate'}_${race.raceNumber || race.number || keptRaces}`;
        let rid;
        if (raceIdMap.has(ridKey)) { rid = raceIdMap.get(ridKey); }
        else { rid = raceSeq; raceIdMap.set(ridKey, raceSeq); raceSeq++; }

        for (const runner of starters) {
            const rankKey  = `${race.trackID}_${runner.number}`;
            const betRank  = betRankMap[rankKey]  ?? neutral;
            const winRank  = winRankMap[rankKey]  ?? neutral;
            const rankMeta = rankMetaMap[rankKey] ?? { betKnown: 0, winKnown: 0 };

            // ── Shoe encoding ─────────────────────────────────────────────────
            const frontStr    = (runner.frontShoes  || '').toUpperCase();
            const rearStr     = (runner.rearShoes   || '').toUpperCase();
            const frontActive = frontStr === 'HAS_SHOES' ? 1 : 0;
            const frontKnown  = (frontStr === 'HAS_SHOES' || frontStr === 'NO_SHOES') ? 1 : 0;
            const rearActive  = rearStr  === 'HAS_SHOES' ? 1 : 0;
            const rearKnown   = (rearStr  === 'HAS_SHOES' || rearStr  === 'NO_SHOES') ? 1 : 0;
            //runner-lr1e-3-reg
            // ── Special cart encoding ─────────────────────────────────────────
            const cartStr    = (runner.specialCart || '').toUpperCase();
            const cartActive = cartStr === 'YES' ? 1 : 0;
            const cartKnown  = (cartStr === 'YES' || cartStr === 'NO') ? 1 : 0;

            // ── prevIndexNorm ─────────────────────────────────────────────────
            // Weighted podium score normalised by ALL starts, including
            // disqualifications (pos=20) and DNFs (pos=21) which score 0 pts.
            // This penalises horses with many non-finishes. r = 0.161 vs top-3.
            const prevStarts  = runner.prevStarts || [];
            const validPrev   = prevStarts.filter(ps => ps.date && ps.driver);
            const histKnown   = validPrev.length > 0 ? 1 : 0;
            let prevIndexNorm = 0;

            if (validPrev.length > 0) {
                let score = 0, count = 0;
                for (const ps of validPrev) {
                    const pos = ps.position;
                    if (pos == null) continue;        // null (MATLAB []) or undefined → skip
                    count++;                          // disq/DNF → 0 pts, still counted
                    if      (pos === 1) score += 1.00;
                    else if (pos === 2) score += 0.50;
                    else if (pos === 3) score += 0.33;
                }
                prevIndexNorm = count > 0 ? score / count : 0;
            }

            // ── Static features — 27 total ────────────────────────────────────
            const staticFeats = [
                (runner.number || 1) / 20,                                        // [0]  start number
                getID(maps.coaches, runner.coach,  'coach')  / 6000,         // [1]  coach ID. Currently 4649
                runner.record ? 1 : 0,                                            // is record known
                (runner.record || 0) / 50,                                         // [2]  race record (imputed if missing)
                getID(maps.drivers, runner.driver, 'driver') / 5000,         // [3]  current driver ID Currently 3759
                runner.age ? 1 : 0,
                (runner.age || 5) / 15,                                           // [4]  age
                (runner.gender || 2) / 3,                                         // [5]  gender (1=mare 2=gelding 3=stallion)
                race.isColdBlood ? 1 : 0,
                race.isColdBlood ? 1 : 0,                                         // [6]  cold blood breed
                frontActive, frontKnown,                                          // [7-8]   front shoes on / known
                rearActive,  rearKnown,                                           // [9-10]  rear shoes on / known
                runner.frontShoesChanged ? 1 : 0,                                 // [11] front shoes changed this race
                runner.rearShoesChanged  ? 1 : 0,                                 // [12] rear shoes changed this race
                (race.distance || 2100) / 3100,                                   // [13] race distance
                race.isCarStart ? 1 : 0,                                          // [14] car start (vs. line start)
                (runner.bettingPercentage || 0) / 100,                            // [15] raw betting %
                (runner.winPercentage     || 0) / 100,                            // [16] historical win %
                (runner.winPercentage     || 0) > 0 ? 1 : 0,                      // [17] win % known flag
                runner.isAutoRecord ? 1 : 0,                                      // [18] record set from car start
                cartActive, cartKnown,                                            // [19-20] special cart on / known
                betRank / 20, rankMeta.betKnown,                                  // [21-22] betting rank / known
                winRank / 20, rankMeta.winKnown,                                  // [23-24] win rank / known
                prevIndexNorm,                                                    // [25] podium index / n_all_starts
                histKnown,                                                        // [26] has prior start history
            ];

            // ── History sequence — MAX_HISTORY × 25 features ─────────────────
            const histSeq = [];

            for (let i = 0; i < MAX_HISTORY; i++) {
                const ps = validPrev[i];

                // Pad missing history slots with -1 (LSTM masking layer ignores these)
                if (!ps) { histSeq.push(new Array(25).fill(-1)); continue; }

                const raceDate  = parseDate(race.date);
                const startDate = parseDate(ps.date);
                const daysSince = (raceDate && startDate)
                    ? Math.min(365, (raceDate - startDate) / 86400000)
                    : 30;

               // const kmNorm  = normaliseKmTime(ps.kmTime, ps.distance);
                const kmKnown = ps.kmTime > 0 ? 1 : 0;
               // const kmFinal = kmKnown ? ps.kmTime / 100 : means[breed].km / 100;

                const distKnown  = (ps.distance   || 0) > 0 ? 1 : 0;
                const distFinal  = distKnown  ? (ps.distance   / 3100) : 0.67;

                const prizeKnown = (ps.firstPrice || 0) > 0 ? 1 : 0;
                const prizeFinal = prizeKnown ? Math.log1p(ps.firstPrice) / 10  : 0.55;

                const oddKnown   = (ps.odd        || 0) > 0 ? 1 : 0;
                const oddFinal   = oddKnown   ? Math.log1p(ps.odd)        / 5   : 0.50;

                const posKnown   = ps.position != null && ps.position > 0 ? 1 : 0;
                const posFinal   = posKnown ? ps.position / 20 : 0.5;

                const psfront = (ps.frontShoes  || '').toUpperCase();
                const psrear  = (ps.rearShoes   || '').toUpperCase();
                const pscart  = (ps.specialCart || '').toUpperCase();

                histSeq.push([
                    ps.kmTime / 100,    kmKnown,                                           // [0-1]   km time normalised to 2100 m
                    distFinal,  distKnown,                                         // [2-3]   distance
                    daysSince / 365,                                               // [4]     days since this start
                    posFinal,   posKnown,                                          // [5-6]   finishing position
                    prizeFinal, prizeKnown,                                        // [7-8]   prize money (log-scaled)
                    oddFinal,   oddKnown,                                          // [9-10]  win odds (log-scaled)
                    ps.isCarStart   ? 1 : 0,                                       // [11]    car start
                    ps.isGallop       ? 1 : 0,                                       // [12]    gait fault (break)
                    (ps.number ?? 1) / 30,                                         // [13]    start position
                    getID(maps.drivers, ps.driver, 'driver') / 5000,               // [14]    driver ID
                    getID(maps.tracks,  ps.track,  'track')  / 600,                // [15]    track ID. currently 492
                    ps.disqualified ? 1 : 0,                                       // [16]    disqualified
                    ps.DNF          ? 1 : 0,                                       // [17]    did not finish
                    psfront === 'HAS_SHOES' ? 1 : 0,                               // [18]    front shoes on
                    (psfront === 'HAS_SHOES' || psfront === 'NO_SHOES') ? 1 : 0,   // [19]    front shoes known
                    psrear  === 'HAS_SHOES' ? 1 : 0,                               // [20]    rear shoes on
                    (psrear  === 'HAS_SHOES' || psrear  === 'NO_SHOES') ? 1 : 0,   // [21]    rear shoes known
                    pscart  === 'YES' ? 1 : 0,                                     // [22]    special cart on
                    (pscart  === 'YES' || pscart  === 'NO') ? 1 : 0,               // [23]    special cart known
                    encodeTrackCondition(ps.trackCondition),                       // [24]    track condition
                ]);
            }

            X_hist.push(histSeq);
            X_static.push(staticFeats);
            RACE_IDS.push(rid);

            if (isTraining) {
                Y.push((runner.position != null && runner.position >= 1 && runner.position <= 3) ? 1 : 0);
            } else {
                metadata.push({
                    number: runner.number,
                    name:   runner.name,
                    driver: runner.driver,
                });
            }
        }
    }

    // Persist maps after training so prediction uses the same IDs
    if (isTraining) {
        fs.writeFileSync(MAPPINGS_FILE, JSON.stringify(maps, null, 2));
        console.log(
            `  Mappings saved: ${Object.keys(maps.coaches).length} coaches, ` +
            `${Object.keys(maps.drivers).length} drivers, ` +
            `${Object.keys(maps.tracks).length} tracks`
        );
    }

    const allDates = keptDates.length ? keptDates.slice().sort() : races.map(r => r.date).filter(Boolean).sort();
    const dataMeta = {
        totalRaces:    keptRaces > 0 ? keptRaces : races.length,
        totalRunners:  X_hist.length,
        dataStartDate: allDates[0]                   ?? null,
        dataEndDate:   allDates[allDates.length - 1] ?? null,
        droppedRacesNoTarget: droppedRaces,
    };

    if (isTraining) {
        console.log(
            `  Data: ${dataMeta.totalRunners} runners across ${dataMeta.totalRaces} races, ` +
            `${dataMeta.dataStartDate} → ${dataMeta.dataEndDate}`
        );
        if (droppedRaces > 0) {
            console.log(`  Filter: dropped ${droppedRaces} races without any valid target (no runner position > 0).`);
        }
    }

    return {
        hist:               tf.tensor3d(X_hist),
        static:             tf.tensor2d(X_static),
        y:                  isTraining ? tf.tensor2d(Y, [Y.length, 1]) : null,
        raceIds:            isTraining ? tf.tensor1d(RACE_IDS, 'int32') : null,
        metadata,
        histFeatureCount:   25,
        staticFeatureCount: 30,
        dataMeta,
    };
}

// ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
//
// History branch:  Masking → LSTM(64) → LSTM(32) → BN → Dropout(0.2)
// Static branch:   Dense(48,L2) → BN → Dense(32,L2) → Dropout(0.2)
// Combined head:   Dense(48,L2) → BN → Dropout(0.25) → Dense(24) → Dense(1, sigmoid)

function buildModel(timeSteps, histFeatures, staticFeatures, options = {}) {
    const opt = Object.assign({
        learningRate: 3e-4,
        l2: 5e-4,
        dropout: 0.2,
        outDropout: 0.25,
        swapBNtoLN: false,
        useLayerNormInStatic: false,
        runnerLstm2Units: 32,
        runnerLstm1Units: 64,
        headUnits: 48,
        midUnits: 24,
        useFocalLoss: undefined,
    }, options);
    const l2reg = tf.regularizers.l2({ l2: opt.l2 });
    const histInput   = tf.input({ shape: [timeSteps, histFeatures], name: 'history_input' });
    const staticInput = tf.input({ shape: [staticFeatures],          name: 'static_input'  });

    // History branch
    let h = tf.layers.masking({ maskValue: -1 }).apply(histInput);
    h = tf.layers.lstm({ units: opt.runnerLstm1Units, returnSequences: true,  recurrentDropout: 0.1,
        kernelRegularizer: l2reg }).apply(h);
    h = tf.layers.lstm({ units: opt.runnerLstm2Units, returnSequences: false, recurrentDropout: 0.1,
        kernelRegularizer: l2reg }).apply(h);
    h = (opt.swapBNtoLN ? tf.layers.layerNormalization() : tf.layers.batchNormalization()).apply(h);
    h = tf.layers.dropout({ rate: opt.dropout }).apply(h);

    // Static branch
    let s = tf.layers.dense({ units: 48, activation: 'relu',
        kernelRegularizer: l2reg }).apply(staticInput);
    s = (opt.useLayerNormInStatic || opt.swapBNtoLN ? tf.layers.layerNormalization() : tf.layers.batchNormalization()).apply(s);
    s = tf.layers.dense({ units: 32, activation: 'relu',
        kernelRegularizer: l2reg }).apply(s);
    s = tf.layers.dropout({ rate: opt.dropout }).apply(s);

    // Combined head
    let out = tf.layers.concatenate().apply([h, s]);
    out = tf.layers.dense({ units: opt.headUnits, activation: 'relu',
        kernelRegularizer: l2reg }).apply(out);
    out = (opt.swapBNtoLN ? tf.layers.layerNormalization() : tf.layers.batchNormalization()).apply(out);
    out = tf.layers.dropout({ rate: opt.outDropout }).apply(out);
    out = tf.layers.dense({ units: opt.midUnits, activation: 'relu' }).apply(out);
    out = tf.layers.dense({ units: 1,  activation: 'sigmoid' }).apply(out);

    const model = tf.model({ inputs: [histInput, staticInput], outputs: out });
    const lr = opt.learningRate;
    const optimizer = tf.train.adam(lr);
    const useFocal = (opt.useFocalLoss === undefined) ? (typeof USE_FOCAL_LOSS !== 'undefined' && USE_FOCAL_LOSS) : !!opt.useFocalLoss;
    model.compile({
        optimizer,
        loss: useFocal ? focalLoss() : 'binaryCrossentropy',
        metrics: ['accuracy', precisionAt05, recallAt05],
    });
    return model;
}

// ─── EVAL/UTIL HELPERS ───────────────────────────────────────────────────────

function seededRng(seed = 42) {
    // Simple LCG
    let s = Math.floor(seed) >>> 0;
    return function() {
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

// Race-grouped deterministic split: keeps all runners of a race in the same fold
function splitTrainValByRace(hist, stat, y, raceIds, valFraction = 0.1, seed = 42) {
    const rids = Array.from(raceIds.dataSync());
    const n = y.shape[0];
    const raceToIdx = new Map();
    for (let i = 0; i < n; i++) {
        const rid = rids[i];
        if (!raceToIdx.has(rid)) raceToIdx.set(rid, []);
        raceToIdx.get(rid).push(i);
    }
    const allRaces = Array.from(raceToIdx.keys());
    seededShuffle(allRaces, seed);
    const valRaceCount = Math.max(1, Math.floor(allRaces.length * valFraction));
    const valRaces = new Set(allRaces.slice(0, valRaceCount));
    const trainIdx = [];
    const valIdx = [];
    for (const [rid, idxs] of raceToIdx.entries()) {
        if (valRaces.has(rid)) valIdx.push(...idxs); else trainIdx.push(...idxs);
    }
    const idxTrain = tf.tensor1d(trainIdx, 'int32');
    const idxVal   = tf.tensor1d(valIdx, 'int32');

    const histTrain = tf.gather(hist, idxTrain);
    const statTrain = tf.gather(stat, idxTrain);
    const yTrain    = tf.gather(y, idxTrain);
    const rTrain    = tf.gather(raceIds, idxTrain);

    const histVal = tf.gather(hist, idxVal);
    const statVal = tf.gather(stat, idxVal);
    const yVal    = tf.gather(y, idxVal);
    const rVal    = tf.gather(raceIds, idxVal);

    idxTrain.dispose(); idxVal.dispose();
    return { histTrain, statTrain, yTrain, rTrain, histVal, statVal, yVal, rVal };
}

/**
 * @deprecated Prefer race-grouped split via splitTrainValByRace(...) to prevent leakage.
 * Kept for potential ad-hoc experiments that don't have raceIds available.
 */
function splitTrainValTensors(hist, stat, y, valFraction = 0.1, seed = 42) {
    const n = y.shape[0];
    const idx = Array.from({ length: n }, (_, i) => i);
    seededShuffle(idx, seed);
    const valCount = Math.max(1, Math.floor(n * valFraction));
    const valIdx = idx.slice(0, valCount);
    const trainIdx = idx.slice(valCount);

    const idxTrain = tf.tensor1d(trainIdx, 'int32');
    const idxVal   = tf.tensor1d(valIdx, 'int32');

    const histTrain = tf.gather(hist, idxTrain);
    const statTrain = tf.gather(stat, idxTrain);
    const yTrain    = tf.gather(y, idxTrain);

    const histVal = tf.gather(hist, idxVal);
    const statVal = tf.gather(stat, idxVal);
    const yVal    = tf.gather(y, idxVal);

    idxTrain.dispose(); idxVal.dispose();
    return { histTrain, statTrain, yTrain, histVal, statVal, yVal };
}

function computeClassWeightsFromLabels(yTensor) {
    const y = Array.from(yTensor.dataSync());
    let pos = 0; let neg = 0;
    for (let i = 0; i < y.length; i++) {
        const v = y[i];
        if (v >= 0.5) pos++; else neg++;
    }
    const wPos = pos > 0 ? Math.sqrt(neg / pos) : 1.0;
    const wNeg = 1.0;
    return { 0: wNeg, 1: wPos, counts: { pos, neg, wPos, wNeg } };
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function sweepMetrics(yTrueArr, yPredArr, tMin = 0.01, tMax = 0.99, step = 0.01) {
    const points = [];
    let bestF1 = -1; let bestT = 0.5; let bestP = 0; let bestR = 0;
    for (let t = tMin; t <= tMax + 1e-8; t += step) {
        let tp = 0, fp = 0, fn = 0;
        for (let i = 0; i < yTrueArr.length; i++) {
            const y = yTrueArr[i] >= 0.5 ? 1 : 0;
            const yhat = yPredArr[i] >= t ? 1 : 0;
            if (yhat === 1 && y === 1) tp++;
            else if (yhat === 1 && y === 0) fp++;
            else if (yhat === 0 && y === 1) fn++;
        }
        const precision = tp / Math.max(1, tp + fp);
        const recall    = tp / Math.max(1, tp + fn);
        const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
        points.push({ t: Number(t.toFixed(2)), p: precision, r: recall, f1 });
        if (f1 > bestF1) { bestF1 = f1; bestT = Number(t.toFixed(2)); bestP = precision; bestR = recall; }
    }
    // AP via trapezoid over recall (ensure monotonic order by recall)
    const pts = points.slice().sort((a, b) => a.r - b.r);
    let ap = 0;
    for (let i = 1; i < pts.length; i++) {
        const r0 = pts[i-1].r, r1 = pts[i].r;
        const p0 = pts[i-1].p, p1 = pts[i].p;
        ap += Math.max(0, (r1 - r0)) * ((p0 + p1) / 2);
    }
    return { prCurve: points, ap, best: { t: bestT, f1: bestF1, p: bestP, r: bestR } };
}

function estimateTemperatureScaling(yTrueArr, pArr) {
    // Avoid 0/1 extremes
    const eps = 1e-7;
    const logits = pArr.map(p => Math.log(Math.min(1 - eps, Math.max(eps, p)) / (1 - Math.min(1 - eps, Math.max(eps, p)))));
    // Grid search T in [0.5, 5.0]
    let bestT = 1.0; let bestNLL = Infinity;
    for (let T = 0.5; T <= 5.0001; T += 0.05) {
        let nll = 0;
        for (let i = 0; i < logits.length; i++) {
            const z = logits[i] / T;
            const pCal = sigmoid(z);
            const y = yTrueArr[i] >= 0.5 ? 1 : 0;
            nll += -(y * Math.log(Math.max(eps, pCal)) + (1 - y) * Math.log(Math.max(eps, 1 - pCal)));
        }
        nll /= logits.length;
        if (nll < bestNLL) { bestNLL = nll; bestT = Number(T.toFixed(2)); }
    }
    return { T: bestT, nll: bestNLL };
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
    // Ensure endpoints
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

// Per-race evaluation for runner-based tensors grouped by raceIds (int32)
function computePerRaceEvalRunner(yTrueTensor, yPredTensor, raceIdsTensor, k = 3) {
    const yT = Array.from(yTrueTensor.dataSync());
    const yP = Array.from(yPredTensor.dataSync());
    const rI = Array.from(raceIdsTensor.dataSync());
    const byRace = new Map();
    for (let i = 0; i < yT.length; i++) {
        const rid = rI[i];
        if (!byRace.has(rid)) byRace.set(rid, { labs: [], scs: [] });
        const rec = byRace.get(rid);
        rec.labs.push(yT[i]);
        rec.scs.push(yP[i]);
    }
    let sumPAtK = 0, sumRAtK = 0, sumNDCG = 0, sumHit1 = 0, sumAP = 0, sumLogLoss = 0, counted = 0;
    const eps = 1e-7;
    for (const [, rec] of byRace.entries()) {
        const labs = rec.labs;
        const scs  = rec.scs;
        const n = scs.length;
        if (n === 0) continue;
        const posCount = labs.reduce((a,b)=>a+(b>=0.5?1:0),0);
        if (posCount === 0) continue;
        const order = Array.from({length:n}, (_,i)=>i).sort((a,b)=>scs[b]-scs[a]);
        const sortedLabs = order.map(i=>labs[i]);
        const kk = Math.min(k, n);
        let tpAtK = 0; for (let i = 0; i < kk; i++) tpAtK += (sortedLabs[i] >= 0.5 ? 1 : 0);
        const pAtK = tpAtK / kk;
        const rAtK = tpAtK / Math.max(1, posCount);
        const hit1 = (sortedLabs[0] >= 0.5) ? 1 : 0;
        const log2 = (x)=>Math.log(x)/Math.log(2);
        let dcg=0; for (let i=0;i<kk;i++){ const rel = sortedLabs[i] >= 0.5 ? 1 : 0; dcg += (Math.pow(2, rel) - 1) / log2(i+2); }
        const idealRels = labs.slice().sort((a,b)=>(b-a));
        const kk2 = Math.min(kk, posCount); let ideal=0; for (let i=0;i<kk2;i++) ideal += (Math.pow(2,1)-1)/log2(i+2);
        const ndcg = dcg / Math.max(eps, ideal);
        let tp=0, apSum=0; for (let i=0;i<n;i++){ if (sortedLabs[i] >= 0.5) { tp++; apSum += tp/(i+1); } }
        const ap = apSum / Math.max(1, posCount);
        let ll = 0; for (let i=0;i<n;i++){ const y=labs[i]; const p=Math.min(1-eps, Math.max(eps, scs[i])); ll += -(y*Math.log(p) + (1-y)*Math.log(1-p)); }
        ll /= n;
        sumPAtK += pAtK; sumRAtK += rAtK; sumNDCG += ndcg; sumHit1 += hit1; sumAP += ap; sumLogLoss += ll; counted++;
    }
    if (counted === 0) {
        return { pAtK: 0, rAtK: 0, ndcgK: 0, hit1: 0, apMacro: 0, loglossRace: 0 };
    }
    return {
        pAtK: sumPAtK / counted,
        rAtK: sumRAtK / counted,
        ndcgK: sumNDCG / counted,
        hit1: sumHit1 / counted,
        apMacro: sumAP / counted,
        loglossRace: sumLogLoss / counted,
    };
}

// ─── TRAINING ─────────────────────────────────────────────────────────────────

let __running = false;
async function runTraining(opts = {}) {
    if (__running) { console.warn('runTraining (runner) is already running. Ignoring second call.'); return null; }
    __running = true;
    const startedAt = new Date();
    const runId = opts.runId || `${startedAt.toISOString().replace(/[:.]/g,'-')}-${Math.random().toString(36).slice(2,7)}`;
    const runFolder = `${MODEL_FOLDER}/runs/${runId}`;
    const options = Object.assign({
        trainingFile: TRAINING_DATA,
        epochs: 50,
        batchSize: 64,
        learningRate: 3e-4,
        minLearningRate: 3e-5,
        scheduler: 'cosine',
        warmupEpochs: 4,
        earlyStopPatience: 16,
        plateauPatience: 5,
        plateauFactor: 0.5,
        l2: 5e-4,
        dropout: 0.2,
        outDropout: 0.25,
        swapBNtoLN: false,
        useLayerNormInStatic: false,
        runnerLstm1Units: 64,
        runnerLstm2Units: 32,
        headUnits: 48,
        midUnits: 24,
        useFocalLoss: undefined,
        bestBy: 'val_auc',
    }, opts);

    console.log(`--- TRAINING (runner-based) runId=${runId} ---`);
    if (!fs.existsSync(runFolder)) fs.mkdirSync(runFolder, { recursive: true });

    const data  = loadData(options.trainingFile, true);
    const model = buildModel(MAX_HISTORY, data.histFeatureCount, data.staticFeatureCount, options);

    console.log(`  Tensor shapes — hist: ${data.hist.shape}  static: ${data.static.shape}  y: ${data.y.shape}`);
    console.log('  Metrikat: AUC(ROC) = erotteleva kyky kaikilla kynnyksillä\n' +
                '            Precision@0.5 = TP / (TP + FP) kynnyksellä 0.5\n' +
                '            Recall@0.5 = TP / (TP + FN) kynnyksellä 0.5.');

    // Deterministic split (runner-level; for race-grouped, extend loader)
    const { histTrain, statTrain, yTrain, rTrain, histVal, statVal, yVal, rVal } = splitTrainValByRace(
        data.hist, data.static, data.y, data.raceIds, 0.1, 1337
    );

    // Dynamic class weights from the TRAIN set
    const cw = computeClassWeightsFromLabels(yTrain);
    console.log(`  Class weights (dynamic): neg=${cw[0].toFixed(3)} pos=${cw[1].toFixed(3)} | counts pos=${cw.counts.pos} neg=${cw.counts.neg}`);

    let bestValLoss     = Infinity;
    let bestValAuc      = -Infinity;
    let bestValAp       = -Infinity;
    let bestByApMeta    = null;
    let lossPatience    = 0;   // patience for val_loss
    let aucPatience     = 0;   // patience for val_auc
    const EARLY_STOP_PATIENCE = 16;
    const LR_REDUCE_PATIENCE  = 5;
    let epochStart;
    const history = [];

    await model.fit([histTrain, statTrain], yTrain, {
        epochs:          options.epochs,
        batchSize:       options.batchSize,
        shuffle:         true,
        classWeight:     (options.useFocalLoss === undefined ? (typeof USE_FOCAL_LOSS !== 'undefined' && USE_FOCAL_LOSS) : !!options.useFocalLoss) ? undefined : { 0: cw[0], 1: cw[1] },
        validationData:  [[histVal, statVal], yVal],
        callbacks: {
            onEpochBegin: async (epoch) => {
                epochStart = Date.now();
                if (options.scheduler === 'cosine'){
                    const baseLR = options.learningRate;
                    const minLR  = options.minLearningRate;
                    const totalE = options.epochs;
                    const warmup = Math.max(0, Math.min(options.warmupEpochs, totalE-1));
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

                // Compute manual ROC-AUC on full validation set (avoid unstable batch AUC)
                // We will compute it after predicting on the full validation set below.
                let aucValNum = NaN;
                const p05ValNum = Number(
                    logs.val_precisionAt05 || logs.val_precision_at_05 || logs.val_precision || 0
                );
                const r05ValNum = Number(
                    logs.val_recallAt05 || logs.val_recall_at_05 || logs.val_recall || 0
                );

                // Predict on validation to compute PR/AP, per-race metrics and threshold sweep
                const yPredValTensor = model.predict([histVal, statVal], { batchSize: 1024 });

                // Per-race metrics (grouped by race IDs)
                const perRace = computePerRaceEvalRunner(yVal, yPredValTensor, rVal, 3);

                // Arrays for PR/AUC/threshold computations
                const yPredVal = Array.from(yPredValTensor.dataSync());
                const yTrueVal = Array.from(yVal.dataSync());

                const sweep = sweepMetrics(yTrueVal, yPredVal, 0.01, 0.99, 0.01);
                const apVal = sweep.ap;
                const tStar = sweep.best.t;
                const calib = estimateTemperatureScaling(yTrueVal, yPredVal);
                const { auc: rocAuc } = computeRocAuc(yTrueVal, yPredVal, 0.01);
                aucValNum = rocAuc;

                // Cleanup prediction tensor
                yPredValTensor.dispose();

                const aucv = aucValNum.toFixed(4);
                const p05v = p05ValNum.toFixed(4);
                const r05v = r05ValNum.toFixed(4);
                const apv  = apVal.toFixed(4);
                const ndcg3v = perRace.ndcgK.toFixed(4);
                const hit1v  = perRace.hit1.toFixed(4);
                const apmv   = perRace.apMacro.toFixed(4);
                const llrv   = perRace.loglossRace.toFixed(4);

                const histItem = {
                    epoch: epoch + 1,
                    loss: logs.loss,
                    val_loss: logs.val_loss,
                    val_acc: Number(logs.val_acc || logs.val_accuracy || 0),
                    val_auc: aucValNum,
                    val_ap:  apVal,
                    val_p05: p05ValNum,
                    val_ndcg3: perRace.ndcgK,
                    val_hit1:  perRace.hit1,
                    val_ap_macro: perRace.apMacro,
                    val_logloss_race: perRace.loglossRace,
                    val_best_threshold: tStar,
                    val_f1_best: sweep.best.f1,
                    recommended_threshold: tStar,
                    calibration: { type: 'temperature', T: calib.T },
                    pr_curve: sweep.prCurve, // optional; remove if file size is a concern
                    lr,
                };
                history.push(histItem);

                console.log(
                    `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} | ` +
                    `val_loss=${logs.val_loss.toFixed(4)} | val_acc=${acc} | val_auc=${aucv} | val_ap=${apv} | ` +
                    `val_p@0.5=${p05v} | val_r@0.5=${r05v} | ndcg@3=${ndcg3v} | hit@1=${hit1v} | AP_macro=${apmv} | race_logloss=${llrv} | ` +
                    `t*=${tStar} | T=${calib.T} | lr=${lr.toFixed(6)} | ` +
                    `${Math.floor(ms / 60000)}m ${((ms % 60000) / 1000).toFixed(0)}s`
                );

                // Save by val_loss
                if (logs.val_loss < bestValLoss) {
                    bestValLoss = logs.val_loss;
                    lossPatience = 0;
                    console.log('   ⭐ New best val_loss — saving...');
                    await saveModel(model, {
                        epoch:         epoch + 1,
                        loss:          Math.round(logs.loss * 10000) / 10000,
                        val_loss:      Math.round(logs.val_loss * 10000) / 10000,
                        val_acc:       Math.round((logs.val_acc || logs.val_accuracy || 0) * 10000) / 10000,
                        val_auc:       Math.round(aucValNum * 10000) / 10000,
                        val_ap:        Math.round(apVal * 10000) / 10000,
                        val_ndcg3:     Math.round(perRace.ndcgK * 10000) / 10000,
                        val_hit1:      Math.round(perRace.hit1 * 10000) / 10000,
                        val_ap_macro:  Math.round(perRace.apMacro * 10000) / 10000,
                        val_logloss_race: Math.round(perRace.loglossRace * 10000) / 10000,
                        val_best_threshold: tStar,
                        val_f1_best:   sweep.best.f1,
                        best_by:       'val_loss',
                        learningRate:  lr,
                        recommended_threshold: tStar,
                        calibration:   { type: 'temperature', T: calib.T },
                        metrics_at_selection: { f1_at_t: sweep.best.f1, p_at_t: sweep.best.p, r_at_t: sweep.best.r },
                        history,
                        run_options:   options,
                        tfjs_backend:  TFJS_BACKEND_LABEL,
                        ...data.dataMeta,
                    }, MODEL, runFolder);
                } else {
                    lossPatience++;
                }

                // Save by val_auc
                if (aucValNum > bestValAuc) {
                    bestValAuc = aucValNum;
                    aucPatience = 0;
                    console.log('   🌟 New best val_auc — saving...');
                    await saveModel(model, {
                        epoch:         epoch + 1,
                        loss:          Math.round(logs.loss * 10000) / 10000,
                        val_loss:      Math.round(logs.val_loss * 10000) / 10000,
                        val_acc:       Math.round((logs.val_acc || logs.val_accuracy || 0) * 10000) / 10000,
                        val_auc:       Math.round(aucValNum * 10000) / 10000,
                        val_ap:        Math.round(apVal * 10000) / 10000,
                        val_ndcg3:     Math.round(perRace.ndcgK * 10000) / 10000,
                        val_hit1:      Math.round(perRace.hit1 * 10000) / 10000,
                        val_ap_macro:  Math.round(perRace.apMacro * 10000) / 10000,
                        val_logloss_race: Math.round(perRace.loglossRace * 10000) / 10000,
                        val_best_threshold: tStar,
                        val_f1_best:   sweep.best.f1,
                        best_by:       'val_auc',
                        learningRate:  lr,
                        recommended_threshold: tStar,
                        calibration:   { type: 'temperature', T: calib.T },
                        metrics_at_selection: { f1_at_t: sweep.best.f1, p_at_t: sweep.best.p, r_at_t: sweep.best.r },
                        history,
                        run_options:   options,
                        tfjs_backend:  TFJS_BACKEND_LABEL,
                        ...data.dataMeta,
                    }, 'model_best_auc.json', runFolder);
                } else {
                    aucPatience++;
                }

                // Save by AP (AUC-PR)
                if (apVal > bestValAp) {
                    bestValAp = apVal;
                    console.log('   💠 New best AP — saving...');
                    bestByApMeta = {
                        epoch:         epoch + 1,
                        loss:          Math.round(logs.loss * 10000) / 10000,
                        val_loss:      Math.round(logs.val_loss * 10000) / 10000,
                        val_acc:       Math.round((logs.val_acc || logs.val_accuracy || 0) * 10000) / 10000,
                        val_auc:       Math.round(aucValNum * 10000) / 10000,
                        val_ap:        Math.round(apVal * 10000) / 10000,
                        val_ndcg3:     Math.round(perRace.ndcgK * 10000) / 10000,
                        val_hit1:      Math.round(perRace.hit1 * 10000) / 10000,
                        val_ap_macro:  Math.round(perRace.apMacro * 10000) / 10000,
                        val_logloss_race: Math.round(perRace.loglossRace * 10000) / 10000,
                        val_best_threshold: tStar,
                        val_f1_best:   sweep.best.f1,
                        best_by:       'val_ap',
                        learningRate:  lr,
                        recommended_threshold: tStar,
                        calibration:   { type: 'temperature', T: calib.T },
                        metrics_at_selection: { f1_at_t: sweep.best.f1, p_at_t: sweep.best.p, r_at_t: sweep.best.r },
                        history,
                        ...data.dataMeta,
                    };
                    await saveModel(model, { ...bestByApMeta, run_options: options, tfjs_backend: TFJS_BACKEND_LABEL }, 'model_best_ap.json', runFolder);
                }

                if (aucPatience >= LR_REDUCE_PATIENCE || lossPatience >= LR_REDUCE_PATIENCE) {
                    model.optimizer.learningRate = lr * 0.5;
                    console.log(`   📉 Reducing LR (plateau): ${(lr * 0.5).toFixed(6)}`);
                    aucPatience = 0;
                    lossPatience = 0;
                }

                if (lossPatience >= EARLY_STOP_PATIENCE && aucPatience >= EARLY_STOP_PATIENCE) {
                    console.log('--- Early stopping (loss & AUC plateau) ---');
                    model.stopTraining = true;
                }
            },
        },
    });

    // Dispose tensors
    data.hist.dispose(); data.static.dispose(); data.y.dispose(); data.raceIds.dispose();
    histTrain.dispose(); statTrain.dispose(); yTrain.dispose(); rTrain.dispose();
    histVal.dispose();   statVal.dispose();   yVal.dispose();   rVal.dispose();
}

// ─── PREDICTION ───────────────────────────────────────────────────────────────
/*
predictionFile should be the data for a single race in JSON format
(structured the same way as the training data).
This function is a leftover from the prototyping phase,
when the front-end was not yet handling the prediction logic.

async function runPrediction(predictionFile) {
    console.log('\n--- PREDICTION ---');
    const model = await loadModel();
    if (!model) { console.log('No saved model found. Run training first.'); return; }

    const data = loadData(predictionFile || PREDICTION_DATA, false);
    if (data.hist.shape[0] === 0) { console.log('No prediction data found.'); return; }

    const scores = await model.predict([data.hist, data.static]).data();

    const results = data.metadata
        .map((m, i) => ({
            No:          m.number,
            Horse:       m.name,
            Driver:      m.driver,
            Probability: (scores[i] * 100).toFixed(1) + '%',
            ImpliedOdds: (1 / scores[i]).toFixed(2),
            Signal:      scores[i] > 0.5 ? 'BET' : 'SKIP',
        }))
        .sort((a, b) => parseFloat(b.Probability) - parseFloat(a.Probability));

    console.table(results);

    data.hist.dispose();
    data.static.dispose();
}
*/
// ─── ENTRY POINT ──────────────────────────────────────────────────────────────
/*
(async () => {
    await runTraining();
    await runPrediction();
})();
*/
module.exports = { runTraining };

if (require.main === module) {
  const opts = {};
  for (const a of process.argv.slice(2)){
    const m = a.match(/^--([^=]+)=(.*)$/);
    if (m){
      const k = m[1]; let v = m[2];
      if (v === 'true') v = true; else if (v === 'false') v = false; else if (!isNaN(Number(v))) v = Number(v);
      opts[k] = v;
    }
  }
  runTraining(opts).catch(err => { console.error(err); process.exit(1); });
}
