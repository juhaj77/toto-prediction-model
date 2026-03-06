// ═══════════════════════════════════════════════════════════════════════════════
// TOTO PREDICTION MODEL — model_race.js
//
// Reads: training_data.json  (training)
// Writes: model-race/model.json  (trained model weights)
//         mappings.json                   (shared with runner-based model)
//
// Architecture: Race-based — TimeDistributed LSTM + Dense branches,
//               Multi-Head Attention over the runner dimension.
//
// Key difference from model_runner.js (runner-based):
//   Tensors include the full race field as one unit so the model can learn
//   within-race relationships between runners. Betting/win rank features are
//   omitted — the attention mechanism learns race-relative context instead.
//
// Tensor shapes:
//   History:  [n_races, MAX_RUNNERS, 8, 25]
//   Static:   [n_races, MAX_RUNNERS, 25]   (25 features, no ranking)
//   Output:   [n_races, MAX_RUNNERS, 1]    sigmoid per runner slot
// ═══════════════════════════════════════════════════════════════════════════════

'use strict';

const tf = require('@tensorflow/tfjs');
const { MultiHeadAttention } = require('./MultiHeadAttention.js')
const fs = require('fs');

const TRAINING_DATA = './training_data.json';
const MAPPINGS_FILE = './mappings_race.json';
const MODEL_FOLDER  = './model-race';
const MODEL = 'model.json';
const MAX_HISTORY   = 8;
const MAX_RUNNERS   = 18;   // maximum field size — smaller fields are zero-padded

// Custom metric: Recall@3 per race (how many actual top-3s captured in top-3 predictions)
function recallAtThree(yTrue, yPred) {
    return tf.tidy(() => {
        const yT = yTrue.squeeze([-1]); // [b, r] // yTrue, yPred shapes: [batch, maxRunners, 1]
        const yP = yPred.squeeze([-1]); // [b, r]
        const topk = tf.topk(yP, 3, true);// Top-3 indices per race
        const idx  = topk.indices; // [b, 3]
        const numRunners = yP.shape[1];// Build one-hot mask for top-3 predictions per race
        const oneHot = tf.oneHot(idx, numRunners); // [b, 3, r]
        const top3Mask = oneHot.sum(1);  // [b, r]  (1 at top-3 positions)

        const captured = yT.mul(top3Mask).sum(1);  // [b]// True positives captured in top-3
        const denom = yT.sum(1);  // Denominator = number of actual positives per race (avoid div by zero)                 // [b]
        const safeDenom = denom.add(denom.equal(0).toFloat()); // add 1 where denom==0
        const perExampleRecall = captured.div(safeDenom); // [b]
        const meanRecall = perExampleRecall.mean();
        return meanRecall;
    });
}

// Custom metric: Precision@3 per race (how many of the predicted top-3 are truly positive)
function precisionAtThree(yTrue, yPred) {
    return tf.tidy(() => {
        const yT = yTrue.squeeze([-1]); // [b, r]
        const yP = yPred.squeeze([-1]); // [b, r]
        const topk = tf.topk(yP, 3, true);
        const idx  = topk.indices; // [b, 3]
        const numRunners = yP.shape[1];
        const oneHot = tf.oneHot(idx, numRunners); // [b, 3, r]
        const top3Mask = oneHot.sum(1); // [b, r]

        const truePos = yT.mul(top3Mask).sum(1);   // [b]
        const precisionPerRace = truePos.div(tf.scalar(3)); // [b]
        return precisionPerRace.mean();
    });
}

// Optional metric: ROC AUC (falls back to 0 if tf.metrics.auc is not available)
function aucRoc(yTrue, yPred) {
    if (tf.metrics && typeof tf.metrics.auc === 'function') {
        return tf.metrics.auc(yTrue, yPred);
    }
    return tf.tidy(() => tf.scalar(0));
}

// ─── LOSSES (race-based) ─────────────────────────────────────────────────────
// Binary cross-entropy implemented explicitly (robust across tfjs versions)
function bceLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const eps = tf.scalar(1e-7);
        const one = tf.scalar(1);
        const p = yPred.clipByValue(1e-7, 1 - 1e-7);
        const term1 = yTrue.mul(p.log());
        const term2 = one.sub(yTrue).mul(one.sub(p).log());
        const loss = term1.add(term2).neg().mean();
        return loss;
    });
}

// Soft Top‑K (K=3) auxiliary loss per race (mask‑free, differentiable).
// Builds a soft target over runner slots and compares to a softmax over logits
// derived from the model's sigmoid outputs. Avoids non‑diff ops (no Greater).
function topKSoftAuxLoss(yTrue, yPred, K = 3, smooth = 0.1) {
    return tf.tidy(() => {
        // shapes: [batch, maxRunners, 1]
        const yT = yTrue.squeeze([-1]);          // [b, r]
        // Use clipped probabilities, then convert to logits for stable softmax
        const p  = yPred.squeeze([-1]).clipByValue(1e-7, 1 - 1e-7); // [b, r]
        const logits = p.log().sub(tf.scalar(1).sub(p).log());       // logit(p) = log(p/(1-p))

        // Prediction distribution over runner slots (implicit masking via logits≈-inf for padded)
        const q = tf.softmax(logits, 1);                              // [b, r]

        // Target distribution: normalized positives with label smoothing.
        // Distribute smoothing mass using a soft uniform derived from p itself (no thresholds),
        // and stop gradients if available so targets don't backprop through p.
        const pos = yT;                                               // [b, r]
        const posDen = pos.sum(1, true).add(tf.scalar(1e-6));         // [b, 1]
        const tHard = pos.div(posDen);                                // [b, r]
        let uRaw = p.div(p.sum(1, true).add(tf.scalar(1e-6)));        // [b, r]
        if (tf.stopGradient) uRaw = tf.stopGradient(uRaw);
        const t = tHard.mul(tf.scalar(1 - smooth)).add(uRaw.mul(tf.scalar(smooth)));

        // Cross-entropy between t and q
        const ce = t.mul(q.add(tf.scalar(1e-7)).log()).sum(1).neg().mean();
        return ce;
    });
}

const AUX_LOSS_WEIGHT = 0.4; // tune 0.3–0.5
function combinedRaceLoss(weight = AUX_LOSS_WEIGHT) {
    return (yTrue, yPred) => tf.tidy(() => {
        const b = bceLoss(yTrue, yPred);
        const a = topKSoftAuxLoss(yTrue, yPred, 3, 0.1);
        return b.mul(tf.scalar(1 - weight)).add(a.mul(tf.scalar(weight)));
    });
}


// ─── MODEL PERSISTENCE ────────────────────────────────────────────────────────

async function saveModel(model, meta = {}, fileName = MODEL) {
    if (!fs.existsSync(MODEL_FOLDER)) fs.mkdirSync(MODEL_FOLDER);
    await model.save(tf.io.withSaveHandler(async (artifacts) => {
        const payload = {
            modelTopology: artifacts.modelTopology,
            weightSpecs:   artifacts.weightSpecs,
            weightData:    Buffer.from(artifacts.weightData).toString('base64'),
            trainingInfo: {
                savedAt:       new Date().toISOString(),
                epoch:         meta.epoch         ?? null,
                loss:          meta.loss          ?? null,
                val_loss:      meta.val_loss       ?? null,
                val_acc:       meta.val_acc        ?? null,
                val_r3:        meta.val_r3        ?? null,
                val_p3:        meta.val_p3        ?? null,
                val_auc:       meta.val_auc       ?? null,
                val_ndcg3:     meta.val_ndcg3     ?? null,
                val_hit1:      meta.val_hit1      ?? null,
                val_ap_macro:  meta.val_ap_macro  ?? null,
                val_logloss_race: meta.val_logloss_race ?? null,
                val_best_threshold: meta.val_best_threshold ?? null,
                val_f1_best:   meta.val_f1_best   ?? null,
                best_by:       meta.best_by       ?? 'val_loss',
                learningRate:  meta.learningRate   ?? null,
                dataStartDate: meta.dataStartDate  ?? null,
                dataEndDate:   meta.dataEndDate    ?? null,
                totalRaces:    meta.totalRaces     ?? null,
                totalRunners:  meta.totalRunners   ?? null,
                history:       meta.history        ?? null,
            },
        };
        fs.writeFileSync(`${MODEL_FOLDER}/${fileName}`, JSON.stringify(payload));
        return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));
}

// ─── FEATURE HELPERS ──────────────────────────────────────────────────────────

function encodeTrackCondition(condition) {
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

function normaliseKmTime(km, distance) {
    if (km === null || km === undefined || isNaN(km) || km <= 0) return null;
    const dist = (distance !== null && distance !== undefined && distance > 0) ? distance : 2100;
    return km + (2100 - dist) / 2000;
}

// ─── DATA LOADING ─────────────────────────────────────────────────────────────

function loadData(filePath) {
    // Load or initialise name → integer ID maps.
    // mappings.json is shared with the runner-based model so driver/track IDs
    // are consistent across both architectures.
    let maps = {
        coaches: {}, drivers: {}, tracks: {},
        counts: { c: 1, d: 1, t: 1 },
    };

    function getID(map, name, type) {
        if (!name || name === 'Unknown' || name === '') return 0;
        const key = (type === 'driver') ? extractSurname(name) : name.trim().toLowerCase();
        if (!map[key]) { map[key] = maps.counts[type[0]]; maps.counts[type[0]]++; }
        return map[key];
    }

    const raw   = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const races = raw.races || [];

    // Per-breed mean imputation values
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
    let maxRunners = 0;
    for(let i = 0; i < races.length; i++)
        if(races[i].runners.length > maxRunners) maxRunners = races[i].runners.length;
    console.log(`  Loaded ${races.length} races, calc max ${maxRunners} runners/race, in code: const MAX_RUNNERS = ${MAX_RUNNERS};`);

    const avg = arr => arr.length ? arr.reduce((a, b) => a + b) / arr.length : null;
    const means = {
        SH: { record: avg(breedStats.SH.records) ?? 28.0, km: avg(breedStats.SH.kmTimes) ?? 29.0 },
        LV: { record: avg(breedStats.LV.records) ?? 15.0, km: avg(breedStats.LV.kmTimes) ?? 16.0 },
    };

    // ── Build race-level tensors ───────────────────────────────────────────────
    // Each race produces one row: [MAX_RUNNERS, 8, 25] history and [MAX_RUNNERS, 25] static.
    // Shorter fields are padded to MAX_RUNNERS with zeros (static) / -1 rows (history).

    const X_hist   = [];   // [n_races, MAX_RUNNERS, 8, 25]
    const X_static = [];   // [n_races, MAX_RUNNERS, 25]
    const Y        = [];   // [n_races, MAX_RUNNERS, 1]
    const X_mask   = [];   // [n_races, MAX_RUNNERS, 1]

    // Filter out races where no runner has a valid position (> 0)
    let keptRaces = 0;
    let keptRunners = 0;
    let droppedRaces = 0;
    const keptDates = [];

    for (const race of races) {
        const breed    = race.isColdBlood ? 'SH' : 'LV';
        const starters = (race.runners || []).filter(r => !r.scratched);

        // Skip races where none of the starters has a positive target (position 1–3)
        const hasPositive = starters.some(r => r && r.position != null && r.position >= 1 && r.position <= 3);
        if (!hasPositive) { droppedRaces++; continue; }
        keptRaces++;
        keptRunners += starters.length;
        if (race.date) keptDates.push(race.date);

        const raceHist   = [];   // [MAX_RUNNERS, 8, 25]
        const raceStatic = [];   // [MAX_RUNNERS, 25]
        const raceY      = [];   // [MAX_RUNNERS, 1]
        const raceMask   = [];   // [MAX_RUNNERS, 1]  — 1 for real runner, 0 for padding

        for (let slot = 0; slot < MAX_RUNNERS; slot++) {
            const runner = starters[slot];

            if (!runner) {
                // Padding slot
                raceStatic.push(new Array(25).fill(0));
                raceHist.push(Array.from({ length: MAX_HISTORY }, () => new Array(25).fill(-1)));
                raceY.push([0]);   // padding slots never count as top-3
                raceMask.push([0]);
                continue;
            }
            // Real runner slot
            raceMask.push([1]);

            // ── Shoe encoding ─────────────────────────────────────────────────
            const frontStr    = (runner.frontShoes  || '').toUpperCase();
            const rearStr     = (runner.rearShoes   || '').toUpperCase();
            const frontActive = frontStr === 'HAS_SHOES' ? 1 : 0;
            const frontKnown  = (frontStr === 'HAS_SHOES' || frontStr === 'NO_SHOES') ? 1 : 0;
            const rearActive  = rearStr  === 'HAS_SHOES' ? 1 : 0;
            const rearKnown   = (rearStr  === 'HAS_SHOES' || rearStr  === 'NO_SHOES') ? 1 : 0;

            // ── Special cart encoding ─────────────────────────────────────────
            const cartStr    = (runner.specialCart || '').toUpperCase();
            const cartActive = cartStr === 'YES' ? 1 : 0;
            const cartKnown  = (cartStr === 'YES' || cartStr === 'NO') ? 1 : 0;

            // ── prevIndexNorm ─────────────────────────────────────────────────
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

            // ── Static features — 25 total (no betting/win rank) ──────────────
            // Ranking features [21-24] from the runner-based model are omitted here.
            // The attention mechanism learns race-relative context instead.
            raceStatic.push([
                (runner.number || 1) / 20,                                        // [0]  start number
                getID(maps.coaches, runner.coach,  'coach')  / 6000,              // [1]  coach ID
                (runner.record || means[breed].record) / 50,                      // [2]  race record (imputed if missing)
                getID(maps.drivers, runner.driver, 'driver') / 5000,              // [3]  current driver ID
                (runner.age || 5) / 15,                                           // [4]  age
                (runner.gender || 2) / 3,                                         // [5]  gender (1=mare 2=gelding 3=stallion)
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
                prevIndexNorm,                                                    // [21] podium index / n_all_starts
                histKnown,                                                        // [22] has prior start history
                // [23-24] reserved / future use — kept as zeros for now so
                // staticFeatureCount stays at 25 and shapes remain clean
                0, 0,
            ]);

            // ── History sequence — MAX_HISTORY × 25 features ─────────────────
            const histSeq = [];
            for (let i = 0; i < MAX_HISTORY; i++) {
                const ps = validPrev[i];
                if (!ps) { histSeq.push(new Array(25).fill(-1)); continue; }

                const raceDate  = parseDate(race.date);
                const startDate = parseDate(ps.date);
                const daysSince = (raceDate && startDate)
                    ? Math.min(365, (raceDate - startDate) / 86400000)
                    : 30;

                const kmNorm  = normaliseKmTime(ps.kmTime, ps.distance);
                const kmKnown = kmNorm > 0 ? 1 : 0;
                const kmFinal = kmKnown ? kmNorm / 100 : means[breed].km / 100;

                const distKnown  = (ps.distance   || 0) > 0 ? 1 : 0;
                const distFinal  = distKnown ? ps.distance / 3100 : 0.67;

                // Normalise prize in case of mixed scales in historical data
                let prizeVal = parseFloat(ps.firstPrice) || 0;
                if (prizeVal > 200000)      prizeVal = prizeVal / 10000; // e.g. 1_000_000 → 100 €
                else if (prizeVal > 2000)    prizeVal = prizeVal / 100;   // e.g. 10_000   → 100 €
                const prizeKnown = prizeVal > 0 ? 1 : 0;
                const prizeFinal = prizeKnown ? Math.log1p(prizeVal) / 10 : 0.55;

                const oddKnown   = (ps.odd        || 0) > 0 ? 1 : 0;
                const oddFinal   = oddKnown ? Math.log1p(ps.odd) / 5 : 0.50;

                const posKnown   = ps.position != null && ps.position > 0 ? 1 : 0;
                const posFinal   = posKnown ? ps.position / 20 : 0.5;

                const psfront = (ps.frontShoes  || '').toUpperCase();
                const psrear  = (ps.rearShoes   || '').toUpperCase();
                const pscart  = (ps.specialCart || '').toUpperCase();

                histSeq.push([
                    kmFinal,    kmKnown,                                           // [0-1]
                    distFinal,  distKnown,                                         // [2-3]
                    daysSince / 365,                                               // [4]
                    posFinal,   posKnown,                                          // [5-6]
                    prizeFinal, prizeKnown,                                        // [7-8]
                    oddFinal,   oddKnown,                                          // [9-10]
                    ps.isCarStart   ? 1 : 0,                                       // [11]
                    ps.isGallop        ? 1 : 0,                                       // [12]
                    (ps.number ?? 1) / 30,                                         // [13]
                    getID(maps.drivers, ps.driver, 'driver') / 5000,               // [14]
                    getID(maps.tracks,  ps.track,  'track')  / 600,                // [15]
                    ps.disqualified ? 1 : 0,                                       // [16]
                    ps.DNF          ? 1 : 0,                                       // [17]
                    psfront === 'HAS_SHOES' ? 1 : 0,                               // [18]
                    (psfront === 'HAS_SHOES' || psfront === 'NO_SHOES') ? 1 : 0,   // [19]
                    psrear  === 'HAS_SHOES' ? 1 : 0,                               // [20]
                    (psrear  === 'HAS_SHOES' || psrear  === 'NO_SHOES') ? 1 : 0,   // [21]
                    pscart  === 'YES' ? 1 : 0,                                     // [22]
                    (pscart  === 'YES' || pscart  === 'NO') ? 1 : 0,               // [23]
                    encodeTrackCondition(ps.trackCondition),                       // [24]
                ]);
            }

            raceHist.push(histSeq);
            raceY.push([(runner.position != null && runner.position >= 1 && runner.position <= 3) ? 1 : 0]);
        }

        X_hist.push(raceHist);
        X_static.push(raceStatic);
        Y.push(raceY);
        X_mask.push(raceMask);
    }

    fs.writeFileSync(MAPPINGS_FILE, JSON.stringify(maps, null, 2));
    console.log(
        `  Mappings saved: ${Object.keys(maps.coaches).length} coaches, ` +
        `${Object.keys(maps.drivers).length} drivers, ` +
        `${Object.keys(maps.tracks).length} tracks`
    );

    const allDates = keptDates.length ? keptDates.slice().sort() : races.map(r => r.date).filter(Boolean).sort();
    const dataMeta = {
        totalRaces:    keptRaces > 0 ? keptRaces : races.length,
        totalRunners:  keptRunners > 0 ? keptRunners : races.reduce((n, r) => n + (r.runners || []).filter(ru => !ru.scratched).length, 0),
        dataStartDate: allDates[0]                   ?? null,
        dataEndDate:   allDates[allDates.length - 1] ?? null,
        droppedRacesNoTarget: droppedRaces,
    };
    console.log(
        `  Data: ${dataMeta.totalRunners} runners across ${dataMeta.totalRaces} races, ` +
        `${dataMeta.dataStartDate} → ${dataMeta.dataEndDate}`
    );
    if (droppedRaces > 0) {
        console.log(`  Filter: dropped ${droppedRaces} races without any valid target (no runner position > 0).`);
    }

    // Y shape: [n_races, MAX_RUNNERS, 1]
    return {
        hist:               tf.tensor4d(X_hist),
        static:             tf.tensor3d(X_static),
        mask:               tf.tensor3d(X_mask),
        y:                  tf.tensor3d(Y),
        histFeatureCount:   25,
        staticFeatureCount: 25,
        dataMeta,
    };
}

// ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
//
// History branch:   TimeDistributed(Masking → LSTM(64) → LSTM(32) → BN → Dropout)
// Static branch:    TimeDistributed(Dense(48) → BN → Dense(32) → Dropout)
// Merge:            TimeDistributed(Dense(64) → BN)
// Attention:        MultiHeadAttention(4 heads, keyDim=16) + residual + LayerNorm
// Output head:      TimeDistributed(Dropout → Dense(24) → Dense(1, sigmoid))
//
// Input shapes  (batch dimension implicit):
//   history_input:  [MAX_RUNNERS, 8, 25]
//   static_input:   [MAX_RUNNERS, 25]
// Output shape:
//   [MAX_RUNNERS, 1]  — top-3 probability per runner slot

function buildModel(maxRunners, timeSteps, histFeatures, staticFeatures) {
    const histInput   = tf.input({ shape: [maxRunners, timeSteps, histFeatures], name: 'history_input' });
    const staticInput = tf.input({ shape: [maxRunners, staticFeatures],          name: 'static_input'  });
    const maskInput   = tf.input({ shape: [maxRunners, 1],                       name: 'mask_input'    });

    // ── History branch ────────────────────────────────────────────────────────
    // TimeDistributed runs the same LSTM stack independently for each runner slot.

    let h = tf.layers.timeDistributed({
        layer: tf.layers.masking({ maskValue: -1 }),
        name: 'hist_masking',
    }).apply(histInput);

    h = tf.layers.timeDistributed({
        layer: tf.layers.lstm({
            units: 64,
            returnSequences: true,
            recurrentDropout: 0.1,
            kernelRegularizer: tf.regularizers.l2({ l2: 0.0005 }),
        }),
        name: 'hist_lstm1',
    }).apply(h);

    h = tf.layers.timeDistributed({
        layer: tf.layers.lstm({
            units: 32,
            returnSequences: false,
            recurrentDropout: 0.1,
            kernelRegularizer: tf.regularizers.l2({ l2: 0.0005 }),
        }),
        name: 'hist_lstm2',
    }).apply(h);
    // Shape: [batch, maxRunners, 32]

    h = tf.layers.timeDistributed({
        layer: tf.layers.batchNormalization(),
        name: 'hist_bn',
    }).apply(h);

    h = tf.layers.timeDistributed({
        layer: tf.layers.dropout({ rate: 0.2 }),
        name: 'hist_dropout',
    }).apply(h);

    // ── Static branch ─────────────────────────────────────────────────────────

    let s = tf.layers.timeDistributed({
        layer: tf.layers.dense({
            units: 48,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: 0.0005 }),
        }),
        name: 'static_dense1',
    }).apply(staticInput);
    // Shape: [batch, maxRunners, 48]

    s = tf.layers.timeDistributed({
        layer: tf.layers.batchNormalization(),
        name: 'static_bn',
    }).apply(s);

    s = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 32, activation: 'relu' }),
        name: 'static_dense2',
    }).apply(s);

    s = tf.layers.timeDistributed({
        layer: tf.layers.dropout({ rate: 0.2 }),
        name: 'static_dropout',
    }).apply(s);

    // ── Merge branches ────────────────────────────────────────────────────────

    let combined = tf.layers.concatenate({ axis: -1, name: 'combine' }).apply([h, s]);
    // Shape: [batch, maxRunners, 64]

    combined = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 64, activation: 'relu' }),
        name: 'combined_dense',
    }).apply(combined);

    combined = tf.layers.timeDistributed({
        layer: tf.layers.batchNormalization(),
        name: 'combined_bn',
    }).apply(combined);
    // Shape: [batch, maxRunners, 64]

    // ── Multi-Head Attention ──────────────────────────────────────────────────
    // Self-attention over the runner dimension: the model learns which other
    // horses in the race are relevant when scoring each individual runner.
    // Query = Key = Value = combined  →  numHeads=4, keyDim=16 → 64-dim output

    // Pre-LayerNorm before attention (stabilizes training)
    const preAttn = tf.layers.layerNormalization({ name: 'pre_attention_ln' }).apply(combined);

    const attended = new MultiHeadAttention({
        numHeads: 8,
        embedDim:   64, // match combined's last-dim (64) so downstream projection works
        dropout:  0.1,
    }).apply(preAttn);
    // Shape: [batch, maxRunners, 64]

    // Residual-style fusion: concatenate [combined, attended] -> project -> LayerNorm.
    const fused = tf.layers.concatenate({ axis: -1, name: 'attention_fuse_concat' }).apply([combined, attended]);
    const fusedProj = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 64, activation: 'linear' }),
        name: 'attention_fuse_proj',
    }).apply(fused);
    let normed   = tf.layers.layerNormalization({ name: 'attention_ln' }).apply(fusedProj);

    // Feed-Forward block (FFN) with residual + LayerNorm
    let ffn = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 128, activation: 'relu' }),
        name: 'ffn_dense1',
    }).apply(normed);

    ffn = tf.layers.timeDistributed({
        layer: tf.layers.dropout({ rate: 0.2 }),
        name: 'ffn_dropout',
    }).apply(ffn);

    ffn = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 64, activation: 'linear' }),
        name: 'ffn_dense2',
    }).apply(ffn);

    const resid = tf.layers.add({ name: 'ffn_residual_add' }).apply([normed, ffn]);
    normed = tf.layers.layerNormalization({ name: 'post_ffn_ln' }).apply(resid);

    // ── Output head ───────────────────────────────────────────────────────────

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
    // Apply mask to zero out padding slots at inference/training outputs
    out = tf.layers.multiply({ name: 'mask_apply' }).apply([out, maskInput]);
    // Shape: [batch, maxRunners, 1]

    const model = tf.model({ inputs: [histInput, staticInput, maskInput], outputs: out });
    // Try to add gradient clipping if supported; tfjs-layers Adam doesn't expose clipNorm directly in all versions.
    const optimizer = tf.train.adam(0.0003);
    model.compile({
        optimizer,
        loss:      combinedRaceLoss(AUX_LOSS_WEIGHT),
        metrics:   ['accuracy', recallAtThree, precisionAtThree],
    });
    return model;
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

function splitTrainValRaces(hist, stat, mask, y, valFraction = 0.1, seed = 42) {
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
    const maskTrain = tf.gather(mask, idxTrain);
    const yTrain    = tf.gather(y, idxTrain);

    const histVal = tf.gather(hist, idxVal);
    const statVal = tf.gather(stat, idxVal);
    const maskVal = tf.gather(mask, idxVal);
    const yVal    = tf.gather(y, idxVal);

    idxTrain.dispose(); idxVal.dispose();
    return { histTrain, statTrain, maskTrain, yTrain, histVal, statVal, maskVal, yVal };
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

// Per-race evaluation metrics: precision@k, recall@k, NDCG@k, Hit@1,
// macro-AP (per race) and per-race average log-loss. Returns dataset means.
function computePerRaceEval(yTrue, yPred, mask, k = 3) {
    const nRaces = yTrue.shape[0];
    const maxR = yTrue.shape[1];
    const yT = Array.from(yTrue.dataSync());
    const yP = Array.from(yPred.dataSync());
    const mK = Array.from(mask.dataSync());
    let sumPAtK = 0, sumRAtK = 0, sumNDCG = 0, sumHit1 = 0, sumAP = 0, sumLogLoss = 0, counted = 0;
    const eps = 1e-7;

    for (let r = 0; r < nRaces; r++) {
        // Collect valid slots for this race
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
        if (posCount === 0) continue; // skip races without any positives (or handle separately)

        // Sort by score desc
        const order = Array.from({length:n}, (_,i)=>i).sort((a,b)=>scs[b]-scs[a]);
        const sortedLabs = order.map(i=>labs[i]);

        // Precision@k and Recall@k
        const kk = Math.min(k, n);
        let tpAtK = 0;
        for (let i = 0; i < kk; i++) tpAtK += (sortedLabs[i] >= 0.5 ? 1 : 0);
        const pAtK = tpAtK / kk;
        const rAtK = tpAtK / Math.max(1, posCount);

        // Hit@1
        const hit1 = (sortedLabs[0] >= 0.5) ? 1 : 0;

        // NDCG@k (binary gains)
        const dcg = (()=>{
            let s=0; const log2 = (x)=>Math.log(x)/Math.log(2);
            for (let i=0;i<kk;i++) {
                const rel = sortedLabs[i] >= 0.5 ? 1 : 0;
                s += (Math.pow(2, rel) - 1) / log2(i+2); // i+2 because ranks are 1-based
            }
            return s;
        })();
        const ideal = (()=>{
            let s=0; const log2 = (x)=>Math.log(x)/Math.log(2);
            const rels = labs.slice().sort((a,b)=>(b-a)); // positives first
            const kk2 = Math.min(kk, posCount);
            for (let i=0;i<kk2;i++) s += (Math.pow(2,1)-1)/log2(i+2);
            return Math.max(eps, s);
        })();
        const ndcg = dcg / ideal;

        // AP (macro per race)
        let tp=0, apSum=0; // precision at each true positive
        for (let i=0;i<n;i++) {
            if (sortedLabs[i] >= 0.5) { tp++; apSum += tp/(i+1); }
        }
        const ap = apSum / Math.max(1, posCount);

        // Per-race logloss
        let ll = 0;
        for (let i=0;i<n;i++) {
            const y = labs[i];
            const p = Math.min(1-eps, Math.max(eps, scs[i]));
            ll += -(y*Math.log(p) + (1-y)*Math.log(1-p));
        }
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

async function runTraining() {
    console.log('--- TRAINING (race-based) ---');
    const data  = loadData(TRAINING_DATA);
    const model = buildModel(MAX_RUNNERS, MAX_HISTORY, data.histFeatureCount, data.staticFeatureCount);

    // Deterministic split at race level so we can predict full validation for metrics
    const { histTrain, statTrain, maskTrain, yTrain, histVal, statVal, maskVal, yVal } = splitTrainValRaces(
        data.hist, data.static, data.mask, data.y, 0.1, 1337
    );

    console.log(`  Tensor shapes — hist: ${data.hist.shape}  static: ${data.static.shape}  y: ${data.y.shape}`);
    console.log('  Metrikat: recall@3 = osuus oikeista top-3:sta, jotka löytyvät mallin\n' +
                '            kunkin lähdön kolmen korkeimman pisteen joukosta;\n' +
                '            precision@3 = osuus top-3 ennusteista, jotka ovat oikeita (keskiarvo);\n' +
                '            AUC(ROC) = erotteleva kyky kaikilla kynnyksillä (suuntaa-antava).');

    let bestValLoss     = Infinity;
    let bestValR3       = -Infinity;
    let patienceCounter = 0;     // val_loss patience
    let r3PatienceCount = 0;     // val_r3 patience
    const EARLY_STOP_PATIENCE = 16;
    const LR_REDUCE_PATIENCE  = 5;
    let epochStart;
    const history = [];

    await model.fit([histTrain, statTrain, maskTrain], yTrain, {
        epochs:          50,
        batchSize:       32,   // smaller batch due to 4D tensors
        shuffle:         true,
        validationData:  [[histVal, statVal, maskVal], yVal],
        // sampleWeight is not supported in tfjs-layers; padding is handled via mask_input gating.
        callbacks: {
            onEpochBegin: async () => { epochStart = Date.now(); },
            onEpochEnd: async (epoch, logs) => {
                const ms  = Date.now() - epochStart;
                const lr  = model.optimizer.learningRate;
                const acc = (logs.val_acc || logs.val_accuracy || 0).toFixed(4);
                const r3ValNum = Number(
                    logs.val_recallAtThree ??
                    logs.val_recall_at_three ??
                    logs.val_recall3 ??
                    logs.val_recall_at_3 ?? 0
                );
                const p3ValNum = Number(
                    logs.val_precisionAtThree ??
                    logs.val_precision_at_three ??
                    logs.val_p3 ?? 0
                );
                // Compute predictions for full validation set
                const yPredValTensor = model.predict([histVal, statVal, maskVal], { batchSize: 16 });
                // Per-race metrics at k=3 (precision/recall/NDCG), plus Hit@1, macro-AP, and per-race logloss
                const perRace = computePerRaceEval(yVal, yPredValTensor, maskVal, 3);
                // Manual ROC-AUC over valid runner slots (mask==1)
                const yPredArr = Array.from(yPredValTensor.dataSync());
                yPredValTensor.dispose();
                const yTrueArrAll = Array.from(yVal.dataSync());
                const maskArr = Array.from(maskVal.dataSync());
                const yTrueArr = [];
                const yScoreArr = [];
                for (let i = 0; i < yTrueArrAll.length; i++) {
                    if (maskArr[i] >= 0.5) { yTrueArr.push(yTrueArrAll[i]); yScoreArr.push(yPredArr[i]); }
                }
                const { auc: rocAuc } = computeRocAuc(yTrueArr, yScoreArr, 0.01);
                const r3v = r3ValNum.toFixed(4);
                const p3v = p3ValNum.toFixed(4);
                const aucv = rocAuc.toFixed(4);
                const ndcg3v = perRace.ndcgK.toFixed(4);
                const hit1v  = perRace.hit1.toFixed(4);
                const apmv   = perRace.apMacro.toFixed(4);
                const llrv   = perRace.loglossRace.toFixed(4);

                history.push({
                    epoch: epoch + 1,
                    loss: logs.loss,
                    val_loss: logs.val_loss,
                    val_acc: Number(logs.val_acc || logs.val_accuracy || 0),
                    val_r3: r3ValNum,
                    val_p3: p3ValNum,
                    val_auc: rocAuc,
                    val_ndcg3: perRace.ndcgK,
                    val_hit1: perRace.hit1,
                    val_ap_macro: perRace.apMacro,
                    val_logloss_race: perRace.loglossRace,
                    lr,
                });

                console.log(
                    `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} | ` +
                    `val_loss=${logs.val_loss.toFixed(4)} | val_acc=${acc} | val_r@3=${r3v} | val_p@3=${p3v} | val_auc=${aucv} | ` +
                    `ndcg@3=${ndcg3v} | hit@1=${hit1v} | AP_macro=${apmv} | race_logloss=${llrv} | ` +
                    `lr=${lr.toFixed(6)} | ` +
                    `${Math.floor(ms / 60000)}m ${((ms % 60000) / 1000).toFixed(0)}s`
                );

                if (logs.val_loss < bestValLoss) {
                    bestValLoss = logs.val_loss;
                    patienceCounter = 0;
                    console.log('   ⭐ New best val_loss — saving...');
                    await saveModel(model, {
                        epoch:         epoch + 1,
                        loss:          Math.round(logs.loss * 10000) / 10000,
                        val_loss:      Math.round(logs.val_loss * 10000) / 10000,
                        val_acc:       Math.round((logs.val_acc || logs.val_accuracy || 0) * 10000) / 10000,
                        val_r3:        Math.round(r3ValNum * 10000) / 10000,
                        val_p3:        Math.round(p3ValNum * 10000) / 10000,
                        val_auc:       Math.round(rocAuc * 10000) / 10000,
                        val_ndcg3:     Math.round(perRace.ndcgK * 10000) / 10000,
                        val_hit1:      Math.round(perRace.hit1 * 10000) / 10000,
                        val_ap_macro:  Math.round(perRace.apMacro * 10000) / 10000,
                        val_logloss_race: Math.round(perRace.loglossRace * 10000) / 10000,
                        learningRate:  lr,
                        best_by:       'val_loss',
                        history,
                        ...data.dataMeta,
                    }, MODEL);
                } else {
                    patienceCounter++;
                }

                // Best global threshold on validation (optimize F1 over valid slots)
                let bestThr = 0.5, bestF1 = 0;
                for (let t = 0.05; t <= 0.951; t += 0.01) {
                    let tp=0, fp=0, fn=0;
                    for (let i=0;i<yTrueArr.length;i++) {
                        const y = yTrueArr[i] >= 0.5 ? 1 : 0;
                        const yhat = yScoreArr[i] >= t ? 1 : 0;
                        if (yhat===1 && y===1) tp++; else if (yhat===1 && y===0) fp++; else if (yhat===0 && y===1) fn++;
                    }
                    const prec = tp/Math.max(1,tp+fp);
                    const rec  = tp/Math.max(1,tp+fn);
                    const f1   = (prec+rec)>0 ? (2*prec*rec)/(prec+rec) : 0;
                    if (f1 > bestF1) { bestF1 = f1; bestThr = t; }
                }
                history[history.length-1].val_best_threshold = bestThr;
                history[history.length-1].val_f1_best = bestF1;

                if (r3ValNum > bestValR3) {
                    bestValR3 = r3ValNum;
                    r3PatienceCount = 0;
                    console.log('   🌟 New best val_r@3 — saving...');
                    await saveModel(model, {
                        epoch:         epoch + 1,
                        loss:          Math.round(logs.loss * 10000) / 10000,
                        val_loss:      Math.round(logs.val_loss * 10000) / 10000,
                        val_acc:       Math.round((logs.val_acc || logs.val_accuracy || 0) * 10000) / 10000,
                        val_r3:        Math.round(r3ValNum * 10000) / 10000,
                        val_p3:        Math.round(p3ValNum * 10000) / 10000,
                        val_auc:       Math.round(rocAuc * 10000) / 10000,
                        val_ndcg3:     Math.round(perRace.ndcgK * 10000) / 10000,
                        val_hit1:      Math.round(perRace.hit1 * 10000) / 10000,
                        val_ap_macro:  Math.round(perRace.apMacro * 10000) / 10000,
                        val_logloss_race: Math.round(perRace.loglossRace * 10000) / 10000,
                        learningRate:  lr,
                        best_by:       'val_r3',
                        history,
                        ...data.dataMeta,
                    }, 'model_best_r3.json');
                } else {
                    r3PatienceCount++;
                }

                // LR scheduling: reduce LR when r3 hasn't improved recently (primary) or loss hasn't (secondary)
                if (r3PatienceCount >= LR_REDUCE_PATIENCE || patienceCounter >= LR_REDUCE_PATIENCE) {
                    model.optimizer.learningRate = lr * 0.5;
                    console.log(`   📉 Reducing LR (plateau): ${(lr * 0.5).toFixed(6)}`);
                    r3PatienceCount = 0;
                    patienceCounter = 0;
                }

                // Early stopping when both metrics plateau sufficiently
                if (patienceCounter >= EARLY_STOP_PATIENCE && r3PatienceCount >= EARLY_STOP_PATIENCE) {
                    console.log('--- Early stopping (loss & r@3 plateau) ---');
                    model.stopTraining = true;
                }
            },
        },
    });

    data.hist.dispose();
    data.static.dispose();
    data.mask.dispose();
    data.y.dispose();
    // Dispose split tensors
    histTrain.dispose(); statTrain.dispose(); maskTrain.dispose(); yTrain.dispose();
    histVal.dispose();   statVal.dispose();   maskVal.dispose();   yVal.dispose();
}

runTraining();
