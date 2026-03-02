// ─── util.js ─────────────────────────────────────────────────────────────────
// Shared feature-building utilities for runner-based and race-based pipelines.
//
// DATA FLOW — two paths, one set of field names after normalisation:
//
//  PATH 1 — Training JSON (ravit_opetusdata.json, produced by scraper.js):
//    Fields already in schema form — scraper.js maps everything on write.
//    ps.date, ps.driver, ps.track, ps.number, ps.kmTime (number|null),
//    ps.distance (number|null), ps.isCarStart, ps.break, ps.disqualified,
//    ps.DNF, ps.trackCondition, ps.position (number|null),
//    ps.odd (winOdd/10), ps.firstPrice (firstPrize/10000 euros),
//    ps.frontShoes, ps.rearShoes, ps.specialCart
//
//  PATH 2 — Live Veikkaus API (App.jsx → normaliseRunners):
//    Raw API field names differ — normalisePrevStart() maps them:
//      h.shortMeetDate         → ps.date
//      h.driver                → ps.driver  (NOT h.driverFullName)
//      h.trackCode             → ps.track
//      h.startTrack            → ps.number
//      h.distance              → ps.distance
//      h.kmTime (raw string)   → ps.kmTime (number|null) via parseKmTime
//                                ps.isCarStart, ps.break derived from kmTime
//      h.result (string)       → ps.position (number|null),
//                                ps.disqualified, ps.DNF
//      h.winOdd   (×10 int)    → ps.odd  (÷10)
//      h.firstprice (cents×10) → ps.firstPrice (÷10000 euros)  ← lowercase p!
//      h.frontShoes/rearShoes/specialCart/trackCondition — same keys
//
//  RUNNER fields — live API (race/{raceId}/runners):
//    r.number         → number   (NOT r.startNumber)
//    r.horseName      → name
//    r.coachName      → coach
//    r.driverName     → driver
//    r.horseAge       → age
//    r.gender         → 'ORI'=3, 'TAMMA'=1, 'RUUNA'=2
//    r.frontShoes     → frontShoes  (HAS_SHOES/NO_SHOES/UNKNOWN)
//    r.rearShoes      → rearShoes
//    r.frontShoesChanged → frontShoesChanged
//    r.rearShoesChanged  → rearShoesChanged
//    r.specialChart   → specialCart  (NOTE: API typo "Chart" not "Cart"!)
//    r.scratched      → scratched
//    r.stats.total.position1/starts*100 → winPct
//    r.betPercentages.KAK.percentage/100 → bettingPct
//    r.mobileStartRecord/handicapRaceRecord/vaultStartRecord → record, isAutoRecord
// ─────────────────────────────────────────────────────────────────────────────

export const MAX_HISTORY = 8;
export const MAX_RUNNERS = 18;

// ─── STRING / DATE HELPERS ────────────────────────────────────────────────────

export function sanitize(str) {
    if (!str) return '';
    return str
        .replace(/ä/g, 'a').replace(/ö/g, 'o').replace(/å/g, 'a')
        .replace(/Ä/g, 'A').replace(/Ö/g, 'O').replace(/Å/g, 'A')
        .replace(/[^a-zA-Z0-9\s\-\.\:]/g, '')
        .trim();
}

export function extractSurname(fullName) {
    if (!fullName) return 'unknown';
    return fullName.trim().toLowerCase().split(' ').at(-1);
}

export function parseDate(str) {
    if (!str || str === '0' || str === 'NaT') return null;
    if (str.includes('-')) return new Date(str);
    if (str.includes('.')) {
        const [d, m, y] = str.split('.');
        const year = parseInt(y) < 100 ? parseInt(y) + 2000 : parseInt(y);
        return new Date(year, parseInt(m) - 1, parseInt(d));
    }
    return null;
}

// ─── KM-TIME PARSER ───────────────────────────────────────────────────────────
// For LIVE API: kmTime is a raw string like "15,5", "15,5a", "15,5x".
// Training JSON: kmTime is already a number (scraper.js pre-parses it).

export function parseKmTime(raw) {
    if (!raw) return { kmNum: null, isCarStart: false, isBreak: false };
    const s          = String(raw);
    const isCarStart = s.includes('a');
    const isBreak    = s.includes('x');
    const match      = s.match(/[\d,]+/);
    const kmNum      = match ? parseFloat(match[0].replace(',', '.')) : null;
    return { kmNum: (kmNum && !isNaN(kmNum)) ? kmNum : null, isCarStart, isBreak };
}

// Normalise km time to 2100 m base. Falls back to 2100 when distance is null.
export function normaliseKmTime(km, distance) {
    if (km === null || km === undefined || isNaN(km) || km <= 0) return null;
    const dist = (distance != null && distance > 0) ? distance : 2100;
    return km + (2100 - dist) / 2000;
}

// ─── RESULT STRING PARSER ─────────────────────────────────────────────────────
// For LIVE API prevStarts: result is a raw string like "1", "Hylätty", "K".
// Training JSON: position/disqualified/DNF already resolved by scraper.js.

export function parsePosition(result) {
    if (!result) return { position: null, disqualified: false, DNF: false };
    const s            = String(result).toLowerCase();
    const disqualified = /[hdp]/.test(s);
    const DNF          = s.includes('k');
    const numMatch     = s.match(/^\d+/);
    let position       = null;
    if (numMatch)          position = parseInt(numMatch[0]);
    else if (disqualified) position = 20;
    else if (DNF)          position = 21;
    return { position, disqualified, DNF };
}

// ─── OTHER ENCODERS ───────────────────────────────────────────────────────────

// breed field from card/{cardId}/races: 'K' = Finnhorse, 'L' = Warmblood
export function detectColdBlood(raceInfo) {
    return raceInfo?.breed === 'K';
}

// Gender from live API: 'ORI'=stallion(3), 'TAMMA'=mare(1), 'RUUNA'=gelding(2)
export function encodeGender(g) {
    if (g === 'TAMMA') return 1;
    if (g === 'ORI')   return 3;
    return 2;   // RUUNA or anything else
}

// Record: same priority order as scraper.js
export function parseRecord(runner, isCarStart) {
    const order = isCarStart
        ? ['mobileStartRecord', 'handicapRaceRecord', 'vaultStartRecord']
        : ['handicapRaceRecord', 'mobileStartRecord', 'vaultStartRecord'];
    for (const key of order) {
        const val = runner[key];
        if (!val) continue;
        const match = String(val).match(/^[\d,]+/);
        if (!match) continue;
        const num = parseFloat(match[0].replace(',', '.'));
        if (!isNaN(num) && num > 0)
            return { record: num, isAutoRecord: key === 'mobileStartRecord' };
    }
    return { record: null, isAutoRecord: false };
}

export function encodeTrackCondition(condition) {
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

// winPct: collection[j].stats.total.position1 / collection[j].stats.total.starts * 100
function parseWinPct(runner) {
    const total = runner.stats?.total;
    if (!total) return 0;
    if (total.winningPercent != null) return parseFloat(total.winningPercent);
    if (total.starts > 0) return Math.round((total.position1 / total.starts) * 10000) / 100;
    return 0;
}

// bettingPct: collection[j].betPercentages.KAK.percentage / 100
function parseBettingPct(runner) {
    const pct = runner.betPercentages?.KAK?.percentage;
    return pct != null ? parseFloat(pct) / 100 : 0;
}

// ─── LIVE API: normalisePrevStart ─────────────────────────────────────────────
// Maps raw Veikkaus API prevStart object to training-schema field names.
// Called inside normaliseRunners for every element of r.prevStarts.

function normalisePrevStart(h) {
    // kmTime is a raw string in live API — parse to get numeric + flags
    const { kmNum, isCarStart, isBreak } = parseKmTime(h.kmTime);

    // result is a raw string — parse to get numeric position + flags
    const { position, disqualified, DNF } = parsePosition(h.result);

    // winOdd: stored as integer ×10 (e.g. "152" → 15.2)
    const rawOdd = parseFloat(h.winOdd);
    const odd    = (!isNaN(rawOdd) && rawOdd > 0) ? rawOdd / 10 : null;

    // firstprice: stored as cents×10 (e.g. 1000000 → 100 €). NOTE: lowercase 'p'!
    const rawPrize   = parseFloat(h.firstprice ?? h.firstPrize);  // lowercase p preferred, uppercase fallback
    const firstPrice = (!isNaN(rawPrize) && rawPrize > 0) ? rawPrize / 10000 : null;

    // shortMeetDate: keep as-is for filterValidPrev (non-empty = valid).
    // Try to normalise to ISO for daysSince calculation, but fall back to
    // the raw string so the entry is never silently dropped due to an
    // unexpected date format (e.g. "100425" without separators).
    const rawDate = h.shortMeetDate || h.meetDate || '';
    let date = null;
    if (rawDate) {
        const d = parseDate(rawDate);
        date = d
            ? `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
            : rawDate;   // keep raw string — filterValidPrev only checks non-empty
    }

    return {
        date,
        driver:         sanitize(h.driverFullName || h.driver || ''),  // driverFullName preferred
        track:          h.trackCode || h.track || null,
        number:         parseInt(h.startTrack) || null,
        distance:       parseInt(h.distance)   || null,
        kmTime:         kmNum,                           // already numeric (null if missing)
        frontShoes:     h.frontShoes   || null,
        rearShoes:      h.rearShoes    || null,
        specialCart:    h.specialCart  || null,
        isCarStart,                                      // derived from kmTime 'a'
        break:          isBreak,                         // derived from kmTime 'x'
        disqualified,                                    // derived from result
        DNF,                                             // derived from result
        trackCondition: h.trackCondition || null,
        position,                                        // numeric (null if unknown)
        odd,                                             // winOdd/10
        firstPrice,                                      // firstprice/10000 euros
    };
}

// Shoe value normalisation — accept multiple API formats (same as old App.jsx)
function encodeShoes(val) {
    if (!val) return 'UNKNOWN';
    const s = String(val).toUpperCase();
    if (['HAS_SHOES', 'TRUE', 'YES', '1'].includes(s)) return 'HAS_SHOES';
    if (['NO_SHOES',  'FALSE', 'NO',  '0'].includes(s)) return 'NO_SHOES';
    return 'UNKNOWN';
}

// ─── LIVE API: normaliseRunners ───────────────────────────────────────────────
// Maps raw Veikkaus race/{raceId}/runners entries to training-schema shape.
// Called in App.jsx immediately after fetching runners.

export function normaliseRunners(runnersArr, isCarStart) {
    return (runnersArr || [])
        .filter(r => r.scratched !== true)
        .map(r => {
            const { record, isAutoRecord } = parseRecord(r, isCarStart);
            return {
                number:            parseInt(r.startNumber || r.number || 0),  // collection[j].startNumber
                name:              sanitize(r.horseName  || ''),  // collection[j].horseName
                coach:             sanitize(r.coachName  || ''),  // collection[j].coachName
                driver:            sanitize(r.driverName || ''),  // collection[j].driverName
                age:               parseInt(r.horseAge)  || 0,   // collection[j].horseAge
                gender:            encodeGender(r.gender),        // 'ORI'/'TAMMA'/'RUUNA'→1/2/3
                frontShoes:        encodeShoes(r.frontShoes),   // collection[j].frontShoes
                rearShoes:         encodeShoes(r.rearShoes),    // collection[j].rearShoes
                frontShoesChanged: r.frontShoesChanged === true,
                rearShoesChanged:  r.rearShoesChanged  === true,
                specialCart:       r.specialCart || 'UNKNOWN',   // NOTE: API typo "specialChart" vitut, "specialCart se on!
                scratched:         r.scratched === true,
                record,
                isAutoRecord,
                bettingPct:        parseBettingPct(r),            // KAK.percentage/100
                winPct:            parseWinPct(r),                // position1/starts*100
                // Map every prevStart from raw API names → schema names
                prevStarts:        (r.prevStarts || []).map(normalisePrevStart),
            };
        });
}

// ─── MAP ACCESSORS ────────────────────────────────────────────────────────────

export function makeMaps(maps) {
    return {
        coachMap:  maps.coaches || {},
        driverMap: maps.drivers || {},
        trackMap:  maps.tracks  || {},
    };
}

export function makeGetID() {
    return (map, name, type) => {
        if (!name || name === '') return 0;
        const key = (type === 'driver') ? extractSurname(name) : name.trim().toLowerCase();
        return map[key] || 0;
    };
}

// ─── VALID PREV FILTER ────────────────────────────────────────────────────────
// After normalisePrevStart, both paths use ps.date and ps.driver.
// Matches scraper.js filter logic exactly.

export function filterValidPrev(prevStarts) {
    return (prevStarts || []).filter(ps => {
        const date   = (ps.date   || '').trim();
        const driver = (ps.driver || '').trim();
        return date !== '' && driver !== '';
    });
}

// ─── PREVINDEXNORM ────────────────────────────────────────────────────────────
// Weighted podium score / all valid starts. ps.position is numeric in both paths.

export function calcPrevIndexNorm(validPrev) {
    if (validPrev.length === 0) return 0;
    let score = 0, count = 0;
    for (const ps of validPrev) {
        const pos = ps.position;
        if (pos == null) continue;
        count++;
        if      (pos === 1) score += 1.00;
        else if (pos === 2) score += 0.50;
        else if (pos === 3) score += 0.33;
    }
    return count > 0 ? score / count : 0;
}

// ─── HISTORY SEQUENCE ─────────────────────────────────────────────────────────
// Builds MAX_HISTORY × 25 tensor for one runner.
// All prevStart entries must already be in schema form (ps.date, ps.driver, etc.)
// before this function is called — normaliseRunners/normalisePrevStart handles
// the live API path; training JSON is already in schema form from scraper.js.

export function buildHistorySequence(validPrev, raceDate, breed, means, driverMap, trackMap, getID) {
    const histSeq = [];

    for (let i = 0; i < MAX_HISTORY; i++) {
        const ps = validPrev[i];
        if (!ps) { histSeq.push(new Array(25).fill(-1)); continue; }

        const raceDateObj  = parseDate(raceDate);
        const startDateObj = parseDate(ps.date);         // ps.date is always ISO after normalisation
        const daysSince    = (raceDateObj && startDateObj)
            ? Math.min(365, (raceDateObj - startDateObj) / 86400000)
            : 30;

        // ps.kmTime: number from training/normalisePrevStart (null if missing)
        // ps.distance: metres, may be null in older training rows → falls back to 2100
        const kmNorm  = normaliseKmTime(ps.kmTime, ps.distance);
        const kmKnown = kmNorm > 0 ? 1 : 0;
        const kmFinal = kmKnown ? kmNorm / 100 : means[breed].km / 100;

        const distKnown  = (ps.distance || 0) > 0 ? 1 : 0;
        const distFinal  = distKnown ? ps.distance / 3100 : 0.67;

        // ps.firstPrice: euros, already scaled in both paths
        const prize      = parseFloat(ps.firstPrice) || 0;
        const prizeKnown = prize > 0 ? 1 : 0;
        const prizeFinal = prizeKnown ? Math.log1p(prize) / 10 : 0.55;

        // ps.odd: actual value, already scaled in both paths
        const odd      = parseFloat(ps.odd) || 0;
        const oddKnown = odd > 0 ? 1 : 0;
        const oddFinal = oddKnown ? Math.log1p(odd) / 5 : 0.50;

        // ps.position: numeric in both paths
        const pos      = ps.position;
        const posKnown = pos != null && pos > 0 ? 1 : 0;
        const posFinal = posKnown ? pos / 20 : 0.5;

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
            ps.break        ? 1 : 0,                                       // [12]
            (ps.number ?? 1) / 30,                                         // [13]  RataNro / startTrack
            getID(driverMap, ps.driver, 'driver') / 3000,                  // [14]  ps.driver
            getID(trackMap,  ps.track,  'track')  / 500,                   // [15]  ps.track / trackCode
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

    return histSeq;
}

// ─── STATIC FEATURE VECTOR ────────────────────────────────────────────────────
// mode 'runner' → 27 features (with betRank/winRank [21-24])
// mode 'race'   → 25 features (attention learns ranking instead)

export function buildStaticFeatures(runner, opts) {
    const {
        breed, means, coachMap, driverMap, raceDistance, isCarStart,
        isColdBlood, betRank, betKnown, winRank, winKnown,
        prevIndexNorm, histKnown, mode, getID,
    } = opts;

    const frontStr    = (runner.frontShoes || '').toUpperCase();
    const rearStr     = (runner.rearShoes  || '').toUpperCase();
    const frontActive = frontStr === 'HAS_SHOES' ? 1 : 0;
    const frontKnown  = (frontStr === 'HAS_SHOES' || frontStr === 'NO_SHOES') ? 1 : 0;
    const rearActive  = rearStr  === 'HAS_SHOES' ? 1 : 0;
    const rearKnown   = (rearStr  === 'HAS_SHOES' || rearStr  === 'NO_SHOES') ? 1 : 0;

    const cartStr    = (runner.specialCart || '').toUpperCase();
    const cartActive = cartStr === 'YES' ? 1 : 0;
    const cartKnown  = (cartStr === 'YES' || cartStr === 'NO') ? 1 : 0;

    const base = [
        (runner.number || 1) / 20,                                        // [0]
        getID(coachMap,  runner.coach,  'coach')  / 2000,                 // [1]
        (runner.record || means[breed].record) / 50,                      // [2]
        getID(driverMap, runner.driver, 'driver') / 3000,                 // [3]
        (runner.age || 5) / 15,                                           // [4]
        (runner.gender || 2) / 3,                                         // [5]  1=mare 2=gelding 3=stallion
        isColdBlood ? 1 : 0,                                              // [6]
        frontActive, frontKnown,                                          // [7-8]
        rearActive,  rearKnown,                                           // [9-10]
        runner.frontShoesChanged ? 1 : 0,                                 // [11]
        runner.rearShoesChanged  ? 1 : 0,                                 // [12]
        (raceDistance || 2100) / 3100,                                    // [13]
        isCarStart ? 1 : 0,                                               // [14]
        (runner.bettingPct || 0) / 100,                                   // [15]
        (runner.winPct     || 0) / 100,                                   // [16]
        (runner.winPct     || 0) > 0 ? 1 : 0,                             // [17]
        runner.isAutoRecord ? 1 : 0,                                      // [18]
        cartActive, cartKnown,                                            // [19-20]
    ];

    if (mode === 'runner') {
        base.push(
            betRank / 20, betKnown,                                       // [21-22]
            winRank / 20, winKnown,                                       // [23-24]
        );
    }

    base.push(
        prevIndexNorm,                                                    // [21] / [25]
        histKnown,                                                        // [22] / [26]
    );

    // Race mode: two reserved zeros [23-24] to match ravimalli-race.js training shape exactly
    if (mode === 'race') base.push(0, 0);

    return base;
}

// ─── RUNNER-BASED FEATURE BUILDER ─────────────────────────────────────────────
// X_hist [n_runners, 8, 25], X_static [n_runners, 27]
// Matches model.js training tensors exactly.

export function buildRunnerBasedFeatures(runners, maps, raceDate, raceDistance, isColdBlood, isCarStart) {
    const breed = isColdBlood ? 'SH' : 'LV';
    const means = { SH: { record: 28.0, km: 29.0 }, LV: { record: 15.0, km: 16.0 } };
    const { coachMap, driverMap, trackMap } = makeMaps(maps);
    const getID = makeGetID();

    const starters = runners.filter(r => !r.scratched);
    const hasBet   = starters.some(r => (r.bettingPct || 0) > 0);
    const hasWin   = starters.some(r => (r.winPct     || 0) > 0);
    const neutral  = (starters.length + 1) / 2;

    const byBet = [...starters].sort((a, b) => (b.bettingPct || 0) - (a.bettingPct || 0));
    const byWin = [...starters].sort((a, b) => (b.winPct     || 0) - (a.winPct     || 0));

    const X_hist = [], X_static = [], metadata = [];

    for (const runner of starters) {
        const betKnown = hasBet && (runner.bettingPct || 0) > 0 ? 1 : 0;
        const winKnown = hasWin && (runner.winPct     || 0) > 0 ? 1 : 0;
        const betRank  = betKnown ? byBet.findIndex(r => r.number === runner.number) + 1 : neutral;
        const winRank  = winKnown ? byWin.findIndex(r => r.number === runner.number) + 1 : neutral;

        const validPrev     = filterValidPrev(runner.prevStarts);
        const prevIndexNorm = calcPrevIndexNorm(validPrev);
        const histKnown     = validPrev.length > 0 ? 1 : 0;

        X_static.push(buildStaticFeatures(runner, {
            breed, means, coachMap, driverMap, raceDistance, isCarStart, isColdBlood,
            betRank, betKnown, winRank, winKnown, prevIndexNorm, histKnown,
            mode: 'runner', getID,
        }));
        X_hist.push(buildHistorySequence(validPrev, raceDate, breed, means, driverMap, trackMap, getID));
        metadata.push({ number: runner.number, name: runner.name, driver: runner.driver });
    }

    return { X_hist, X_static, metadata };
}

// ─── RACE-BASED FEATURE BUILDER ───────────────────────────────────────────────
// X_hist [1, MAX_RUNNERS, 8, 25], X_static [1, MAX_RUNNERS, 25]
// Matches ravimalli-race.js training tensors exactly.

export function buildRaceBasedFeatures(runners, maps, raceDate, raceDistance, isColdBlood, isCarStart) {
    const breed = isColdBlood ? 'SH' : 'LV';
    const means = { SH: { record: 28.0, km: 29.0 }, LV: { record: 15.0, km: 16.0 } };
    const { coachMap, driverMap, trackMap } = makeMaps(maps);
    const getID = makeGetID();

    const starters  = runners.filter(r => !r.scratched);
    const raceHist  = [], raceStatic = [], raceMask = [], metadata = [];

    for (let slot = 0; slot < MAX_RUNNERS; slot++) {
        const runner = starters[slot];
        if (!runner) {
            raceStatic.push(new Array(25).fill(0));
            raceHist.push(Array.from({ length: MAX_HISTORY }, () => new Array(25).fill(-1)));
            raceMask.push([0]);
            continue;
        }

        const validPrev     = filterValidPrev(runner.prevStarts);
        const prevIndexNorm = calcPrevIndexNorm(validPrev);
        const histKnown     = validPrev.length > 0 ? 1 : 0;

        raceStatic.push(buildStaticFeatures(runner, {
            breed, means, coachMap, driverMap, raceDistance, isCarStart, isColdBlood,
            betRank: null, betKnown: null, winRank: null, winKnown: null,
            prevIndexNorm, histKnown, mode: 'race', getID,
        }));
        raceHist.push(buildHistorySequence(validPrev, raceDate, breed, means, driverMap, trackMap, getID));
        raceMask.push([1]);
        metadata.push({ number: runner.number, name: runner.name, driver: runner.driver, slot });
    }

    return { X_hist: [raceHist], X_static: [raceStatic], X_mask: [raceMask], metadata, numStarters: starters.length };
}

// ─── JSON FETCH ───────────────────────────────────────────────────────────────

export async function fetchJSON(path) {
    const res = await fetch(path);
    if (!res.ok)
        throw new Error(`HTTP ${res.status} — file not found: ${path}`);
    const text = await res.text();
    if (text.trimStart().startsWith('<'))
        throw new Error(`Server returned HTML instead of JSON: ${path}`);
    return JSON.parse(text);
}
