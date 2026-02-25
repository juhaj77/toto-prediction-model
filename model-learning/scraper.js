// ═══════════════════════════════════════════════════════════════════════════════
// TOTO HISTORY SCRAPER
// Fetches harness racing data from the Veikkaus API and appends to JSON.
//
// Usage:
//   node scraper.js                        → continues from last date in JSON
//   node scraper.js 2026-01-01             → from this date to yesterday
//   node scraper.js 2026-01-01 2026-01-31  → explicit date range
//   node scraper.js 2026-01-01 --force     → re-fetch and overwrite existing
// ═══════════════════════════════════════════════════════════════════════════════

'use strict';

const fs    = require('fs');
const https = require('https');

const OUTPUT_FILE = './ravit_opetusdata.json';
const BASE_URL    = 'https://www.veikkaus.fi/api/toto-info/v1';

// ─── HTTP ─────────────────────────────────────────────────────────────────────

function fetchJSON(url) {
    return new Promise((resolve, reject) => {
        const req = https.get(url, {
            headers: { 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json' },
            timeout: 20000,
        }, res => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                if (res.statusCode !== 200) {
                    reject(new Error(`HTTP ${res.statusCode}: ${url}`));
                    return;
                }
                try   { resolve(JSON.parse(data)); }
                catch (e) { reject(new Error(`JSON parse error: ${url}\n${e.message}`)); }
            });
        });
        req.on('error', reject);
        req.on('timeout', () => { req.destroy(); reject(new Error(`Timeout: ${url}`)); });
    });
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ─── DATE HELPERS ─────────────────────────────────────────────────────────────

const dateToISO = d => d.toISOString().slice(0, 10);

function addDays(dateStr, n) {
    const d = new Date(dateStr + 'T12:00:00Z');
    d.setUTCDate(d.getUTCDate() + n);
    return dateToISO(d);
}

function dateRange(start, end) {
    const dates = [];
    let cur = start;
    while (cur <= end) { dates.push(cur); cur = addDays(cur, 1); }
    return dates;
}

// ─── PARSERS ──────────────────────────────────────────────────────────────────

// Transliterate Finnish characters and strip non-ASCII punctuation.
// Mirrors the MATLAB scraper's `puhdista` function.
function sanitize(str) {
    if (!str) return '';
    return str
        .replace(/ä/g, 'a').replace(/ö/g, 'o').replace(/å/g, 'a')
        .replace(/Ä/g, 'A').replace(/Ö/g, 'O').replace(/Å/g, 'A')
        .replace(/[^a-zA-Z0-9 \-\.]/g, '')
        .trim();
}

// Parse raw km-time string from the API.
// Formats: "15,5"  "15,5a" (car start)  "15,5x" (gait fault / break)
function parseKmTime(raw) {
    if (!raw) return { kmNum: null, isCarStart: false, isBreak: false };
    const s          = String(raw);
    const isCarStart = s.includes('a');
    const isBreak    = s.includes('x');
    const match      = s.match(/[\d,]+/);
    const kmNum      = match ? parseFloat(match[0].replace(',', '.')) : null;
    return { kmNum: (kmNum && !isNaN(kmNum)) ? kmNum : null, isCarStart, isBreak };
}

// Parse finishing position from API result string.
// Returns numeric position (1–16), 20 for disqualification, 21 for DNF.
function parsePosition(result) {
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

// Select the best available race record.
// Priority depends on start type: car starts prefer mobileStartRecord.
// Format: "15,2ke" or "12,6aly" — leading digits are the value.
function parseRecord(runner, isCarStart) {
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

// Track condition — return null for unknown/missing values
function parseTrackCondition(val) {
    if (!val) return null;
    const s = String(val).toLowerCase().trim();
    return (s === 'unknown' || s === '') ? null : s;
}

// Betting percentage — betPercentages.KAK.percentage / 100
function parseBettingPct(runner) {
    const pct = runner.betPercentages?.KAK?.percentage;
    return pct != null ? parseFloat(pct) / 100 : null;
}

// Historical win percentage from runner stats
function parseWinPct(runner) {
    const total = runner.stats?.total;
    if (!total) return null;
    if (total.winningPercent != null) return parseFloat(total.winningPercent);
    if (total.starts > 0) return Math.round((total.position1 / total.starts) * 10000) / 100;
    return null;
}

// Gender → integer (1 = mare, 2 = gelding, 3 = stallion)
function encodeGender(g) {
    if (g === 'TAMMA') return 1;
    if (g === 'ORI')   return 3;
    return 2;
}

// ─── FETCH ONE DAY ────────────────────────────────────────────────────────────

async function fetchDay(dateStr) {
    const races = [];

    let cardsData;
    try {
        cardsData = await fetchJSON(`${BASE_URL}/cards/date/${dateStr}`);
    } catch (e) {
        console.warn(`  No cards for ${dateStr}: ${e.message}`);
        return races;
    }

    const cards = cardsData.collection || cardsData.cards || [];
    if (cards.length === 0) { console.log(`  No races on ${dateStr}`); return races; }

    for (const card of cards) {
        const cardId = String(card.cardId || card.id);
        await sleep(200);

        let racesData;
        try {
            racesData = await fetchJSON(`${BASE_URL}/card/${cardId}/races`);
        } catch (e) {
            console.warn(`  Error fetching card ${cardId} races: ${e.message}`);
            continue;
        }

        const raceList = racesData.collection || racesData.races || [];

        // Fetch the canonical meet date from the card endpoint
        let meetDate = dateStr;
        try {
            const cardInfo = await fetchJSON(`${BASE_URL}/card/${cardId}`);
            meetDate = (cardInfo.meetDate || cardInfo.date || dateStr).slice(0, 10);
        } catch (_) {}

        // Build result map: raceId → [startNumber1, startNumber2, startNumber3]
        const resultMap = {};
        for (const r of raceList) {
            const rid = String(r.raceId || r.id);
            const res = r.toteResultString || '';
            if (res) resultMap[rid] = res.split('-').map(s => s.trim());
        }

        for (const raceInfo of raceList) {
            const raceId      = String(raceInfo.raceId || raceInfo.id);
            const raceDistance = parseInt(raceInfo.distance || 2100);
            const isColdBlood  = (raceInfo.breed === 'K' || raceInfo.breed === 'FINNHORSE');
            const isCarStart   = (raceInfo.startType === 'CAR_START' || raceInfo.startType === 'AUTO');

            await sleep(150);

            let runnersRaw;
            try {
                runnersRaw = await fetchJSON(`${BASE_URL}/race/${raceId}/runners`);
            } catch (e) {
                console.warn(`  Error fetching race ${raceId} runners: ${e.message}`);
                continue;
            }

            const runnersArr   = Array.isArray(runnersRaw)
                ? runnersRaw
                : (runnersRaw.collection || runnersRaw.runners || []);

            const topPositions = resultMap[raceId] || [];

            const runners = runnersArr.map(r => {
                const startNum = parseInt(r.startNumber || r.number || 0);

                // Actual finishing position from the race result string
                let position = null;
                const posIdx = topPositions.indexOf(String(startNum));
                if (posIdx !== -1) position = posIdx + 1;   // 1–3

                const { record, isAutoRecord } = parseRecord(r, isCarStart);

                // Filter out prior starts that have neither a date nor a driver
                const prevStarts = (r.prevStarts || [])
                    .filter(h => {
                        const date   = (h.shortMeetDate || h.meetDate || '').trim();
                        const driver = (h.driverFullName || h.driver  || '').trim();
                        return date !== '' && driver !== '';
                    })
                    .map(h => {
                        const { kmNum, isCarStart: psIsCarStart, isBreak } = parseKmTime(h.kmTime);
                        const { position: psPosition, disqualified, DNF }  = parsePosition(h.result);
                        return {
                            date:           h.shortMeetDate || h.meetDate?.slice(0, 10) || null,
                            driver:         sanitize(h.driverFullName || h.driver || ''),
                            track:          h.trackCode   || h.track  || null,
                            distance:       parseInt(h.distance)       || null,
                            number:         parseInt(h.startTrack)     || null,
                            kmTime:         kmNum,
                            frontShoes:     h.frontShoes  || null,
                            rearShoes:      h.rearShoes   || null,
                            specialCart:    h.specialCart || null,
                            isCarStart:     psIsCarStart,
                            break:          isBreak,
                            disqualified,
                            DNF,
                            trackCondition: parseTrackCondition(h.trackCondition),
                            position:       psPosition,
                            // winOdd is ×10 (e.g. 152 = 15.2); store as actual value
                            odd:            h.winOdd    ? parseFloat(h.winOdd)    / 10    : null,
                            // firstPrize is in cents×10 (e.g. 1 000 000 = 100 €)
                            firstPrice:     h.firstPrize ? parseFloat(h.firstPrize) / 10000 : null,
                        };
                    });

                return {
                    number:            startNum,
                    name:              sanitize(r.horseName  || r.name         || ''),
                    coach:             sanitize(r.coachName  || r.trainerName  || ''),
                    driver:            sanitize(r.driverName || ''),
                    age:               parseInt(r.horseAge   || r.age)         || null,
                    gender:            encodeGender(r.gender),
                    frontShoes:        r.frontShoes          || null,
                    rearShoes:         r.rearShoes           || null,
                    frontShoesChanged: r.frontShoesChanged   === true,
                    rearShoesChanged:  r.rearShoesChanged    === true,
                    specialCart:       r.specialCart         || null,
                    scratched:         r.scratched           === true,
                    winPercentage:     parseWinPct(r),
                    bettingPercentage: parseBettingPct(r),
                    record,
                    isAutoRecord,
                    position,
                    prevStarts,
                };
            });

            races.push({ trackID: raceId, distance: raceDistance, isColdBlood, isCarStart, date: meetDate, runners });
        }

        console.log(`  ${card.trackName || cardId}: ${races.length} races processed`);
    }

    return races;
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────

async function main() {
    const today       = dateToISO(new Date());
    const yesterday   = addDays(today, -1);
    const forceUpdate = process.argv.includes('--force');

    let startDate, endDate;

    const args = process.argv.slice(2).filter(a => !a.startsWith('--'));

    if (args[0] && args[1]) {
        startDate = args[0];
        endDate   = args[1];
    } else if (args[0]) {
        startDate = args[0];
        endDate   = yesterday;
    } else {
        // Automatic: continue from the day after the latest stored date
        endDate   = yesterday;
        startDate = null;

        if (fs.existsSync(OUTPUT_FILE)) {
            try {
                const existing  = JSON.parse(fs.readFileSync(OUTPUT_FILE, 'utf8'));
                const allDates  = (existing.races || []).map(r => r.date).filter(Boolean).sort();
                if (allDates.length > 0) {
                    const lastDate = allDates[allDates.length - 1];
                    startDate = addDays(lastDate, 1);
                    console.log(`Last stored date: ${lastDate} → fetching from: ${startDate}`);
                }
            } catch (e) {
                console.warn('Could not read existing JSON, starting fresh:', e.message);
            }
        }

        if (!startDate) {
            console.error('Provide a start date: node scraper.js 2026-01-01');
            process.exit(1);
        }
    }

    console.log(`Fetching: ${startDate} → ${endDate}${forceUpdate ? ' (--force)' : ''}`);

    if (startDate > endDate) {
        console.log('Nothing to fetch — start date is after end date.');
        return;
    }

    // Load existing data
    let existing = { races: [] };
    if (fs.existsSync(OUTPUT_FILE)) {
        try {
            existing = JSON.parse(fs.readFileSync(OUTPUT_FILE, 'utf8'));
            console.log(`Loaded ${existing.races.length} existing races.`);
        } catch (e) {
            console.warn('Could not parse existing JSON, creating new file:', e.message);
        }
    }

    let newCount = 0;

    for (const date of dateRange(startDate, endDate)) {
        console.log(`\nDate: ${date}`);
        try {
            const dayRaces = await fetchDay(date);
            for (const race of dayRaces) {
                const idx = existing.races.findIndex(r => r.trackID === race.trackID);
                if (idx === -1) {
                    existing.races.push(race);
                    newCount++;
                } else if (forceUpdate) {
                    existing.races[idx] = race;
                    newCount++;
                }
            }
        } catch (e) {
            console.error(`  Error on ${date}:`, e.message);
        }
        await sleep(500);
    }

    console.log(`\nSaving ${existing.races.length} races (${newCount} new/updated)...`);
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(existing), 'utf8');

    const sizeMB = (fs.statSync(OUTPUT_FILE).size / 1e6).toFixed(1);
    console.log(`Done! ${OUTPUT_FILE} (${sizeMB} MB)`);
}

main().catch(e => { console.error('Fatal error:', e); process.exit(1); });
