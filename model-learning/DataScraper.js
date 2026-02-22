/**
 * TOTO-ENNUSTE-SCRAPER (Node.js versio)
 * Muunnettu MATLAB-skriptistä (Versio 2026 - 39 COLUMNS)
 *
 * Käyttö:
 *   node scraper.js --card 443071231 --lahto 8
 *   node scraper.js --card 443071231 --lahto all   (kaikki lähdöt)
 *
 * Asennus: npm install axios
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');

// --- ASETUKSET ---
const args = process.argv.slice(2);
const getArg = (flag) => { const i = args.indexOf(flag); return i !== -1 ? args[i+1] : null; };

const TARGET_CARD_ID = getArg('--card') || '443071231';
const HALUTTU_LAHTO  = getArg('--lahto') || '8';   // numero tai 'all'
const MAX_HISTORY    = 8;                            // sama kuin ravimalli.js

const BASE_URL = 'https://www.veikkaus.fi/api/toto-info/v1';
const HEADERS  = {
    'User-Agent': 'Mozilla/5.0',
    'Accept': 'application/json',
    'Accept-Language': 'fi-FI,fi;q=0.9'
};

// --- APUFUNKTIOT ---

/** Poistaa skandit ja erikoismerkit (sama kuin MATLAB puhdista) */
function puhdista(str) {
    if (!str) return '';
    return str
        .replace(/ä/g, 'a').replace(/ö/g, 'o').replace(/å/g, 'a')
        .replace(/Ä/g, 'A').replace(/Ö/g, 'O').replace(/Å/g, 'A')
        .replace(/[^a-zA-Z0-9\s\-\.\ :]/g, '')
        .trim();
}

/**
 * Parsii km-ajan merkkijonosta.
 * Palauttaa { kmNum, histIsAuto, isLaukka }
 * Esim. "1,14.3a" → { kmNum: 74.3, histIsAuto: 1, isLaukka: 0 }
 *       "1,15.2x" → { kmNum: 75.2, histIsAuto: 0, isLaukka: 1 }
 */
function parseKmAika(kmRaw) {
    if (!kmRaw) return { kmNum: 0, histIsAuto: 0, isLaukka: 0 };
    const s = String(kmRaw);
    const histIsAuto = s.includes('a') ? 1 : 0;
    const isLaukka   = s.includes('x') ? 1 : 0;
    // Poistetaan kirjaimet, muutetaan pilkku pisteeksi
    const cleaned = s.replace(/[^0-9,\.]/g, '').replace(',', '.');
    const kmNum = parseFloat(cleaned) || 0;
    return { kmNum, histIsAuto, isLaukka };
}

/** Parsii sijoituksen tekstistä → numero */
function parseSijoitus(sijRaw) {
    if (!sijRaw) return 10;
    const s = String(sijRaw).toLowerCase();
    const isHyl  = /[hdp]/.test(s) ? 1 : 0;
    const isKesk  = s.includes('k') ? 1 : 0;
    const numMatch = s.match(/^\d+/);
    let sijNum = 10;
    if (numMatch)       sijNum = parseInt(numMatch[0]);
    else if (isHyl)     sijNum = 20;
    else if (isKesk)    sijNum = 21;
    return { sijNum, isHyl, isKesk };
}

/** Sukupuoli tekstistä numeroksi (sama kuin MATLAB) */
function parseSukupuoli(genderStr) {
    if (!genderStr) return 2;
    if (genderStr === 'TAMMA') return 1;
    if (genderStr === 'ORI')   return 3;
    return 2; // RUUNA tai muu
}

/** Ennätys JSON:sta → { ennatys, isAutoRecord }
 *  isAutoRecord=1 jos ennätys on mobileStartRecord (autolähtöennätys)
 *  sama logiikka kuin MATLAB: isAutoRec = ~isempty(strfind(eMatch{1}, 'mobile'))
 */
function parseEnnatys(runner) {
    for (const key of ['handicapRaceRecord', 'mobileStartRecord', 'vaultStartRecord']) {
        const val = runner[key];
        if (val) {
            const num = parseFloat(String(val).replace(',', '.'));
            if (!isNaN(num) && num > 0) {
                return { ennatys: num, isAutoRecord: key === 'mobileStartRecord' ? 1 : 0 };
            }
        }
    }
    return { ennatys: 0, isAutoRecord: 0 };
}

/** Kengät tekstistä → HAS_SHOES / NO_SHOES / UNKNOWN */
function parseKengat(val) {
    if (!val) return 'UNKNOWN';
    const s = String(val).toUpperCase();
    if (s === 'TRUE' || s === 'YES' || s === '1') return 'HAS_SHOES';
    if (s === 'FALSE' || s === 'NO' || s === '0') return 'NO_SHOES';
    // Jotkut APIt palauttavat suoraan "HAS_SHOES"/"NO_SHOES"
    if (s === 'HAS_SHOES') return 'HAS_SHOES';
    if (s === 'NO_SHOES')  return 'NO_SHOES';
    return 'UNKNOWN';
}

/** Voitto-% laskenta historiasta (sama kuin MATLAB statBlock) */
function parseVoittoProsentti(runner) {
    try {
        const stats = runner.statistics?.total || runner.stats?.total;
        if (!stats) return 0;
        const starts = parseInt(stats.starts || stats.startCount || 0);
        const wins   = parseInt(stats.position1 || stats.wins || 0);
        if (starts > 0) return Math.round((wins / starts) * 100 * 100) / 100;
    } catch {}
    return 0;
}

/** Peli-% nykyisestä startista */
function parsePeliProsentti(runner) {
    try {
        const pct = runner.percentage ?? runner.winPercentage ?? 0;
        return parseFloat(pct) / 100;
    } catch {}
    return 0;
}

/** HTTP GET apari retry-logiikalla */
async function fetchJson(url, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const res = await axios.get(url, { headers: HEADERS, timeout: 20000 });
            return res.data;
        } catch (err) {
            if (i === retries - 1) throw err;
            console.warn(`  Uudelleenyritys (${i+1}/${retries}): ${url}`);
            await new Promise(r => setTimeout(r, 1500));
        }
    }
}

// --- TYHJÄ RIVI (historia puuttuu) ---
function tyhjaRivi(raceId, nro, nimi, valmentaja, currOhj, ika, sukupuoli, isSH,
                   kEtu, kTaka, kEtuCh, kTakaCh, currSpecCart, isScratched,
                   currDist, currIsAuto, currDate, peliP, voittoP, ennatys, isAutoRecord) {
    return {
        RaceID: raceId, Nro: nro, Nimi: nimi, Valmentaja: valmentaja,
        Current_Ohjastaja: currOhj, Ika: ika, Sukupuoli: sukupuoli,
        Is_Suomenhevonen: isSH, Kengat_Etu: kEtu, Kengat_Taka: kTaka,
        Kengat_etu_changed: kEtuCh, Kengat_taka_changed: kTakaCh,
        Current_Special_Cart: currSpecCart, Scratched: isScratched,
        Current_Distance: currDist, Current_Is_Auto: currIsAuto,
        Current_Start_Date: currDate, Peli_pros: peliP, Voitto_pros: voittoP,
        Ennatys_nro: ennatys, Is_Auto_Record: isAutoRecord,
        // Historia-sarakkeet tyhjinä
        Hist_PVM: '', Ohjastaja: '', Rata: '', Matka: 0, RataNro: 0,
        Km_aika: 0, Hist_kengat_etu: '', Hist_kengat_taka: '', Hist_Special_Cart: '',
        Hist_Is_Auto: 0, Laukka: 0, Hylatty: 0, Keskeytys: 0,
        Track_Condition: '', Hist_Sij: 10, Kerroin: 0, Palkinto: 0, SIJOITUS: 0
    };
}

// --- PÄÄFUNKTIO: SCRAPE YKSI LÄHTÖ ---
async function scrapeOneLahto(cardId, raceId, raceNum, currDist, isSH, currIsAuto, currDate) {
    console.log(`  Haetaan lähtö ${raceNum} (raceId: ${raceId}, ${currDist}m)...`);

    const runnersData = await fetchJson(`${BASE_URL}/race/${raceId}/runners`);

    // API voi palauttaa runners-arrayn suoraan tai wrapattuna
    const runners = Array.isArray(runnersData) ? runnersData
        : (runnersData.runners || runnersData.data || []);

    const rivit = [];

    for (const runner of runners) {
        // --- NYKYHETKI ---
        const nro         = parseInt(runner.startNumber || runner.number || 0);
        const nimi        = puhdista(runner.horseName || runner.name || '');
        const valmentaja  = puhdista(runner.coachName || runner.trainerName || 'Unknown');
        const currOhj     = puhdista(runner.driverName || '');
        const ika         = parseInt(runner.horseAge || runner.age || 0);
        const sukupuoli   = parseSukupuoli(runner.gender);
        const kEtu        = parseKengat(runner.frontShoes);
        const kTaka       = parseKengat(runner.rearShoes);
        const kEtuCh      = runner.frontShoesChanged === true ? 1 : 0;
        const kTakaCh     = runner.rearShoesChanged  === true ? 1 : 0;
        const currSpecCart= runner.specialCart || 'UNKNOWN';
        const isScratched = runner.scratched === true ? 1 : 0;
        const { ennatys, isAutoRecord } = parseEnnatys(runner);
        const peliP       = parsePeliProsentti(runner);
        const voittoP     = parseVoittoProsentti(runner);

        // --- HISTORIA (priorStarts) ---
        const priorStarts = runner.priorStarts || runner.previousStarts || runner.starts || [];

        if (priorStarts.length === 0) {
            rivit.push(tyhjaRivi(raceId, nro, nimi, valmentaja, currOhj, ika, sukupuoli, isSH,
                kEtu, kTaka, kEtuCh, kTakaCh, currSpecCart, isScratched,
                currDist, currIsAuto, currDate, peliP, voittoP, ennatys, isAutoRecord));
        } else {
            // Otetaan MAX_HISTORY viimeisintä starttia
            const otettavat = priorStarts.slice(0, MAX_HISTORY);

            for (const s of otettavat) {
                const oNimi  = puhdista(s.driverFullName || s.driverName || '');
                const rNimi  = puhdista(s.trackCode || s.track || '');
                const tCond  = puhdista(s.trackCondition || 'unknown') || 'unknown';

                const { kmNum, histIsAuto, isLaukka } = parseKmAika(s.kmTime);

                const hKEtu    = parseKengat(s.frontShoes);
                const hKTaka   = parseKengat(s.rearShoes);
                const hSpecCart= s.specialCart || 'UNKNOWN';

                const { sijNum, isHyl, isKesk } = parseSijoitus(s.result);

                const matka   = parseInt(s.distance || 0);
                const rataNro = parseInt(s.startTrack || 0);
                const kerroin = parseFloat(s.winOdd || 0) / 10;
                const palkinto= parseFloat(s.firstPrize || 0) / 100;
                const histPvm = puhdista(s.shortMeetDate || s.meetDate || '');

                rivit.push({
                    RaceID: raceId, Nro: nro, Nimi: nimi, Valmentaja: valmentaja,
                    Current_Ohjastaja: currOhj, Ika: ika, Sukupuoli: sukupuoli,
                    Is_Suomenhevonen: isSH, Kengat_Etu: kEtu, Kengat_Taka: kTaka,
                    Kengat_etu_changed: kEtuCh, Kengat_taka_changed: kTakaCh,
                    Current_Special_Cart: currSpecCart, Scratched: isScratched,
                    Current_Distance: currDist, Current_Is_Auto: currIsAuto,
                    Current_Start_Date: currDate, Peli_pros: peliP,
                    Voitto_pros: voittoP, Ennatys_nro: ennatys, Is_Auto_Record: isAutoRecord,
                    Hist_PVM: histPvm, Ohjastaja: oNimi, Rata: rNimi,
                    Matka: matka, RataNro: rataNro, Km_aika: kmNum,
                    Hist_kengat_etu: hKEtu, Hist_kengat_taka: hKTaka,
                    Hist_Special_Cart: hSpecCart, Hist_Is_Auto: histIsAuto,
                    Laukka: isLaukka, Hylatty: isHyl, Keskeytys: isKesk,
                    Track_Condition: tCond, Hist_Sij: sijNum,
                    Kerroin: kerroin, Palkinto: palkinto, SIJOITUS: 0
                });
            }
        }
    }

    return rivit;
}

// --- CSV-KIRJOITUS (puolipisteerotettu, UTF-8) ---
const HEADERS_ORDER = [
    'RaceID','Nro','Nimi','Valmentaja','Current_Ohjastaja','Ika','Sukupuoli',
    'Is_Suomenhevonen','Kengat_Etu','Kengat_Taka','Kengat_etu_changed','Kengat_taka_changed',
    'Current_Special_Cart','Scratched','Current_Distance','Current_Is_Auto','Current_Start_Date',
    'Peli_pros','Voitto_pros','Ennatys_nro','Is_Auto_Record',
    'Hist_PVM','Ohjastaja','Rata','Matka','RataNro','Km_aika',
    'Hist_kengat_etu','Hist_kengat_taka','Hist_Special_Cart','Hist_Is_Auto','Laukka',
    'Hylatty','Keskeytys','Track_Condition','Hist_Sij','Kerroin','Palkinto','SIJOITUS'
];

function kirjoitaCsv(rivit, tiedostonimi) {
    const lines = [HEADERS_ORDER.join(';')];
    for (const rivi of rivit) {
        const line = HEADERS_ORDER.map(h => {
            const val = rivi[h] ?? '';
            // Ympäröi puolipisteellä tai lainausmerkillä jos tarve
            const s = String(val);
            return s.includes(';') ? `"${s}"` : s;
        }).join(';');
        lines.push(line);
    }
    fs.writeFileSync(tiedostonimi, lines.join('\n'), 'utf8');
    console.log(`\n✓ Tallennettu: ${tiedostonimi} (${rivit.length} riviä)`);
}

// --- PÄÄOHJELMA ---
async function main() {
    console.log(`\n=== TOTO-SCRAPER ===`);
    console.log(`Kortti: ${TARGET_CARD_ID}, Lähtö: ${HALUTTU_LAHTO}`);

    // 1. Hae kortin lähdöt
    console.log('\nHaetaan kortin lähdöt...');
    const racesData = await fetchJson(`${BASE_URL}/card/${TARGET_CARD_ID}/races`);
    const races = Array.isArray(racesData) ? racesData
        : (racesData.races || racesData.data || []);

    // 2. Hae kortin päivämäärä
    const cardInfo = await fetchJson(`${BASE_URL}/card/${TARGET_CARD_ID}`);
    const currDate = puhdista(cardInfo.meetDate || cardInfo.date || '');

    // 3. Suodata haluttu lähtö / kaikki
    const halututLahdot = HALUTTU_LAHTO === 'all'
        ? races
        : races.filter(r => String(r.number || r.raceNumber) === String(HALUTTU_LAHTO));

    if (halututLahdot.length === 0) {
        console.error(`Lähtöä ${HALUTTU_LAHTO} ei löytynyt kortilta ${TARGET_CARD_ID}`);
        console.log('Saatavilla olevat lähdöt:', races.map(r => r.number || r.raceNumber).join(', '));
        process.exit(1);
    }

    // 4. Scrape jokainen lähtö
    let kaikkiRivit = [];

    for (const race of halututLahdot) {
        const raceId    = String(race.raceId || race.id);
        const raceNum   = race.number || race.raceNumber;
        const currDist  = parseInt(race.distance || 2100);
        const isSH      = (race.breed === 'K' || race.breed === 'FINNHORSE') ? 1 : 0;
        const currIsAuto= (race.startType === 'CAR_START' || race.startType === 'AUTO') ? 1 : 0;

        try {
            const rivit = await scrapeOneLahto(
                TARGET_CARD_ID, raceId, raceNum,
                currDist, isSH, currIsAuto, currDate
            );
            kaikkiRivit = kaikkiRivit.concat(rivit);
            console.log(`  → ${rivit.length} riviä (${new Set(rivit.map(r=>r.Nimi)).size} hevosta)`);
        } catch (err) {
            console.error(`  ✗ Lähtö ${raceNum} epäonnistui: ${err.message}`);
        }
    }

    // 5. Kirjoita CSV
    const suffix = HALUTTU_LAHTO === 'all' ? 'kaikki' : `lahto${HALUTTU_LAHTO}`;
    const tiedostonimi = `Ravit_${TARGET_CARD_ID}_${suffix}.csv`;
    kirjoitaCsv(kaikkiRivit, tiedostonimi);

    // 6. Tulosta yhteenveto
    const hevoset = new Set(kaikkiRivit.map(r => r.Nimi));
    console.log(`\nYhteenveto:`);
    console.log(`  Lähtöjä:  ${halututLahdot.length}`);
    console.log(`  Hevosia:  ${hevoset.size}`);
    console.log(`  Rivejä:   ${kaikkiRivit.length}`);
    console.log(`  Hist/hev: ${(kaikkiRivit.length / hevoset.size).toFixed(1)} keskimäärin`);
}

main().catch(err => {
    console.error('\n✗ Virhe:', err.message);
    if (err.response) {
        console.error('  HTTP status:', err.response.status);
        console.error('  URL:', err.response.config?.url);
    }
    process.exit(1);
});
