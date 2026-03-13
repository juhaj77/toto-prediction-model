#!/usr/bin/env node
/*
 Summarize Mixed-Model runs → Markdown.
 - Scans model-training/model-mixed/runs/** for run folders.
 - Prefers metrics from: model_best_ndcg3.json → model.json → model_best_r3.json → model_best_auc.json → model_last.json
 - Builds a ranked table (default sort: val_ndcg3 desc) and per-run blocks with a ready-to-copy command line.

 Usage (from project root):
   node model-training/tools/summarize_runs.js [--sort=val_ndcg3|val_auc|val_r3|val_loss] [--top=30] [--filter=substr]
   node model-training/tools/summarize_runs.js --backendHint

 Output:
   model-training/model-mixed/runs/summary.md
*/

'use strict';

const fs = require('fs');
const path = require('path');

const RUNS_ROOT = path.join(__dirname, '..', 'model-mixed', 'runs');
const OUTPUT_MD = path.join(RUNS_ROOT, 'summary.md');

const PREFERRED_FILES = [
  'model_best_ndcg3.json',
  'model.json',
  'model_best_r3.json',
  'model_best_auc.json',
  'model_last.json',
];

function parseArgs() {
  const args = { sort: 'val_ndcg3', top: 50, filter: '', backendHint: false };
  for (const a of process.argv.slice(2)) {
    const m = a.match(/^--([^=]+)(=(.*))?$/);
    if (!m) continue;
    const k = m[1];
    const v = m[3];
    if (k === 'sort' && v) args.sort = v;
    else if (k === 'top' && v && !isNaN(Number(v))) args.top = Number(v);
    else if (k === 'filter' && v) args.filter = v.toLowerCase();
    else if (k === 'backendHint') args.backendHint = true;
  }
  return args;
}

function isDir(p) { try { return fs.statSync(p).isDirectory(); } catch { return false; } }
function isFile(p) { try { return fs.statSync(p).isFile(); } catch { return false; } }

function readJson(p) { try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch { return null; } }

function findBestFile(runDir) {
  for (const f of PREFERRED_FILES) {
    const p = path.join(runDir, f);
    if (isFile(p)) return p;
  }
  return null;
}

function extractInfo(modelJsonPath) {
  const payload = readJson(modelJsonPath);
  if (!payload) return null;
  const ti = payload.trainingInfo || {};
  // Backfill numeric conversions if strings
  function num(x) { return typeof x === 'number' ? x : (x != null ? Number(x) : undefined); }
  const info = {
    sourceFile: path.basename(modelJsonPath),
    savedAt: ti.savedAt || null,
    epoch: num(ti.epoch),
    val_loss: num(ti.val_loss),
    val_acc: num(ti.val_acc),
    val_r3: num(ti.val_r3),
    val_p3: num(ti.val_p3),
    val_auc: num(ti.val_auc),
    val_ndcg3: num(ti.val_ndcg3),
    val_hit1: num(ti.val_hit1),
    val_ap_macro: num(ti.val_ap_macro),
    val_logloss_race: num(ti.val_logloss_race),
    best_by: ti.best_by || 'val_loss',
    learningRate: num(ti.learningRate),
    dataStartDate: ti.dataStartDate || null,
    dataEndDate: ti.dataEndDate || null,
    totalRaces: num(ti.totalRaces),
    totalRunners: num(ti.totalRunners),
    recommended_threshold: num(ti.recommended_threshold),
    recommended_K: num(ti.recommended_K) || 3,
    tfjs_backend: ti.tfjs_backend || 'unknown',
    run_options: ti.run_options || null,
  };
  return info;
}

function reconstructCommand(runId, info) {
  // Prefer programmatic node -e with explicit params. We avoid embedding runId to make it reusable.
  const opts = info.run_options || {};
  const keysOrder = [
    'epochs','temporalSplit','scheduler','warmupEpochs','learningRate','minLearningRate','batchSize',
    'bestBy','auxLossWeight','useListNet','attnHeads','embedDim','ffnDim','runnerProjDim','runnerLstm2Units',
    'dropout','outDropout','outUnits','l2','valFraction','plateauPatience','plateauFactor','swapBNtoLN','useLayerNormInStatic'
  ];
  const parts = [];
  for (const k of keysOrder) {
    if (opts[k] == null) continue;
    const v = opts[k];
    if (typeof v === 'string') parts.push(`${k}: '${v}'`);
    else parts.push(`${k}: ${v}`);
  }
  const body = parts.join(', ');
  const cmdProgrammatic = `node -e "require('./model-training/model_mixed_runner_and_race').runTraining({ runId: '${runId}', ${body} })"`;

  // Backend hint: suggest --backend=cpu|gpu or env var based on recorded backend
  const be = (info.tfjs_backend || '').toLowerCase();
  let backendHint = '';
  if (be.includes('gpu')) {
    backendHint = "# runs with GPU backend (default if installed): add --backend=gpu when using CLI entrypoint";
  } else if (be.includes('node')) {
    backendHint = "# recorded with CPU backend: use --backend=cpu or set $env:TFJS_BACKEND='cpu'";
  }

  const envHint = be.includes('gpu')
    ? "$env:TFJS_BACKEND='gpu'; "
    : (be.includes('node') ? "$env:TFJS_BACKEND='cpu'; " : '');
  const cmdWithEnv = envHint + cmdProgrammatic;

  return { cmdProgrammatic: cmdProgrammatic, cmdWithEnv, backendHint };
}

function formatNumber(x, digits=4) { return (typeof x === 'number' && isFinite(x)) ? x.toFixed(digits) : '—'; }

function buildMarkdown(rows, args) {
  const lines = [];
  lines.push('### Mixed-model runs — summary');
  lines.push('');
  lines.push(`- Scanned: \\` + RUNS_ROOT.replace(/\\/g, '/') );
  lines.push(`- Sorted by: \`${args.sort}\` (top ${args.top}${args.filter ? `, filter contains "${args.filter}"` : ''})`);
  lines.push('');
  lines.push('| # | runId | file | epoch | val_ndcg@3 | val_hit@1 | val_r@3 | val_p@3 | val_auc | backend |');
  lines.push('|:-:|:------|:-----|------:|-----------:|----------:|--------:|--------:|-------:|:-------|');

  rows.forEach((r, i) => {
    lines.push(`| ${i+1} | ${r.runId} | ${r.info.sourceFile} | ${r.info.epoch ?? '—'} | ${formatNumber(r.info.val_ndcg3)} | ${formatNumber(r.info.val_hit1)} | ${formatNumber(r.info.val_r3)} | ${formatNumber(r.info.val_p3)} | ${formatNumber(r.info.val_auc)} | ${r.info.tfjs_backend} |`);
  });

  lines.push('');
  lines.push('---');
  lines.push('');

  for (const r of rows) {
    const { cmdProgrammatic, cmdWithEnv, backendHint } = reconstructCommand(r.runId, r.info);
    lines.push(`#### ${r.runId}`);
    lines.push('');
    lines.push('- Folder: `' + r.runDir.replace(/\\/g, '/') + '`');
    lines.push(`- Source file: \`${r.info.sourceFile}\``);
    lines.push(`- Saved at: \`${r.info.savedAt || 'n/a'}\``);
    lines.push(`- Metric snapshot: ndcg@3=${formatNumber(r.info.val_ndcg3)}, hit@1=${formatNumber(r.info.val_hit1)}, r@3=${formatNumber(r.info.val_r3)}, p@3=${formatNumber(r.info.val_p3)}, auc=${formatNumber(r.info.val_auc)}, val_loss=${formatNumber(r.info.val_loss)}`);
    lines.push(`- Data: races=${r.info.totalRaces ?? '—'}, runners=${r.info.totalRunners ?? '—'}, ${r.info.dataStartDate || '—'} → ${r.info.dataEndDate || '—'}`);
    if (backendHint) lines.push(`- Backend: ${r.info.tfjs_backend} (${backendHint})`);
    lines.push('');
    lines.push('Ready-to-copy command (programmatic):');
    lines.push('');
    lines.push('```');
    lines.push(cmdProgrammatic);
    lines.push('```');
    if (cmdWithEnv !== cmdProgrammatic) {
      lines.push('');
      lines.push('PowerShell (with backend env override):');
      lines.push('');
      lines.push('```powershell');
      lines.push(cmdWithEnv);
      lines.push('```');
    }
    if (r.info.run_options) {
      lines.push('');
      lines.push('Parameters used:');
      lines.push('');
      lines.push('```json');
      try { lines.push(JSON.stringify({ runId: r.runId, ...r.info.run_options }, null, 2)); } catch { lines.push('{ /* parameters unavailable */ }'); }
      lines.push('```');
    }
    lines.push('');
  }

  return lines.join('\n');
}

function main() {
  const args = parseArgs();
  if (!isDir(RUNS_ROOT)) {
    console.error('Runs directory not found: ' + RUNS_ROOT);
    process.exit(1);
  }

  const runIds = fs.readdirSync(RUNS_ROOT).filter(name => isDir(path.join(RUNS_ROOT, name)));
  const rows = [];

  for (const runId of runIds) {
    if (args.filter && !runId.toLowerCase().includes(args.filter)) continue;
    const runDir = path.join(RUNS_ROOT, runId);
    const f = findBestFile(runDir);
    if (!f) continue;
    const info = extractInfo(f);
    if (!info) continue;
    rows.push({ runId, runDir, info });
  }

  function sortVal(r) {
    const i = r.info;
    switch (args.sort) {
      case 'val_auc': return i.val_auc ?? -Infinity;
      case 'val_r3':  return i.val_r3  ?? -Infinity;
      case 'val_loss':return (i.val_loss != null ? -i.val_loss : -Infinity); // smaller loss is better
      case 'val_ndcg3':
      default:        return i.val_ndcg3 ?? -Infinity;
    }
  }

  rows.sort((a,b)=> sortVal(b) - sortVal(a));
  const limited = rows.slice(0, Math.max(1, args.top));

  const md = buildMarkdown(limited, args);
  fs.writeFileSync(OUTPUT_MD, md, 'utf8');
  console.log('✔ Wrote summary → ' + OUTPUT_MD);
  console.log('Runs found:', rows.length, ' | Included in summary:', limited.length);
}

if (require.main === module) {
  try { main(); } catch (e) { console.error(e); process.exit(1); }
}
