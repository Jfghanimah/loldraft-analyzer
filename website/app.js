// ── Constants ──────────────────────────────────────────────────────────────
const ROLES = ['Top', 'Jgl', 'Mid', 'Bot', 'Sup'];

// DDragon names that differ from the canonical champion name
const DDRAGON_SPECIAL = {
  "Aurelion Sol":   "AurelionSol",
  "Bel'Veth":       "Belveth",
  "Cho'Gath":       "Chogath",
  "Dr. Mundo":      "DrMundo",
  "Jarvan IV":      "JarvanIV",
  "Kai'Sa":         "Kaisa",
  "Kha'Zix":        "Khazix",
  "Kog'Maw":        "KogMaw",
  "LeBlanc":        "Leblanc",
  "Lee Sin":        "LeeSin",
  "Master Yi":      "MasterYi",
  "Miss Fortune":   "MissFortune",
  "Nunu & Willump": "Nunu",
  "Rek'Sai":        "RekSai",
  "Renata Glasc":   "Renata",
  "Tahm Kench":     "TahmKench",
  "Twisted Fate":   "TwistedFate",
  "Vel'Koz":        "Velkoz",
  "Wukong":         "MonkeyKing",
  "Xin Zhao":       "XinZhao",
};

// ── State ──────────────────────────────────────────────────────────────────
let ddragonVersion  = '15.7.1';
let champions       = [];           // sorted list from /api/champions
let champTagMap     = {};           // champion name → DDragon tag array
let activePicker    = null;         // { team, idx, isBan } | null
let activeTagFilter = null;         // active class filter string or null
let _retryFn        = null;         // last retryable action

// ── Helpers ────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function champIconUrl(name) {
  const file = DDRAGON_SPECIAL[name] ?? name.replace(/[' .]/g, '');
  return `https://ddragon.leagueoflegends.com/cdn/${ddragonVersion}/img/champion/${file}.png`;
}

function setStatus(msg, type = '', retryFn = null) {
  const el = $('lookup-status');
  el.textContent = msg;
  el.className   = 'status-msg' + (type ? ` ${type}` : '');
  const retryBtn = $('btn-retry');
  if (retryBtn) {
    _retryFn = retryFn ?? null;
    retryBtn.toggleAttribute('hidden', !retryFn);
  }
}

// ── DDragon version ────────────────────────────────────────────────────────
async function fetchDDragonVersion() {
  try {
    const res  = await fetch('https://ddragon.leagueoflegends.com/api/versions.json');
    const list = await res.json();
    if (list?.[0]) ddragonVersion = list[0];
  } catch { /* keep fallback */ }
}

// ── Champion list ──────────────────────────────────────────────────────────
async function loadChampions() {
  try {
    const res  = await fetch('/api/champions');
    if (!res.ok) return;
    const data = await res.json();
    champions  = data.champions ?? [];
    buildPickerGrid();
  } catch { /* server offline — picker stays empty, text input still works */ }
}

// ── Champion class tags (from DDragon) ─────────────────────────────────────
async function loadChampionTags() {
  if (!champions.length) return;
  try {
    const res  = await fetch(`https://ddragon.leagueoflegends.com/cdn/${ddragonVersion}/data/en_US/champion.json`);
    const data = await res.json();
    const ddIdToTags = {};
    for (const [id, entry] of Object.entries(data.data)) {
      ddIdToTags[id] = entry.tags ?? [];
    }
    for (const name of champions) {
      const ddId = DDRAGON_SPECIAL[name] ?? name.replace(/[' .]/g, '');
      champTagMap[name] = ddIdToTags[ddId] ?? [];
    }
    buildFilterButtons();
  } catch { /* no tag data — filters won't appear */ }
}

function buildFilterButtons() {
  const container = $('picker-filters');
  if (!container) return;
  const tagOrder = ['Fighter', 'Tank', 'Mage', 'Assassin', 'Marksman', 'Support'];
  const usedTags = new Set(Object.values(champTagMap).flat());
  const tags = tagOrder.filter(t => usedTags.has(t));
  if (!tags.length) return;

  container.innerHTML = '';
  const allBtn = document.createElement('button');
  allBtn.className = 'picker-filter-btn active';
  allBtn.textContent = 'All';
  allBtn.dataset.tag = '';
  allBtn.addEventListener('click', () => setTagFilter(''));
  container.appendChild(allBtn);

  tags.forEach(tag => {
    const btn = document.createElement('button');
    btn.className = 'picker-filter-btn';
    btn.textContent = tag;
    btn.dataset.tag = tag;
    btn.addEventListener('click', () => setTagFilter(tag));
    container.appendChild(btn);
  });
}

function setTagFilter(tag) {
  activeTagFilter = tag || null;
  document.querySelectorAll('.picker-filter-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tag === (tag || ''));
  });
  refreshPickerUsed($('picker-search').value);
}

// ── Draft slot DOM ─────────────────────────────────────────────────────────
function buildSlots(containerId, team) {
  const container = $(containerId);
  container.innerHTML = '';
  ROLES.forEach((role, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'slot-wrap';

    const playerLabel = document.createElement('span');
    playerLabel.className = 'slot-player';
    playerLabel.id = `player-${team}-${i}`;
    wrap.appendChild(playerLabel);

    const slot = document.createElement('div');
    slot.className = 'slot';
    slot.innerHTML = `
      <button class="slot-icon-btn empty" id="iconbtn-${team}-${i}"
              title="Pick champion" aria-label="Open champion picker for ${role}">
        <img src="${champIconUrl('Aatrox')}" alt="" />
      </button>
      <span class="slot-role">${role}</span>
      <input id="slot-${team}-${i}" class="slot-input" type="text"
             autocomplete="off" spellcheck="false" />
    `;
    wrap.appendChild(slot);

    const statsRow = document.createElement('div');
    statsRow.className = 'slot-stats';
    statsRow.id        = `stats-${team}-${i}`;
    statsRow.setAttribute('hidden', '');
    statsRow.innerHTML = `
      <span class="rank-badge"    id="rank-${team}-${i}"></span>
      <span class="mastery-badge" id="mastery-${team}-${i}"></span>
      <span class="champ-wr"      id="wr-${team}-${i}"></span>
    `;
    wrap.appendChild(statsRow);
    container.appendChild(wrap);

    $(`iconbtn-${team}-${i}`).addEventListener('click', () => openPicker(team, i, false));

    const input = $(`slot-${team}-${i}`);
    input.addEventListener('input',  () => syncSlotIcon(team, i));
    input.addEventListener('change', () => syncSlotIcon(team, i));
  });
}

function setPlayerName(team, i, name) {
  const el = $(`player-${team}-${i}`);
  if (el) el.textContent = name ?? '';
}

function setPlayerStats(team, i, { rank, mastery, wr, games } = {}) {
  const statsEl   = $(`stats-${team}-${i}`);
  const rankEl    = $(`rank-${team}-${i}`);
  const masteryEl = $(`mastery-${team}-${i}`);
  const wrEl      = $(`wr-${team}-${i}`);
  if (!statsEl) return;

  const hasData = rank || mastery != null || wr != null;
  if (!hasData) {
    statsEl.setAttribute('hidden', '');
    [rankEl, masteryEl, wrEl].forEach(el => { if (el) el.textContent = ''; });
    return;
  }

  if (rank) {
    const tier = rank.split(' ')[0].toLowerCase();
    rankEl.textContent = rank;
    rankEl.className   = `rank-badge rank-${tier}`;
  }
  if (mastery != null) {
    masteryEl.textContent = `M${mastery}`;
    masteryEl.className   = `mastery-badge${mastery >= 7 ? ' mastery-high' : ''}`;
  }
  if (wr != null) {
    wrEl.textContent = games != null ? `${wr}% · ${games}g` : `${wr}%`;
  }
  statsEl.removeAttribute('hidden');
}

function syncSlotIcon(team, i) {
  const name = $(`slot-${team}-${i}`).value.trim();
  const btn  = $(`iconbtn-${team}-${i}`);
  if (champions.includes(name)) {
    btn.querySelector('img').src = champIconUrl(name);
    btn.classList.remove('empty');
  } else {
    btn.classList.add('empty');
  }
}

function setSlot(team, i, name) {
  $(`slot-${team}-${i}`).value = name;
  syncSlotIcon(team, i);
}

function getSlot(team, i) {
  return $(`slot-${team}-${i}`).value.trim();
}

// ── Ban slot DOM ───────────────────────────────────────────────────────────
function buildBans(groupId, team) {
  const group = $(groupId);
  group.innerHTML = '';
  for (let i = 0; i < 5; i++) {
    const btn = document.createElement('button');
    btn.className  = 'ban-slot';
    btn.id         = `ban-${team}-${i}`;
    btn.title      = 'Ban champion';
    btn.dataset.champ = '';
    btn.innerHTML  = `<img src="${champIconUrl('Aatrox')}" alt="" style="display:none" />`;
    btn.addEventListener('click', () => openPicker(team, i, true));
    group.appendChild(btn);
  }
}

function setBan(team, i, name) {
  const btn = $(`ban-${team}-${i}`);
  btn.dataset.champ = name;
  const img = btn.querySelector('img');
  if (name) {
    img.src   = champIconUrl(name);
    img.style.display = '';
    btn.classList.add('filled');
    btn.title = `Ban: ${name} (click to change)`;
  } else {
    img.style.display = 'none';
    btn.classList.remove('filled');
    btn.title = 'Ban champion';
  }
}

function getBan(team, i) {
  return $(`ban-${team}-${i}`)?.dataset.champ ?? '';
}

// ── Champion picker ────────────────────────────────────────────────────────
function buildPickerGrid() {
  const grid = $('picker-grid');
  grid.innerHTML = '';

  if (!champions.length) {
    grid.innerHTML = '<p class="picker-empty-msg">No champions loaded. Is the server running?</p>';
    return;
  }

  champions.forEach(name => {
    const tile = document.createElement('button');
    tile.className    = 'picker-tile';
    tile.dataset.name = name;
    tile.innerHTML    = `
      <img src="${champIconUrl(name)}" alt="${name}" loading="lazy"
           onerror="this.style.opacity='0'" />
      <span>${name}</span>
    `;
    tile.addEventListener('click', () => selectChampion(name));
    grid.appendChild(tile);
  });
}

function getUsedChampions() {
  const used = new Set();
  ['blue', 'red'].forEach(team => {
    for (let i = 0; i < 5; i++) {
      const pick = getSlot(team, i);
      const ban  = getBan(team, i);
      if (pick) used.add(pick);
      if (ban)  used.add(ban);
    }
  });
  // Exclude the slot currently being edited so it's not marked as used
  if (activePicker) {
    const cur = activePicker.isBan
      ? getBan(activePicker.team, activePicker.idx)
      : getSlot(activePicker.team, activePicker.idx);
    used.delete(cur);
  }
  return used;
}

function openPicker(team, idx, isBan) {
  activePicker = { team, idx, isBan };

  const role  = ROLES[idx];
  const label = isBan ? `Ban — ${team === 'blue' ? 'Blue' : 'Red'} ${role}`
                      : `Pick — ${team === 'blue' ? 'Blue' : 'Red'} ${role}`;
  $('picker-title').textContent = label;

  // Reset search + tag filter
  $('picker-search').value = '';
  activeTagFilter = null;
  document.querySelectorAll('.picker-filter-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tag === '');
  });
  refreshPickerUsed('');

  $('picker-modal').removeAttribute('hidden');
  $('picker-search').focus();
}

function closePicker() {
  $('picker-modal').setAttribute('hidden', '');
  activePicker = null;
}

function refreshPickerUsed(query) {
  const used  = getUsedChampions();
  const lower = query.toLowerCase();
  document.querySelectorAll('.picker-tile').forEach(tile => {
    const name      = tile.dataset.name;
    const matchText = !lower || name.toLowerCase().includes(lower);
    const matchTag  = !activeTagFilter || (champTagMap[name] ?? []).includes(activeTagFilter);
    tile.toggleAttribute('hidden', !matchText || !matchTag);
    tile.classList.toggle('used', used.has(name));
  });
}

function selectChampion(name) {
  if (!activePicker) return;
  const { team, idx, isBan } = activePicker;
  if (isBan) setBan(team, idx, name);
  else       setSlot(team, idx, name);
  closePicker();
}

// ── Read / fill draft ──────────────────────────────────────────────────────
function readDraft() {
  return [
    ...ROLES.map((_, i) => getSlot('blue', i)),
    ...ROLES.map((_, i) => getSlot('red',  i)),
  ];
}

function fillDraft({ blue_team, red_team, blue_players, red_players, blue_stats, red_stats, blue_bans, red_bans }) {
  blue_team.forEach((name, i) => setSlot('blue', i, name));
  red_team.forEach( (name, i) => setSlot('red',  i, name));
  if (blue_players) blue_players.forEach((name, i) => setPlayerName('blue', i, name));
  if (red_players)  red_players.forEach( (name, i) => setPlayerName('red',  i, name));
  if (blue_stats)   blue_stats.forEach(  (s,    i) => setPlayerStats('blue', i, s));
  if (red_stats)    red_stats.forEach(   (s,    i) => setPlayerStats('red',  i, s));
  if (blue_bans)    blue_bans.forEach(   (name, i) => { if (name) setBan('blue', i, name); });
  if (red_bans)     red_bans.forEach(    (name, i) => { if (name) setBan('red',  i, name); });
}

function clearAll() {
  ['blue', 'red'].forEach(team => {
    for (let i = 0; i < 5; i++) {
      setSlot(team, i, '');
      setBan(team, i, '');
      setPlayerName(team, i, '');
      setPlayerStats(team, i);
    }
  });
  $('lane-matchups').innerHTML = '<p class="placeholder-text">Analyze a draft to see per-lane breakdowns.</p>';
  $('draft-strengths').innerHTML = '<p class="placeholder-text">Analyze a draft to see composition breakdown.</p>';
  $('analysis-section').setAttribute('hidden', '');
  $('prediction-result').className = 'prediction-empty';
  $('prediction-result').innerHTML = '<span>Fill the draft<br/>and click <strong>Analyze</strong></span>';
  setStatus('');
  history.replaceState(null, '', location.pathname);
}

// ── Share link ─────────────────────────────────────────────────────────────
function buildShareUrl() {
  const p  = new URLSearchParams();
  p.set('bd', ROLES.map((_, i) => getSlot('blue', i)).join(','));
  p.set('rd', ROLES.map((_, i) => getSlot('red',  i)).join(','));
  const bb = ROLES.map((_, i) => getBan('blue', i)).join(',');
  const rb = ROLES.map((_, i) => getBan('red',  i)).join(',');
  if (bb.replace(/,/g, '')) p.set('bb', bb);
  if (rb.replace(/,/g, '')) p.set('rb', rb);
  return `${location.origin}${location.pathname}?${p}`;
}

async function handleShare() {
  const url = buildShareUrl();
  try {
    await navigator.clipboard.writeText(url);
  } catch {
    prompt('Copy this link:', url);
    return;
  }
  const btn = $('btn-share');
  btn.textContent = 'Copied!';
  btn.classList.add('copied');
  setTimeout(() => { btn.textContent = 'Share'; btn.classList.remove('copied'); }, 2000);
}

function loadFromUrl() {
  const p = new URLSearchParams(location.search);
  const parse = key => (p.get(key) ?? '').split(',');

  const bd = parse('bd'), rd = parse('rd');
  const bb = parse('bb'), rb = parse('rb');

  const hasPickData = bd.some(Boolean) || rd.some(Boolean);
  if (!hasPickData) return;

  bd.forEach((name, i) => { if (name) setSlot('blue', i, name); });
  rd.forEach((name, i) => { if (name) setSlot('red',  i, name); });
  bb.forEach((name, i) => { if (name) setBan('blue',  i, name); });
  rb.forEach((name, i) => { if (name) setBan('red',   i, name); });
}

// ── Prediction ─────────────────────────────────────────────────────────────
// ── Draft Strengths ────────────────────────────────────────────────────────
function renderDraftStrengths() {
  const container = $('draft-strengths');
  if (!container) return;

  const picks = {
    blue: ROLES.map((_, i) => getSlot('blue', i)).filter(Boolean),
    red:  ROLES.map((_, i) => getSlot('red',  i)).filter(Boolean),
  };

  if (!picks.blue.length && !picks.red.length) return;

  if (!Object.keys(champTagMap).length) {
    container.innerHTML = '<p class="placeholder-text">Tag data unavailable (DDragon offline).</p>';
    return;
  }

  function countTags(names) {
    const counts = {};
    for (const name of names)
      for (const tag of champTagMap[name] ?? [])
        counts[tag] = (counts[tag] ?? 0) + 1;
    return counts;
  }

  const TAG_ORDER = ['Fighter', 'Tank', 'Mage', 'Assassin', 'Marksman', 'Support'];

  function archetypesFor(counts) {
    const out = [];
    if ((counts.Fighter ?? 0) + (counts.Tank ?? 0) >= 3) out.push('Frontline');
    if ((counts.Assassin ?? 0) >= 2)                      out.push('Dive');
    if ((counts.Mage ?? 0) >= 2)                          out.push('AP Heavy');
    if ((counts.Marksman ?? 0) >= 2)                      out.push('Poke');
    if ((counts.Support ?? 0) >= 2)                       out.push('Utility');
    return out;
  }

  function teamHTML(side, names) {
    const counts     = countTags(names);
    const pills      = TAG_ORDER.filter(t => counts[t])
      .map(t => `<span class="tag-pill">${t}<span class="tag-count">×${counts[t]}</span></span>`)
      .join('');
    const archetypes = archetypesFor(counts)
      .map(a => `<span class="archetype-pill ${side}">${a}</span>`)
      .join('');
    return `
      <div class="tag-team-row ${side}-row">
        <span class="tag-team-label">${side === 'blue' ? 'Blue' : 'Red'}</span>
        <div>
          <div class="tag-pills">${pills || '<span class="placeholder-text" style="font-size:.72rem">—</span>'}</div>
          ${archetypes ? `<div class="archetype-pills">${archetypes}</div>` : ''}
        </div>
      </div>`;
  }

  container.innerHTML = teamHTML('blue', picks.blue) + teamHTML('red', picks.red);
}


function renderLaneMatchups(lane_scores) {
  const container = $('lane-matchups');
  container.innerHTML = '';
  ROLES.forEach((role, i) => {
    const score    = lane_scores?.[i] ?? 0;   // -1 (red) to +1 (blue)
    const bluePct  = ((score + 1) / 2 * 100); // 0–100, 50 = even
    const isBlue   = score >= 0;
    const side     = isBlue ? 'blue' : 'red';
    const abs      = Math.abs(score);
    const label    = abs < 0.20 ? 'Even'
      : abs < 0.55 ? `Slight ${isBlue ? 'Blue' : 'Red'}`
      : abs < 0.75 ? `${isBlue ? 'Blue' : 'Red'}`
      : `Strong ${isBlue ? 'Blue' : 'Red'}`;
    const labelCls = abs < 0.20 ? '' : side;

    const row = document.createElement('div');
    row.className = 'lane-row';
    row.innerHTML = `
      <span class="lane-label">${role}</span>
      <div class="lane-bar-track">
        <div class="lane-bar-blue" style="width:${bluePct.toFixed(1)}%"></div>
        <div class="lane-bar-red"></div>
      </div>
      <span class="lane-verdict ${labelCls}">${label}</span>
    `;
    container.appendChild(row);
  });
}

function renderResult({ blue_win_probability, red_win_probability, confidence, lane_scores }) {
  const bluePct = (blue_win_probability * 100).toFixed(1);
  const redPct  = (red_win_probability  * 100).toFixed(1);
  const winner  = blue_win_probability >= 0.5 ? 'Blue' : 'Red';
  const cls     = winner.toLowerCase();

  $('prediction-result').className = 'prediction-result';
  $('prediction-result').innerHTML = `
    <div class="prob-label-row">
      <span class="prob-label-blue">Blue</span>
      <span class="prob-label-red">Red</span>
    </div>
    <div class="prob-bar-track">
      <div class="prob-bar-fill" style="width:${bluePct}%"></div>
    </div>
    <div class="prob-numbers">
      <span class="prob-big blue">${bluePct}%</span>
      <span class="prob-big red">${redPct}%</span>
    </div>
    <div class="prob-verdict">
      <strong class="prob-label-${cls}">${winner} side</strong> favoured
      <span class="confidence-badge confidence-${confidence}">${confidence}</span>
    </div>
  `;
  if (lane_scores) renderLaneMatchups(lane_scores);
  renderDraftStrengths();
  $('analysis-section').removeAttribute('hidden');
}

async function runPrediction() {
  const draft  = readDraft();
  const filled = draft.filter(Boolean).length;
  if (filled < 10) {
    setStatus(`Fill all 10 champion slots first (${filled}/10).`, 'error');
    return;
  }

  const blue_side = $('chk-blue-side').checked ? 1.0 : 0.0;
  $('btn-analyze').disabled = true;
  $('btn-analyze').textContent = 'Analyzing…';

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ champions: draft, blue_side }),
    });
    const data = await res.json();
    if (!res.ok) {
      if (res.status === 503)
        setStatus('No trained model available. Run "python -m ml.trainer.train" first.', 'error');
      else if (res.status === 400)
        setStatus(`Invalid draft: ${data.detail ?? res.statusText}`, 'error');
      else
        setStatus(`Prediction error: ${data.detail ?? res.statusText}`, 'error');
      return;
    }
    renderResult(data);
    setStatus('');
    history.replaceState(null, '', '?' + new URLSearchParams(new URL(buildShareUrl()).search));
  } catch {
    setStatus('Could not reach the server. Is uvicorn running?', 'error');
  } finally {
    $('btn-analyze').disabled = false;
    $('btn-analyze').textContent = 'Analyze';
  }
}

// ── Live game ──────────────────────────────────────────────────────────────
async function handleLookup() {
  const name   = $('input-name').value.trim();
  const tag    = $('input-tag').value.trim();
  const region = $('input-region').value;
  if (!name || !tag) { setStatus('Enter both Game Name and tag.', 'error'); return; }

  setStatus('Looking up live game…');
  $('btn-lookup').disabled = true;

  try {
    const res  = await fetch(`/api/live-game?name=${encodeURIComponent(name)}&tag=${encodeURIComponent(tag)}&region=${region}`);
    const data = await res.json();
    if (!res.ok) {
      if (res.status === 404)
        setStatus(`Summoner "${name}#${tag}" not found. Check the name and tag.`, 'error');
      else if (res.status === 429)
        setStatus('Rate limited — wait a moment and try again.', 'error', handleLookup);
      else if (res.status === 501)
        setStatus('Live game lookup not yet implemented (see TODO in server.py).', 'error');
      else
        setStatus(`Error: ${data.detail ?? res.statusText}`, 'error');
      return;
    }
    if (!data.in_game) { setStatus(`${name}#${tag} is not currently in a game.`); return; }
    fillDraft(data);
    setStatus(`Live game loaded — ${name}#${tag}`, 'success');
    await runPrediction();
  } catch {
    setStatus('Cannot reach the server. Is uvicorn running?', 'error', handleLookup);
  } finally {
    $('btn-lookup').disabled = false;
  }
}

// ── Demo game ──────────────────────────────────────────────────────────────
async function handleDemo() {
  setStatus('Loading random match…');
  $('btn-demo').disabled = true;

  try {
    const res  = await fetch('/api/demo-game');
    const data = await res.json();
    if (!res.ok) { setStatus(`Error: ${data.detail ?? res.statusText}`, 'error'); return; }
    clearAll();
    fillDraft(data);
    const outcome = data.blue_win !== undefined
      ? ` · ${data.blue_win ? 'Blue' : 'Red'} won`
      : '';
    setStatus(`${data.match_id}${outcome}`, 'success');
    await runPrediction();
  } catch {
    setStatus('Could not reach the server. Is uvicorn running?', 'error');
  } finally {
    $('btn-demo').disabled = false;
  }
}

// ── Global keyboard / click handlers ─────────────────────────────────────
function handleGlobalKeydown(e) {
  if (e.key === 'Escape') closePicker();
}

function handleModalOverlayClick(e) {
  if (e.target === $('picker-modal')) closePicker();
}

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  // Build DOM
  buildSlots('blue-slots', 'blue');
  buildSlots('red-slots',  'red');
  buildBans('blue-bans', 'blue');
  buildBans('red-bans',  'red');

  // Event listeners
  $('btn-lookup').addEventListener('click', handleLookup);
  $('btn-retry').addEventListener('click', () => { if (_retryFn) _retryFn(); });
  $('btn-demo').addEventListener('click', handleDemo);
  $('btn-analyze').addEventListener('click', runPrediction);
  $('btn-share').addEventListener('click', handleShare);
  $('btn-clear').addEventListener('click', clearAll);
  $('btn-picker-close').addEventListener('click', closePicker);
  $('picker-search').addEventListener('input', e => refreshPickerUsed(e.target.value));
  $('picker-modal').addEventListener('click', handleModalOverlayClick);
  document.addEventListener('keydown', handleGlobalKeydown);

  [$('input-name'), $('input-tag')].forEach(el =>
    el.addEventListener('keydown', e => { if (e.key === 'Enter') handleLookup(); })
  );

  // Data loading (sequential: version → champions → tags)
  await fetchDDragonVersion();
  await loadChampions();
  await loadChampionTags();

  // Restore from URL if present (after champions are loaded so icons render)
  loadFromUrl();
}

document.addEventListener('DOMContentLoaded', init);
