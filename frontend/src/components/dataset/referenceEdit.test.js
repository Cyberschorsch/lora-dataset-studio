import test from 'node:test';
import assert from 'node:assert/strict';
import {
  EDIT_ENGINES, defaultEditEngine, editBlockedReason, editEngineChoiceMessage,
  batchLiveNote, editPhase, editEngineOptions, editCostNote, editKeepNote,
  editRefNote, acceptsExtraEditRefs, editRefSupport,
} from './referenceEdit.js';
import {
  STORAGE_ENGINES, STORAGE_PRIMARY, ENGINES, API_ENGINES, LOCAL_ENGINES, ENGINE_LABELS,
} from './engineSelection.js';

function fakeStorage(seed = {}) {
  const data = { ...seed };
  return { getItem(k) { return k in data ? data[k] : null; }, setItem(k, v) { data[k] = String(v); } };
}

/* This file must NOT pin a fixed length: the first assertion here said "exactly
   these two", which is why OpenRouter shipping as a generation engine left the ✦
   Edit modal silently one engine short. Its replacement then said "Klein is OUT",
   which outlived its own reason — the exclusion was about the edit being a
   BLOCKING provider call, and a local edit now waits on the ComfyUI queue like
   every other local render. Both local engines were then missing from the app's
   only free edit lane, for a rule nobody re-read. */
test('every engine can edit the reference, including the local ones', () => {
  assert.ok(EDIT_ENGINES.includes('krea'));
  assert.ok(EDIT_ENGINES.includes('klein'));
  for (const e of API_ENGINES) assert.ok(EDIT_ENGINES.includes(e), e);
});

test('EDIT_ENGINES is derived from ENGINES, so it cannot drift from it', () => {
  assert.deepEqual(EDIT_ENGINES, [...ENGINES]);
  assert.notEqual(EDIT_ENGINES, ENGINES);   // a copy: mutating one can't move the other
});

test('the free local engines are listed FIRST — cheapest option first', () => {
  // Not a ranking: this gesture is billed per press, and the list is read
  // top-down. Burying the free option under three paid ones is a price tag.
  assert.deepEqual(EDIT_ENGINES.slice(0, LOCAL_ENGINES.length), [...LOCAL_ENGINES]);
});

test('OpenRouter can edit the reference, like the other API engines', () => {
  // Regression pin for the gap this wave closed: the engine existed for
  // generation while the edit path still refused it.
  assert.ok(EDIT_ENGINES.includes('openrouter'));
  assert.equal(editBlockedReason('add glasses', 'openrouter'), null);
});

test('defaultEditEngine mirrors the primary generation engine when it can edit', () => {
  assert.equal(defaultEditEngine(fakeStorage({ [STORAGE_ENGINES]: JSON.stringify(['nanobanana']) })),
    'nanobanana');
  assert.equal(defaultEditEngine(fakeStorage({ [STORAGE_PRIMARY]: 'chatgpt' })), 'chatgpt');
});

test('defaultEditEngine opens on Klein when Klein is the primary generation engine', () => {
  // It used to fall back to ChatGPT here: Klein could not edit. Sending a
  // Klein-only, ComfyUI-only profile to a paid API by default is exactly the
  // behaviour this wave removes.
  assert.equal(defaultEditEngine(fakeStorage({ [STORAGE_ENGINES]: JSON.stringify(['klein']) })),
    'klein');
});

test('defaultEditEngine skips a local primary this install cannot run', () => {
  // No ComfyUI: opening on a disabled button is a bad first impression, so the
  // first API engine that IS usable takes over.
  const storage = fakeStorage({ [STORAGE_ENGINES]: JSON.stringify(['krea']) });
  assert.equal(defaultEditEngine(storage, (e) => !LOCAL_ENGINES.includes(e)), 'nanobanana');
});

test('defaultEditEngine with no stored preference uses the historic default (Nano Banana)', () => {
  // readEngines falls back to DEFAULT_ENGINE (nanobanana), which CAN edit.
  assert.equal(defaultEditEngine(fakeStorage()), 'nanobanana');
});

test('editBlockedReason blocks an empty prompt and an un-editable engine', () => {
  assert.equal(editBlockedReason('add glasses', 'chatgpt'), null);
  assert.equal(editBlockedReason('add glasses', 'krea'), null);
  assert.match(editBlockedReason('', 'chatgpt'), /describe/i);
  assert.match(editBlockedReason('   ', 'nanobanana'), /describe/i);
  assert.equal(editBlockedReason('x', 'midjourney'), editEngineChoiceMessage());
});

test('editBlockedReason surfaces WHY a local engine is unavailable, before the click', () => {
  // The whole point: not a greyed button, the one action that fixes it. The
  // engine reason wins over "describe the edit first" — typing a prompt would
  // not make Krea appear.
  const reason = '⚠ Install the comfyui-krea2edit node pack in ComfyUI, then restart it';
  assert.equal(editBlockedReason('add glasses', 'krea', reason), reason);
  assert.equal(editBlockedReason('', 'krea', reason), reason);
});

test('the refusal names the engines that DO edit, derived from the list', () => {
  // Pinned by construction, not by a fixed sentence: a hardcoded sentence is the
  // hardcoded list again, and it is what made the old message name two engines
  // after a third became editable.
  const msg = editEngineChoiceMessage();
  for (const e of EDIT_ENGINES) assert.ok(msg.includes(ENGINE_LABELS[e]), e);
  assert.equal(msg, 'Pick Klein, Krea 2 Edit, Z-Image Turbo, Nano Banana Pro, ChatGPT or OpenRouter');
});

/* ── The three things a local engine does differently ─────────────────────── */

test('an install with no ComfyUI is offered no local engine at all', () => {
  // Not a gap to fix from this modal — a product the user hasn't got. Two
  // permanently dead buttons would be worse than three live ones.
  const opts = editEngineOptions({ comfyuiConfigured: false });
  assert.deepEqual(opts.map((o) => o.engine), [...API_ENGINES]);
  assert.ok(opts.every((o) => o.usable));
});

test('a configured ComfyUI keeps the local engines VISIBLE and says what to do', () => {
  const opts = editEngineOptions({
    comfyuiConfigured: true,
    available: { klein: true, krea: false },
    reasonFor: (e) => (e === 'krea' ? '⚠ Krea base model missing — Setup can download it' : null),
  });
  const krea = opts.find((o) => o.engine === 'krea');
  assert.ok(krea, 'a fixable gap must not hide the engine');
  assert.equal(krea.usable, false);
  assert.match(krea.blocked, /Setup can download/);
  assert.equal(opts.find((o) => o.engine === 'klein').usable, true);
});

test('an unavailable local engine is never silently offered as usable', () => {
  // No diagnostic available (older server, unknown gap): still says something,
  // still not usable. Silence is the failure mode being removed.
  const opts = editEngineOptions({ comfyuiConfigured: true, available: {} });
  for (const e of LOCAL_ENGINES) {
    const o = opts.find((x) => x.engine === e);
    assert.equal(o.usable, false, e);
    assert.ok(o.blocked && o.blocked.length, e);
  }
});

test('API engines are never blocked by capabilities here — the paid lane is untouched', () => {
  const opts = editEngineOptions({ comfyuiConfigured: true, available: {},
    reasonFor: () => '⚠ nope' });
  for (const e of API_ENGINES) {
    assert.equal(opts.find((o) => o.engine === e).usable, true, e);
  }
});

test('the cost line tells the truth per engine — free is not "a paid API call"', () => {
  assert.match(editCostNote('chatgpt'), /paid API call/);
  for (const e of LOCAL_ENGINES) {
    assert.doesNotMatch(editCostNote(e), /paid/i, e);
    assert.match(editCostNote(e), /your own ComfyUI/, e);
  }
});

test('the Keep line does not claim a refund that never applied', () => {
  assert.match(editKeepNote('nanobanana'), /doesn’t refund/);
  assert.match(editKeepNote('krea'), /costs you nothing/);
  // Both still say the Keep itself is irreversible — that part is engine-agnostic.
  for (const e of ['krea', 'nanobanana']) assert.match(editKeepNote(e), /can’t be undone/);
});

test('an engine that takes fewer references SAYS so at pick time', () => {
  assert.equal(editRefSupport('chatgpt'), 'all');
  assert.equal(editRefNote('chatgpt'), null);          // nothing to warn about
  assert.match(editRefNote('krea'), /main reference only/);
  assert.match(editRefNote('klein', { datasetExtraCount: 2 }), /2 extra reference photos/);
  assert.match(editRefNote('klein'), /not sent/);
});

test('the transient reference picker is hidden for engines that cannot take it', () => {
  // Hidden, not ignored: an input whose files are silently dropped returns an
  // edit that used half of what the user handed it.
  assert.equal(acceptsExtraEditRefs('chatgpt'), true);
  assert.equal(acceptsExtraEditRefs('klein'), false);
  assert.equal(acceptsExtraEditRefs('krea'), false);
});

test('batchLiveNote informs only while a generate batch runs, never blocks', () => {
  assert.equal(batchLiveNote(null), null);
  assert.equal(batchLiveNote({ kind: 'caption' }), null);
  assert.match(batchLiveNote({ kind: 'generate' }), /future batches/i);
});

test('editPhase derives the modal phase from the server reference_edit object', () => {
  assert.equal(editPhase(null), 'idle');
  assert.equal(editPhase(undefined), 'idle');
  assert.equal(editPhase({ status: 'running' }), 'running');
  assert.equal(editPhase({ status: 'ready', candidate_filename: 'x.webp' }), 'ready');
  assert.equal(editPhase({ status: 'failed', error: 'boom' }), 'failed');
  assert.equal(editPhase({ status: 'weird' }), 'idle');   // unknown → idle (form)
});
