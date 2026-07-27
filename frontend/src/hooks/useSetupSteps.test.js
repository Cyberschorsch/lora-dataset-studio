import test from 'node:test';
import assert from 'node:assert/strict';
import {
  deriveSetupSteps, deriveCapabilitySummary, kleinMissingLabels, KLEIN_ASSET_LABELS,
  comfyuiDirVerdict, COMFYUI_SKIP_LOST, COMFYUI_SKIP_KEPT,
  aitoolkitVerdict, AITOOLKIT_INSTALL_STEPS,
} from './useSetupSteps.js';
// installAllPlan / installCatalog are imported further down, next to their own
// sections; kreaInstallPlan has no other importer here.
import { kreaInstallPlan } from './useSetupSteps.js';
import { localEngineUnavailableReason } from '../utils/localEngineReason.js';
import fs from 'node:fs';

const comfyStep = (comfyui) => deriveSetupSteps({ comfyui }).find((s) => s.id === 'comfyui');
// The same step, driven by a FULL capabilities payload — engines included. That is
// the shape the app actually receives, and the one where Setup reads the backend's
// verdict instead of re-deriving its own.
const comfyStepFull = (caps) => deriveSetupSteps(caps).find((s) => s.id === 'comfyui');

test('Klein readiness needs the full trio, not just the UNET', () => {
  // UNET landed, but the backend still lists the text-encoder + VAE as missing:
  // the step must NOT go "ready", and hasKlein must be false so "Nothing to do"
  // and the disappearing-download-buttons bugs cannot fire.
  const step = comfyStep({
    reachable: true,
    models: { klein: ['flux-2-klein-9b-fp8.safetensors'] },
    klein_missing: ['klein_text_encoder', 'klein_vae', 'klein_lora'],
  });
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'partial'); // reachable but incomplete
  assert.deepEqual(step.kleinMissing, ['klein_text_encoder', 'klein_vae', 'klein_lora']);
});

test('all three weights present -> ready even with the recommended LoRA still missing', () => {
  const step = comfyStep({
    reachable: true,
    models: { klein: ['flux-2-klein-9b-fp8.safetensors'] },
    klein_missing: ['klein_lora'], // recommended only, does not gate the engine
  });
  assert.equal(step.hasKlein, true);
  assert.equal(step.status, 'ready');
});

test('reachable with nothing missing -> ready', () => {
  const step = comfyStep({ reachable: true, klein_missing: [] });
  assert.equal(step.hasKlein, true);
  assert.equal(step.status, 'ready');
});

test('a present-but-INVALID required asset keeps the step from going green', () => {
  // The #help incident: the UNET file is on disk (so klein_missing is empty) but
  // it is really an HTML licence-gate page. The step must NOT go "ready" — otherwise
  // Setup stays green and the user hits the cryptic UNETLoader crash at generate.
  const step = comfyStep({
    reachable: true,
    klein_missing: [],
    klein_invalid: [{
      asset: 'klein_model', filename: 'flux-2-klein-9b-fp8.safetensors', blocking: true,
      verdict: 'html_or_text', reason: 'flux-2-klein-9b-fp8.safetensors is not a real model (looks like an HTML page …)',
    }],
  });
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'partial'); // reachable but a required weight is broken
  assert.equal(step.kleinInvalid.length, 1);
});

test('a widget value this ComfyUI does not offer keeps the step from going green', () => {
  // Reported by IndependentProcess0 (Reddit). Every weight is on disk and every
  // node class exists, so both existing checks pass — and the first generation
  // still dies, because the graph pins a scheduler this ComfyUI never had. Since
  // nothing is substituted (a scheduler changes the render), Setup is where the
  // user has to find out, not the ComfyUI console after a failed batch.
  const step = comfyStep({
    reachable: true,
    klein_missing: [],
    klein_unsupported_enums: [{
      node_id: '77', class_type: 'KSampler', input: 'scheduler', value: 'beta57',
      pack: 'RES4LYF', url: 'https://github.com/ClownsharkBatwing/RES4LYF',
    }],
  });
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'partial');
  assert.equal(step.unsupportedEnums.length, 1);
});

test('an install that offers every pinned value is unaffected', () => {
  // No regression for people whose ComfyUI already has the value: same render,
  // same green step. The server sends an empty list both when it verified and
  // found nothing AND when it could not verify at all (fail-open).
  const ready = comfyStep({ reachable: true, klein_missing: [], klein_unsupported_enums: [] });
  assert.equal(ready.hasKlein, true);
  assert.equal(ready.status, 'ready');
  assert.deepEqual(ready.unsupportedEnums, []);
  // An older backend that doesn't publish the field at all must not gate either.
  const legacy = comfyStep({ reachable: true, klein_missing: [] });
  assert.equal(legacy.hasKlein, true);
  assert.deepEqual(legacy.unsupportedEnums, []);
});

test('an advisory too_small invalid does NOT gate readiness', () => {
  const step = comfyStep({
    reachable: true,
    klein_missing: [],
    klein_invalid: [{ asset: 'klein_model', filename: 'k.safetensors', blocking: false, verdict: 'too_small', reason: 'k.safetensors is only 10 B …' }],
  });
  assert.equal(step.hasKlein, true);
  assert.equal(step.status, 'ready');
});

test('unreachable ComfyUI is "available" regardless of assets on disk', () => {
  const step = comfyStep({ reachable: false, klein_missing: [] });
  assert.equal(step.status, 'available');
  assert.equal(step.reachable, false);
});

test('legacy payload without klein_missing falls back to the UNET scan', () => {
  const withUnet = comfyStep({ reachable: true, models: { klein: ['a.safetensors'] } });
  assert.equal(withUnet.hasKlein, true);
  assert.deepEqual(withUnet.kleinMissing, []);

  const noUnet = comfyStep({ reachable: true, models: { klein: [] } });
  assert.equal(noUnet.hasKlein, false);
  assert.deepEqual(noUnet.kleinMissing, ['klein_model']);
});

/* --- "Ready" must mean the same thing on every screen ------------------------

   Reported by zigzag4794 (Discord): Setup showed ✓ Installed for every required
   Klein model, ComfyUI tested OK, the 9.5 GB file was in models/unet/klein/ — and
   the Generate page kept Klein greyed out with "⚠ Klein model missing — download
   it in the Setup step". Both screens were sincere. `/api/capabilities` had the
   answer all along: both model files under `klein_invalid`, verdict
   `truncated_or_garbage`. The integrity validator existed and worked; the Setup
   screen simply never asked it, and judged each file on presence alone.

   These tests pin the contract, not the wording: ONE verdict (the backend's),
   consumed by both, and a sentence that names the real gap. */

const CORRUPT_KLEIN = [
  { asset: 'klein_model', filename: 'flux-2-klein-9b-kv-fp8.safetensors',
    verdict: 'truncated_or_garbage', blocking: true,
    reason: 'flux-2-klein-9b-kv-fp8.safetensors is shorter than its header declares' },
];

const zigzagCaps = () => ({
  engines: { klein: false, krea: false },   // what the BACKEND concluded
  comfyui: {
    reachable: true, dir_valid: true,
    klein_missing: ['klein_lora'],          // recommended only — nothing required is absent
    klein_unsupported_enums: [],
    klein_invalid: CORRUPT_KLEIN,
  },
});

test('a corrupted weight keeps Setup from claiming Klein is ready', () => {
  const step = comfyStepFull(zigzagCaps());
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'partial');
  assert.equal(step.kleinBroken.length, 1);
  assert.equal(step.kleinFilesReady, false);
});

test('Setup and the generation panel give the SAME reason, word for word', () => {
  // Not "both say something is wrong" — literally the same sentence, because both
  // read utils/localEngineReason.js. Two wordings for one gap is how the user ends
  // up believing they are two problems.
  const caps = zigzagCaps();
  assert.equal(comfyStepFull(caps).kleinReason,
    localEngineUnavailableReason('klein', caps));
});

test('Setup never says "missing" about a file that is on the disk', () => {
  const reason = comfyStepFull(zigzagCaps()).kleinReason;
  assert.doesNotMatch(reason, /missing/i);
  assert.match(reason, /flux-2-klein-9b-kv-fp8\.safetensors/);
  assert.match(reason, /Delete it and download it again/);
});

test('the install menu stops badging an unreadable file "✓ Installed"', () => {
  // THE screen zigzag was reading. `present: false` is not cosmetic — it is what
  // puts the asset back into the install plans so the button has work to do.
  const rows = installCatalog(zigzagCaps());
  const unet = rows.find((r) => r.action === 'klein_model');
  assert.equal(unet.present, false);
  assert.equal(unet.state, 'broken');
  assert.match(unet.stateLabel, /unreadable/);
  assert.match(unet.brokenReason, /flux-2-klein-9b-kv-fp8\.safetensors/);
  // A file that is genuinely fine is untouched.
  assert.equal(rows.find((r) => r.action === 'klein_vae').present, true);
});

test('a corrupted weight is re-queued by the one-click install, not skipped', () => {
  // Without this the plan is empty, the card says "everything is in place", and
  // the engine stays dark with no way forward from inside the app.
  assert.ok(installAllPlan(zigzagCaps()).includes('klein_model'));
  const kreaBroken = {
    engines: { krea: false },
    comfyui: { reachable: true, dir_valid: true, krea_missing: [],
      krea_nodes_installed: true, krea_nodes_missing: [],
      krea_invalid: [{ asset: 'krea_model', filename: 'k.safetensors',
        verdict: 'html_or_text', blocking: true }] },
  };
  assert.deepEqual(kreaInstallPlan(kreaBroken), ['krea_model']);
});

test('an unreachable ComfyUI produces "not checked", never a ✓', () => {
  // The unsupported-value and node probes fail OPEN by design — an unreachable
  // ComfyUI reports no gap rather than inventing one. So "no warnings" here is the
  // absence of an answer, not a clean bill of health, and the step must not read
  // as ready on the strength of it.
  const step = comfyStepFull({
    engines: { klein: false },
    comfyui: { reachable: false, dir_valid: true, klein_missing: [],
      klein_unsupported_enums: [], klein_invalid: [] },
  });
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'available');
  assert.equal(step.kleinVerified, false);
  // …and the files themselves ARE fine, which is a different sentence.
  assert.equal(step.kleinFilesReady, true);
});

test('Setup does not second-guess the backend when it says the engine is up', () => {
  const step = comfyStepFull({
    engines: { klein: true },
    comfyui: { reachable: true, dir_valid: true, klein_missing: ['klein_lora'] },
  });
  assert.equal(step.hasKlein, true);
  assert.equal(step.status, 'ready');
  assert.equal(step.kleinReason, null);
  assert.equal(step.kleinVerified, true);
});

test('comfy step lists missing Z-Image weights for download', () => {
  const step = comfyStep({
    reachable: true, klein_missing: [],
    zimage_missing: ['zimage_model', 'zimage_vae'],
  });
  assert.deepEqual(step.zimageMissing, ['zimage_model', 'zimage_vae']);
});

test('Z-Image missing weights do not gate the ComfyUI step ready/green', () => {
  // Z-Image is optional like the recommended Klein LoRA: listed for download,
  // never blocks the step from going green once Klein itself is complete.
  const step = comfyStep({
    reachable: true, klein_missing: [],
    zimage_missing: ['zimage_model', 'zimage_text_encoder', 'zimage_vae'],
  });
  assert.equal(step.hasKlein, true);
  assert.equal(step.status, 'ready');
});

// --- Conscious "continue without ComfyUI" skip (Setup Volet 2) --------------

test('skipped ComfyUI (flag set, unreachable) renders a neutral "skipped" status', () => {
  const step = comfyStep({ skipped: true, reachable: false });
  assert.equal(step.status, 'skipped');
  assert.equal(step.skipped, true);
});

test('a running ComfyUI is never shown as skipped even if the flag lingers', () => {
  // The backend already annuls the skip once a dir is set; belt-and-suspenders on the
  // client: a reachable server always shows its real status, never "skipped".
  const step = comfyStep({ skipped: true, reachable: true, klein_missing: [] });
  assert.equal(step.skipped, false);
  assert.equal(step.status, 'ready');
});

test('no skip flag -> normal available status, skipped false', () => {
  const step = comfyStep({ reachable: false });
  assert.equal(step.skipped, false);
  assert.equal(step.status, 'available');
});

test('comfyuiDirVerdict maps each backend status to an actionable message', () => {
  assert.equal(comfyuiDirVerdict({ status: 'valid', resolved: 'C:/Comfy' }).tone, 'ok');
  assert.match(comfyuiDirVerdict({ status: 'valid', resolved: 'C:/Comfy' }).message, /C:\/Comfy/);

  const nested = comfyuiDirVerdict({ status: 'nested', suggestion: 'C:/x/ComfyUI' });
  assert.equal(nested.tone, 'warn');
  assert.equal(nested.suggestion, 'C:/x/ComfyUI');   // drives the adopt button
  assert.match(nested.message, /launcher\/parent folder/);

  assert.match(comfyuiDirVerdict({ status: 'missing' }).message, /doesn't exist/);
  assert.match(comfyuiDirVerdict({ status: 'empty_dir' }).message, /empty/);
  assert.match(comfyuiDirVerdict({ status: 'not_comfyui' }).message, /isn't a ComfyUI install/);
  for (const s of ['missing', 'empty_dir', 'not_comfyui']) {
    assert.equal(comfyuiDirVerdict({ status: s }).tone, 'warn');
    assert.equal(comfyuiDirVerdict({ status: s }).suggestion, '');
  }
  // Blank / in-flight / unknown -> muted, nothing to render.
  assert.deepEqual(comfyuiDirVerdict({ status: 'empty' }),
    { tone: 'muted', suggestion: '', note: '', message: '' });
  assert.equal(comfyuiDirVerdict(null).message, '');
});

/* The wizard used to certify half the contract: it tested the URL and accepted the
   folder, but never checked that the app can actually PUT a file in ComfyUI's input
   folder — which is how every local engine hands over its source image. With ComfyUI
   in a second container that folder isn't shared, setup went green and the first
   generation died on a detail-free 500 (reported on Discord by nofaceman). */
test('a valid ComfyUI folder the app cannot write into warns without blocking', () => {
  const v = comfyuiDirVerdict({
    status: 'valid', resolved: 'C:/Comfy',
    input_check: { path: 'C:/Comfy/input', ok: false,
      problem: "ComfyUI's input folder is not writable from LoRA Dataset Studio: "
        + 'C:/Comfy/input. If ComfyUI runs in another container, in WSL or on another '
        + 'machine, this folder must be a shared volume visible to LoRA Dataset Studio '
        + 'at that exact path.' },
  });
  assert.equal(v.tone, 'ok');                 // still a valid install: NOT a blocker
  assert.match(v.message, /ComfyUI found/);
  assert.match(v.note, /not writable/);       // ...and the second half is said
  assert.match(v.note, /shared volume/);
});

test('a working install, or a backend that says nothing, adds no note', () => {
  assert.equal(comfyuiDirVerdict({ status: 'valid', resolved: 'C:/Comfy',
    input_check: { path: 'C:/Comfy/input', ok: true, problem: '' } }).note, '');
  // ok=null (nothing probed) and a missing field must never read as a failure
  assert.equal(comfyuiDirVerdict({ status: 'valid', resolved: 'C:/Comfy',
    input_check: { ok: null, problem: '' } }).note, '');
  assert.equal(comfyuiDirVerdict({ status: 'valid', resolved: 'C:/Comfy' }).note, '');
});

test('the wizard renders the input-folder note', () => {
  const jsx = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');
  assert.match(jsx, /v\.note/);
});

test('skip panel lists what turns off and what stays on', () => {
  assert.ok(COMFYUI_SKIP_LOST.length >= 4 && COMFYUI_SKIP_KEPT.length >= 4);
  const lost = COMFYUI_SKIP_LOST.join(' | ');
  assert.match(lost, /Klein/);
  assert.match(lost, /Test Studio/);
  const kept = COMFYUI_SKIP_KEPT.join(' | ');
  assert.match(kept, /Scraping/);
  assert.match(kept, /ai-toolkit/);
  assert.match(kept, /Hugging Face/);
});

test('kleinMissingLabels maps required assets to words in a stable order', () => {
  // Order is canonical (model, text encoder, VAE), independent of input order.
  assert.deepEqual(
    kleinMissingLabels(['klein_vae', 'klein_lora', 'klein_model']),
    ['model', 'VAE'],
  );
  assert.deepEqual(kleinMissingLabels(['klein_text_encoder']), ['text encoder']);
  assert.deepEqual(kleinMissingLabels([]), []);
  assert.deepEqual(kleinMissingLabels(undefined), []);
  // The recommended LoRA is never surfaced as a required gap.
  assert.deepEqual(kleinMissingLabels(['klein_lora']), []);
  assert.equal(KLEIN_ASSET_LABELS.klein_text_encoder, 'text encoder');
});

// --- installAllPlan (mirror of the backend orchestrator) --------------------
import { installAllPlan, INSTALL_ALL_ORDER } from './useSetupSteps.js';

// A fully-installed snapshot; each test flips just the pieces it needs MISSING.
const fullCaps = () => ({
  python: { ml_supported: true },
  face_scoring: true, masks: true, watermark_inpaint: true,
  ollama: { reachable: true, vision_model_ready: true, vision_model: 'qwen3-vl:8b' },
  // reachable matters for the Krea node pack: an unreachable ComfyUI's node probe
  // fails open, so "nothing missing" from a stopped ComfyUI must not read as
  // "the pack is installed".
  // zimage_base_present is PRESENCE, not absence: the optional Base checkpoint is a
  // second checkpoint, never a missing dependency, so it is reported the other way up.
  comfyui: { dir_valid: true, reachable: true, klein_missing: [], krea_missing: [],
    zimage_base_present: true },
});

test('installAllPlan is empty when everything installable is present', () => {
  assert.deepEqual(installAllPlan(fullCaps()), []);
});

test('installAllPlan folds null/empty caps to the always-runnable ML extras', () => {
  const mlOnly = ['face_scoring', 'masks', 'watermark_inpaint'];
  assert.deepEqual(installAllPlan(null), mlOnly);
  assert.deepEqual(installAllPlan({}), mlOnly);
});

test('installAllPlan skips face/masks on an unsupported Python but keeps watermark', () => {
  const caps = { ...fullCaps(), python: { ml_supported: false },
    face_scoring: false, masks: false, watermark_inpaint: false };
  assert.deepEqual(installAllPlan(caps), ['watermark_inpaint']);
});

test('installAllPlan queues the vision model only when Ollama is up and named', () => {
  const up = { ...fullCaps(),
    ollama: { reachable: true, vision_model_ready: false, vision_model: 'qwen3-vl:8b' } };
  assert.ok(installAllPlan(up).includes('ollama_model'));
  const noName = { ...fullCaps(),
    ollama: { reachable: true, vision_model_ready: false, vision_model: '' } };
  assert.ok(!installAllPlan(noName).includes('ollama_model'));
  const down = { ...fullCaps(),
    ollama: { reachable: false, vision_model_ready: false, vision_model: 'qwen3-vl:8b' } };
  assert.ok(!installAllPlan(down).includes('ollama_model'));
});

test('installAllPlan takes Klein weights only into a validated ComfyUI, in order', () => {
  const valid = { ...fullCaps(),
    comfyui: { dir_valid: true, klein_missing: ['klein_lora', 'klein_model', 'klein_vae'] } };
  assert.deepEqual(installAllPlan(valid), ['klein_model', 'klein_vae', 'klein_lora']);
  const invalid = { ...fullCaps(),
    comfyui: { dir_valid: false, klein_missing: ['klein_model'] } };
  assert.deepEqual(installAllPlan(invalid), []);
});

test('installAllPlan full order groups ML -> vision model -> Klein', () => {
  const caps = {
    python: { ml_supported: true },
    face_scoring: false, masks: false, watermark_inpaint: false,
    ollama: { reachable: true, vision_model_ready: false, vision_model: 'm' },
    comfyui: { dir_valid: true,
      klein_missing: ['klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora'],
      zimage_missing: ['zimage_model', 'zimage_text_encoder', 'zimage_vae'] },
  };
  // Z-Image (like Krea) is a secondary engine: it is deliberately NOT part of
  // "Install everything", so its weights never appear even when all are missing.
  const plan = installAllPlan(caps);
  assert.deepEqual(plan, INSTALL_ALL_ORDER);
  assert.ok(!plan.some((a) => a.startsWith('zimage_')));
});

// --- installCatalog (the full one-by-one install/reinstall menu) -------------
import { installCatalog } from './useSetupSteps.js';

const byAction = (cat) => Object.fromEntries(cat.map((c) => [c.action, c]));

test('installCatalog lists every app-installable component, present + available', () => {
  const cat = byAction(installCatalog(fullCaps()));
  // Every component the app can install itself (never ComfyUI/Ollama/API keys).
  // The Krea 2 Edit and Z-Image rows land here too — each engine's install is its
  // own path, this menu is the per-piece repair each of their weights also deserves.
  assert.deepEqual(
    installCatalog(fullCaps()).map((c) => c.action),
    ['face_scoring', 'masks', 'watermark_inpaint', 'ollama_model',
      'klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora',
      'krea_nodes', 'krea_model', 'krea_text_encoder', 'krea_vae',
      'krea_identity_lora',
      'zimage_model', 'zimage_text_encoder', 'zimage_vae', 'zimage_base_model'],
  );
  // Everything installed in fullCaps -> every tile present, and available to REINSTALL.
  for (const c of Object.values(cat)) {
    assert.equal(c.present, true, `${c.action} present`);
    assert.equal(c.available, true, `${c.action} available to reinstall`);
  }
});

test('installCatalog stays fully available for reinstall when all is green', () => {
  // The menu must never collapse once installed — each item can always be repaired.
  const cat = installCatalog(fullCaps());
  assert.ok(cat.length === 17 && cat.every((c) => c.available));
});

test('installCatalog marks missing ML extras not-present but still available', () => {
  const cat = byAction(installCatalog({ ...fullCaps(),
    face_scoring: false, masks: false, watermark_inpaint: false }));
  for (const a of ['face_scoring', 'masks', 'watermark_inpaint']) {
    assert.equal(cat[a].present, false);
    assert.equal(cat[a].available, true);   // installable now (supported Python)
  }
});

test('installCatalog blocks fresh ML installs on an unsupported Python, with a hint', () => {
  const cat = byAction(installCatalog({ ...fullCaps(),
    python: { ml_supported: false, ml_range: '3.10–3.12' },
    face_scoring: false, masks: false }));
  // Can't install into the app's out-of-range Python -> unavailable + an actionable hint.
  assert.equal(cat.face_scoring.available, false);
  assert.match(cat.face_scoring.hint, /3\.10–3\.12/);
  // watermark auto-provisions its own venv, so it stays available regardless.
  assert.equal(cat.watermark_inpaint.available, true);
});

test('installCatalog still lets you REPAIR a present ML extra on an unsupported Python', () => {
  // A face-scoring already installed (into a dedicated env) can be reinstalled/repaired
  // even when the app's own Python is out of the wheel range.
  const cat = byAction(installCatalog({ ...fullCaps(),
    python: { ml_supported: false, ml_range: '3.10–3.12' }, face_scoring: true }));
  assert.equal(cat.face_scoring.present, true);
  assert.equal(cat.face_scoring.available, true);
});

test('installCatalog gates the vision model on a reachable, named Ollama', () => {
  const down = byAction(installCatalog({ ...fullCaps(),
    ollama: { reachable: false, vision_model_ready: false, vision_model: 'm' } }));
  assert.equal(down.ollama_model.available, false);
  assert.match(down.ollama_model.hint, /Start Ollama/);
  const noName = byAction(installCatalog({ ...fullCaps(),
    ollama: { reachable: true, vision_model_ready: false, vision_model: '' } }));
  assert.equal(noName.ollama_model.available, false);
  assert.match(noName.ollama_model.hint, /model name/);
});

test('installCatalog gates Klein weights on a validated ComfyUI', () => {
  const invalid = byAction(installCatalog({ ...fullCaps(),
    comfyui: { dir_valid: false, klein_missing: ['klein_model'] } }));
  for (const a of ['klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora']) {
    assert.equal(invalid[a].available, false);
    assert.match(invalid[a].hint, /valid ComfyUI folder/);
  }
  const valid = byAction(installCatalog({ ...fullCaps(),
    comfyui: { dir_valid: true, klein_missing: ['klein_vae'] } }));
  assert.equal(valid.klein_vae.present, false);       // still missing
  assert.equal(valid.klein_vae.available, true);      // installable into the valid tree
  assert.equal(valid.klein_model.present, true);      // not in klein_missing -> installed
});


test('capability summary: configured-but-not-running ComfyUI reads as OK with a launch note', () => {
  const row = (caps, label) => deriveCapabilitySummary(caps).find((s) => s.label === label);
  // dir valid, API down -> Klein/Studio are pending with the launch note, not a bare miss
  const off = { comfyui: { dir_valid: true, reachable: false }, engines: {} };
  assert.equal(row(off, 'Klein (local)').pending, true);
  assert.match(row(off, 'Klein (local)').note, /launch ComfyUI/);
  assert.equal(row(off, 'Test Studio').pending, true);
  // no ComfyUI configured at all -> honest plain miss, no note
  const none = { comfyui: { dir_valid: false, reachable: false }, engines: {} };
  assert.equal(row(none, 'Klein (local)').pending, undefined);
  // running -> plain ok, no note
  const on = { comfyui: { dir_valid: true, reachable: true }, engines: { klein: true }, studio_visible: true };
  assert.equal(row(on, 'Klein (local)').ok, true);
  assert.equal(row(on, 'Klein (local)').note, undefined);
});


// ── ai-toolkit setup copy ─────────────────────────────────────────────────────
// Reported on Reddit by Psyko_2000: an ai-toolkit installed by a community
// easy-install script ships a `python_embeded` folder and no venv, and the wizard
// answered "set up its Python venv per the README" — a cause the app never
// verified, and a remedy that install can never follow. He concluded the app
// REQUIRED a venv and asked in public, instead of filling the `aitoolkit.python`
// setting that already existed. These tests lock the CONTENT, since the content
// is what was false: showing the block proves nothing.
const trainStep = (aitoolkit) => deriveSetupSteps({ aitoolkit }).find((s) => s.id === 'training');
const DIR = '/opt/ai-toolkit';

test('a folder with no interpreter reports the FINDING, never "a venv is missing"', () => {
  const v = aitoolkitVerdict(trainStep({ valid: false, dir_valid: true }), DIR);
  assert.equal(v.kind, 'no_interpreter');
  assert.match(v.headline, /no Python interpreter was found/i);
  assert.match(v.headline, new RegExp(DIR));
  // The headline must not diagnose a missing venv — that is the claim that was wrong.
  assert.doesNotMatch(v.headline, /venv/i);
});

test('both routes are offered, and the existing-interpreter one is not a footnote', () => {
  const v = aitoolkitVerdict(trainStep({ valid: false, dir_valid: true }), DIR);
  // Route A: make one.
  assert.match(v.body, /create a venv/i);
  // Route B: use the one you already have — named, in the same sentence, with the
  // real-world shapes spelled out (Psyko_2000's is the portable/embedded one).
  assert.match(v.body, /already run ai-toolkit with/i);
  for (const shape of [/conda/i, /\buv\b/i, /system Python/i, /python_embeded/])
    assert.match(v.body, shape, `the body must name ${shape}`);
  // And a click that reaches the setting, not just prose about it.
  assert.equal(v.settingsSection, 'local-tools');
  assert.match(v.action, /ai-toolkit Python interpreter/);
  // The section id is a real Settings route, not a hopeful string.
  const registry = fs.readFileSync(new URL('../components/settings/registry.js', import.meta.url), 'utf8');
  const known = new Set([...registry.matchAll(/id: '([a-z0-9-]+)'/g)].map((m) => m[1]));
  assert.ok(known.size >= 8, 'settings registry did not parse');
  assert.ok(known.has(v.settingsSection));
});

test('an interpreter found in the folder becomes a one-click offer', () => {
  const found = ['/opt/ai-toolkit/python_embeded/python.exe'];
  const v = aitoolkitVerdict(trainStep({ valid: false, dir_valid: true, python_candidates: found }), DIR);
  assert.deepEqual(v.candidates, found);
  // ...and the prose still stands on its own when nothing was found.
  const bare = aitoolkitVerdict(trainStep({ valid: false, dir_valid: true }), DIR);
  assert.deepEqual(bare.candidates, []);
  assert.ok(bare.body.length > 80);
});

test('a folder that is not an ai-toolkit checkout gets its own, different answer', () => {
  const v = aitoolkitVerdict(trainStep({ valid: false, dir_valid: false }), DIR);
  assert.equal(v.kind, 'not_a_checkout');
  assert.match(v.headline, /run\.py/);
  // Not the interpreter story: that would send the user to the wrong setting.
  assert.doesNotMatch(v.body, /interpreter/i);
});

test('the first-install steps do not define a working ai-toolkit as "has a venv"', () => {
  const all = AITOOLKIT_INSTALL_STEPS.map((s) => s.text).join(' ');
  assert.match(all, /venv/);              // still the documented happy path
  assert.match(all, /conda/i);            // but not the only one
  assert.match(all, /portable/i);
  assert.match(all, /ai-toolkit Python interpreter/);
  assert.ok(AITOOLKIT_INSTALL_STEPS.some((s) => s.command), 'the clone command survives');
});

test('SetupPage renders the verdict instead of hardcoding the old sentence', () => {
  const page = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');
  assert.doesNotMatch(page, /set up its Python venv per the README/);
  assert.doesNotMatch(page, /set up its venv per its README/);
  assert.match(page, /aitoolkitVerdict\(step, dir\)/);
  assert.match(page, /<SettingsLink section=\{verdict\.settingsSection\}/);
});
