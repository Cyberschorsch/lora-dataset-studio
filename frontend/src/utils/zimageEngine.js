/* Z-Image Turbo — the pure, testable half of the engine's UI. PURE JS (no JSX)
   so `node --test` can import it directly, same split as kreaEngine.js.

   Z-Image is core-nodes-only (no custom pack), so there is NO node-pack branch —
   the failure modes are: disabled in Settings, ComfyUI unreachable, a weight file
   missing, or a weight file present-but-invalid (an HTML licence page or a
   truncated download saved as .safetensors). One branch each, in fix-order. */

export const ZIMAGE_ASSET_LABELS = {
  zimage_model: 'base model',
  zimage_text_encoder: 'text encoder',
  zimage_vae: 'VAE',
};

export function zimageMissingLabels(missing) {
  const set = new Set(Array.isArray(missing) ? missing : []);
  return Object.keys(ZIMAGE_ASSET_LABELS)
    .filter((k) => set.has(k))
    .map((k) => ZIMAGE_ASSET_LABELS[k]);
}

export function zimageUnavailableReason({
  enabledInSettings = true, comfyuiReachable = true,
  missingAssets = [], invalidAssets = [],
} = {}) {
  if (!enabledInSettings) return '⚠ Z-Image Turbo is disabled in Settings (engines)';
  if (!comfyuiReachable) return '⚠ Configure ComfyUI in Settings';
  const words = zimageMissingLabels(missingAssets);
  if (words.length) return `⚠ Z-Image ${words.join(' + ')} missing — download it in Setup, or drop it in yourself`;
  const broken = (Array.isArray(invalidAssets) ? invalidAssets : []).filter((i) => i && i.blocking);
  if (broken.length) {
    const b = broken[0];
    const what = ZIMAGE_ASSET_LABELS[b.asset] || b.asset;
    const why = b.verdict === 'html_or_text'
      ? 'it is a web page, not weights — the download skipped a step'
      : 'the file is truncated or corrupt';
    return `⚠ Z-Image ${what} (${b.filename}) cannot be loaded: ${why}. Delete it and download it again.`;
  }
  return null;
}

/** What the denoise dial currently means, in one short phrase. It drives BOTH graph
 *  shapes — the sampler denoise on close-ups, and how much of the held head is
 *  repainted on restaged shots — so the wording stays about likeness, not repaint. */
export function denoiseDescription(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 'default (0.75)';
  if (n <= 0.55) return `${n} · sticks to the reference, strong likeness, less variety`;
  if (n < 0.85) return `${n} · follows the prompt with usable likeness (recommended)`;
  return `${n} · strongly follows the prompt, likeness drifts`;
}

/** Turbo vs Base, in one short phrase. Turbo is guidance-distilled (cfg pinned to
 *  1.0, negative prompt inert); Base takes real guidance and is much slower. */
export function variantDescription(v) {
  if (v === 'turbo') return 'Turbo · 8 steps, no guidance, fastest';
  if (v === 'base') return 'Base · ~28 steps, real guidance and negative prompt, much slower';
  return 'Auto · detected from the filename, falls back to Turbo';
}
