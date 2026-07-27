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

/** What the denoise dial currently means, in one short phrase. Img2img: LOW keeps
 *  the reference, HIGH follows the prompt. */
export function denoiseDescription(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 'default (0.65)';
  if (n <= 0.45) return `${n} · sticks to the reference, strong likeness, less variety`;
  if (n < 0.75) return `${n} · balanced (recommended)`;
  return `${n} · follows the prompt, looser likeness`;
}
