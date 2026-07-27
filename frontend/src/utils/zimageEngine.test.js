import { test } from 'node:test';
import assert from 'node:assert/strict';
import { zimageUnavailableReason, denoiseDescription } from './zimageEngine.js';

test('ordered failure reasons', () => {
  assert.match(zimageUnavailableReason({ enabledInSettings: false }), /disabled in Settings/);
  assert.match(zimageUnavailableReason({ comfyuiReachable: false }), /Configure ComfyUI/);
  assert.match(
    zimageUnavailableReason({ missingAssets: ['zimage_model'] }),
    /base model missing/,
  );
  assert.equal(zimageUnavailableReason({}), null);
});

test('present-but-invalid beats "ready"', () => {
  const r = zimageUnavailableReason({
    invalidAssets: [{ asset: 'zimage_vae', filename: 'z_image_ae.safetensors', blocking: true, verdict: 'html_or_text' }],
  });
  assert.match(r, /cannot be loaded/);
});

test('denoise bands read plainly', () => {
  assert.match(denoiseDescription(0.3), /sticks to the reference/);
  assert.match(denoiseDescription(0.9), /follows the prompt/);
});
