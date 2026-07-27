# Using the app

The workspace is a **guided flow**: each stage stays folded until the one
before it is done, and the progress rail on the left tells you where you are
and what's blocking the next step. You never have to guess what comes next —
this chapter just explains what each stage does and where the useful buttons
hide.

The walkthrough below follows a **character** dataset end to end because it
exercises the most stages; **concept**, **style** and the **image bank** each
get their own section after it. The flow is the same for all of them — only the
captioning rules and a few guards change with the dataset kind.

---

## The character walkthrough (reference photo → trained LoRA)

1. **Create the dataset** — Datasets → New. Pick **Character**, name it, set a
   **trigger word** (the token your prompts will use), and choose the **target
   model** (Z-Image / SDXL / Krea 2 / FLUX.1 / FLUX.2 Klein — changes the caption
   style; you can change it later).
2. **Upload the reference photo.** The app head-crops it automatically; use the
   crop editor (or *Reset to auto*) if the framing is off. Up to 3 extra angles
   can be added for multi-view consistency. **✦ Edit** retouches the reference
   itself from a prompt ("plain studio-grey background", "add glasses") through
   **any engine that generates** — and shows you a Before/After to Keep or
   Discard. **Klein** and **Krea 2 Edit** run it on your own ComfyUI: free,
   private, and the sane choice when you expect to retry the prompt a few times;
   they appear only when your ComfyUI can actually run them, and say what is
   missing when it nearly can. **Nano Banana Pro, ChatGPT and OpenRouter** charge
   per edit, on the model set in *Settings › Image engines*. The engines differ in
   what else they look at: the API ones take the dataset's extra references *and*
   images you drop into the dialog, Klein takes the dataset's extra references,
   and Krea edits the main reference only — the dialog says which, before you
   press Generate.
3. **Generate variations** — pick an engine (Nano Banana Pro, ChatGPT,
   OpenRouter, or local Klein) and fire the **variation catalog**: 53 shots
   across expression, angle, lighting, framing, outfit and background, each
   wrapped in an identity guard so the face stays the same person.
4. **Import** your own photos too (drag & drop) — each is auto-cropped to the
   face on the way in.
5. **Auto-classify framing.** A local vision model tags every image
   **face / bust / body / back**; the badges feed the composition meter.
6. **Curate** — keep / reject / crop, guided by the live meter targeting
   **12 face · 6 bust · 6 body · 1 back**. Watch the face-similarity badges
   (green = strong match, orange = review) to drop off-identity shots before
   they poison training.
7. **Caption** — one click captions the kept set (prose or booru tags,
   matched to the target model). The **identity-leak check** flags any caption
   that describes hair/face/skin — fix every flagged one. A find/replace +
   tag-frequency panel sweeps the whole set at once; its **💾 Write .txt
   files** button drops a kohya-style `<image>.txt` next to each kept image
   in the dataset folder (same format as the export ZIP) for external tools.
8. **Fix individual shots** — every generated tile has a ✏️ button: edit the
   exact prompt that made it and regenerate in place, without losing the rest.
9. **Train** — the pre-flight check runs the full checklist (count, balance,
   captions, leaks, duplicates). It no longer *blocks*: leaking captions and
   near-duplicates are editable right inside the confirm, and missing captions
   just ask you to **Start anyway** (captions stay strongly recommended). Steps
   are computed automatically; ⚙️ Advanced options exposes every knob (each with
   its own why/how) and a **Presets** row — apply a shipped ★ recipe (*Krea
   character*, *Concept*, *Style*) or save/import/export your own as a JSON.
   No GPU? **☁️ Train in cloud** rents one per run. Watch this run — and every
   other, cloud or local — from the **🏋️ Runs** tab, where you can retry a
   failed run (↻), continue a finished cloud run for more steps (▶), and download
   the LoRA.
10. **Pick the best checkpoint** — open the **Test Studio** from the dataset:
    grid-test checkpoint × strength, vote, rank by face similarity, and star ★
    the winning settings. The last checkpoint is almost never the best one.
11. **Export** — at any point, **Export ZIP** gives you the curated, captioned
    set as a standard ai-toolkit dataset. Nothing is locked in.

## Retry a reference edit

After an **✦ Edit** candidate appears, **Retry** repeats the exact prompt, selected
engine and temporary reference files used for that candidate. Use **Try another
prompt** only when you want to change the instruction. The candidate also names the
engine/API that actually returned it, so you can see which service produced the
image before you Keep or Discard it.

## Test a run straight from Runs

The **🏋️ Runs** hub is also a shortcut back to the right **Test Studio**. Every
active or recent run that still has a dataset shows **🧪 Test in Studio** beside
its actions. Click it to open Studio with that run’s dataset already selected —
there is no need to return to the library and find the dataset first. The button
is also available on a folded Recent dataset group, so you can start comparing
checkpoints without expanding its run history.

## Recover a paused Test Studio batch

If ComfyUI drops while Test Studio is processing a batch, the affected tile says
**paused** and shows its paste-safe reason. The queue deliberately stops there:
it does **not** submit or start a later job, so nothing else runs against a
recovered or different ComfyUI state.

First recover or restart ComfyUI. For a valid local portable install,
**Setup → ComfyUI → ▶ Start ComfyUI** uses the app's fixed local-safe profile.
It does not read, change or execute any `.bat` file; your existing launcher and
its settings stay untouched. Once ComfyUI is responding, **Cancel** the paused
batch and resume it from Studio. That makes the next prompt an explicit choice,
never an automatic continuation.

## Concept datasets (an object or action, not a person)

Pick **Concept** at creation and describe the concept in the required field —
the captioner needs to know exactly *what to omit*. What changes vs character:

- **No reference photo.** Images come from **import** or the built-in
  **scraper** (paste a gallery URL or run a Reddit keyword search, tick the
  frames you want, they land straight in the dataset — deduplicated and
  quality-filtered). Already have a kohya-style dataset on disk (images +
  same-name `.txt` captions)? **⋯ More → 📂 Import from folder…** merges it in
  from a pasted folder path — captions attach, duplicates are skipped (a ZIP
  works too, via **📦 Import dataset**). On gallery sites (PornPics), a category/tag/search scan
  shows **the same previews the listing page does** — one per gallery, the shot
  that actually matches your keyword. Tick **Scan full albums** to pull every
  photo of each matched gallery instead, or paste a single `/galleries/…` URL
  to get that whole album. Sex.com works the same way for keyword searches
  (`sex.com/en/pics?search=…`) — every pin **is** a single matching image, so
  there is no album option to worry about. Civitai searches return **SFW
  results only** unless you add a Civitai API key in **Settings → Scraping &
  sources**.

  > **Reddit says "wait N seconds" (429)?** By default Reddit scans share a
  > public client id (and its ~1000 requests / 10 min quota) with many other
  > people, so it can be exhausted before your first scan. Add your own free
  > client ID in **Settings → Scraping & sources** — a one-minute, step-by-step
  > guide is built into that page.
- **Captions invert**: they describe everything *except* the concept, so the
  concept is what binds to the trigger. The leak check watches for stray
  descriptions of it.
- **Person masking is off** (a person mask would erase the very thing you're
  teaching), and imports keep the full frame instead of head-cropping.
- **You can mask the faces instead** — the opposite polarity. *Advanced training
  options ▸ Mask faces* weighs the detected faces down in the loss so the concept
  learns the act, not the people demonstrating it, and you can preview exactly what
  it would cover before training. Off by default. See the dataset guide, §8.

## Style datasets (a global aesthetic)

Pick **Style** at creation. What changes:

- **No trigger word** — the style tints every image once the LoRA is loaded.
- **Captions describe content only** (never the rendering), and they're
  optional; caption dropout rises so the style generalizes.
- **Step count switches to a sublinear √n scale** built for the large sets
  (hundreds of images) style LoRAs want.

## Caption your images in another tool

You are not locked into the captioners shipped here. The round trip is:

1. **⬇ Export ZIP** from *Import & export*. The archive is a plain kohya layout —
   one folder of `image.png` + same-name `image.txt` pairs. If some kept images
   have no caption yet, the app asks before exporting instead of refusing:
   confirm and their `.txt` files come out empty, ready to be filled.
2. **Caption them wherever you like.** Any tool that writes a `<image>.txt`
   sidecar next to each image works — that is the convention this app reads,
   whatever the file names are and whatever folder depth you use.
3. **📦 Import dataset (ZIP)** (or **📂 Import from folder…**) with the same
   images and their new `.txt` files. Images already in the dataset are **not
   duplicated**: their caption lands on the row that already holds them, and the
   toast says how many were applied.

Two things worth knowing before you start:

- **A caption you already wrote here is never overwritten.** Re-importing only
  fills the empty ones; the toast reports the rest as *"kept the caption written
  here"*. Clear a caption in the app first if you want the external one to win.
- **Only the caption travels back.** Statuses, scores and framing stay as they
  are here — the returning archive is read as captions for images you already
  have, not as a replacement dataset.

**A Style dataset asks louder, on purpose.** A Style LoRA learns everything its
captions do *not* name, so an empty `.txt` teaches it nothing; the export
confirmation says so before letting you through. Cancelling takes you straight
to the captions instead.

*Requested by Qeeyana (Reddit).*

## Krea and the shape of your reference photo

**Krea 2 Edit now follows the framing of the selected shot card** during dataset
generation. The reference photo still anchors identity, but Krea's v1.2 Fit path
adapts it to the requested output: **1:1** for face cards and **3:4** for bust,
body and back cards. A square reference therefore no longer forces a full-body or
sitting card into a tight bust crop.

This is deliberately limited to Krea dataset variations. The separate **Edit
reference** action keeps the source layout for a free-form edit, while Klein and
the API engines keep their existing, separate generation paths.

You can still crop a reference when you want a different identity anchor or
composition, but you no longer need to crop it merely to give a selected body
card enough vertical room. Reference quality still matters for likeness; the
selected card now owns the output frame.

## Z-Image — a local engine without an identity model

**Z-Image** is the third local engine, running through your own ComfyUI alongside
Klein and Krea 2 Edit. Where Klein restages your reference and Krea 2 Edit holds an
identity with a dedicated edit LoRA, Z-Image has **no identity-edit model at all**:
it builds each shot from your reference photo directly with a fast base model. That
makes likeness **looser than Klein or Krea** at every setting — a good match for
style and concept datasets, and on a character dataset it needs the denoise dial
(below) turned down to hold a face.

It needs three weight files inside your ComfyUI: a Z-Image Turbo base model, the
Qwen3-4B text encoder, and the matching VAE. Unlike Krea 2 Edit these are **not a
manual install** — **Setup ▸ Install everything** downloads all three with one
click, and the engine card in the workspace names whichever is still missing.

**How a shot is built.** The reference Z-Image works from is your *square head crop*,
so painting every shot over it could only ever hand back a head — which is why body
shots used to come out as busts. Close-ups now render at the shot's own aspect, while
body, bust and wide shots are **restaged**: the shot is built on a fresh canvas at its
own aspect, your subject's head is composited in and held, and the pose, outfit and
background are generated around it. Back-view shots never get a face pasted in
backwards — they lean on your full-frame original when your dataset kept one, and
otherwise carry no identity link at all.

**Reference denoise** (`zimage.denoise`, Settings ▸ Image engines) is the identity ↔
prompt dial: **low** keeps more of the reference (stronger likeness, less variety),
**high** follows the prompt more closely (looser likeness, more variety). 0.75 is the
balanced default; **sampler steps** (`zimage.steps`, default 8) sits right next to it
for the rare case where more render time actually helps.

**Want it to follow the prompt harder?** Download the optional **Z-Image Base**
checkpoint from Setup ▸ Install. Turbo is guidance-distilled, which means it runs
without guidance and ignores negative prompts; Base takes both, and follows your shot
description considerably more closely on restaged shots — at several times the
generation time.

## Your own shot catalog (JSON import)

The workspace ships a built-in shot catalog per subject type (53 shots for a
human, ~59 for an animal, 55 for an anime character, and so on). If you want shots nobody wrote for you —
40 breed-specific poses for a dog, a product line's signature angles — you don't
have to type them one at a time. Open **📥 Shot catalog (JSON)** under the shot
grid.

**Export first.** The exported file is the format, and the example an LLM needs:

```json
{
  "format": "lds-shots/1",
  "subject_type": "animal",
  "shots": [
    {
      "label": "Dog, zoomies on the lawn",
      "framing": "body",
      "prompt": "full body photo of the animal running fast across a lawn, side view, sunny day"
    }
  ],
  "examples": []
}
```

Then ask a chat assistant for more shots *in that exact shape*, and import the
file it gives you.

Each shot needs three things:

- **`label`** — a short name, max 80 characters, shown on the card. It must be
  unique: not a built-in label (of *any* subject type), and not one of your
  existing shots. The app refuses a collision and tells you which label is at
  fault — two shots sharing a label would make it resolve the wrong prompt the
  day you regenerate one.
- **`framing`** — exactly one of `face`, `bust`, `body`, `back`. Anything else is
  refused; it is never quietly remapped.
- **`prompt`** — the text sent to the image engine, max 500 characters.

`nsfw: true` is optional and only has an effect when Klein is the only engine
checked. Everything under **`examples`** is ignored on import — that's how the
export can show you samples without them coming back as duplicates. Any other
field (including `aspect`) is ignored too, and the import summary says so: an
imported shot uses its framing's default aspect ratio.

**Nothing is written until you confirm.** The app reads the file, lists what
would land and what it refuses (naming the entry and the reason), and waits. A
40-shot file whose 37th entry is broken never leaves 36 shots half-imported.

Imported shots appear in their own **📥 Imported** group after the built-ins, one
set per subject type. They never replace a built-in, you can delete them one by
one or all at once, and they're stored with the app — not in the browser — so
they survive a cache wipe, show up on your phone and ride along in the backup.

### Keeping a shot you wrote by hand

The **✨ Custom shot** box below the grid is the quick way to add one shot: type a
prompt, pick a framing, Add. Those cards are stored **in your browser**, so
clearing its data takes them with it.

Any card you want to keep, press **Keep** on it. It moves into the 📥 Imported
group and is saved with the app, exactly like an imported shot — surviving a
cache wipe, following you to another device, included in the backup. The card
keeps its identity, so a shot preset that had it selected still works. If its
label happens to clash with a built-in shot or with one you already imported, the
app says which label and refuses rather than creating a duplicate; rename the
card (remove it and add it again) and press **Keep** once more.

*Feature requested by ashish.sinha (Discord).*

## Back up everything

The **💾 Back up everything** button on the Datasets library packs your whole
setup into a single file so you can move to a new machine — or recover from one
— without losing anything.

- **What's inside**: every dataset (all images, captions, statuses, face and
  watermark states, references), its **training history** (which runs produced
  which version, the settings each used), plus your **settings** — engine
  choices, training defaults, cloud tuning, watermark preferences. It's a
  *logical* backup, one entry per dataset, not a raw disk dump.
- **Include trained LoRAs** (checkbox next to the button): also bundle the
  trained `.safetensors` files themselves. These are large — hundreds of MB per
  checkpoint — so it's **off by default**; the light training history above is
  always included, so a dataset comes back under **Trained** either way. Tick it
  when you want the finished LoRAs to travel too.
- **What's never inside**: your **API keys, Hugging Face token and scraping
  credentials**. They are deliberately left out so the file is safe to copy
  around; re-enter them once on the new install.
- **How it runs**: in the background. A library can be gigabytes, so you get a
  live "X / N datasets" progress bar and can keep working. When it's done, use
  **⬇ Download** to save the archive, or **📂 Open folder** to find it on disk.
- **Restoring**: hand the master archive to the same **📦 Import backup** button.
  It restores your settings (without overwriting keys you've already entered),
  rebuilds each dataset **and its training history** — so it lands back under
  **Trained** instead of "Not trained yet", with its runs in the Runs hub.
  Bundled LoRA files are re-deployed to ComfyUI when it's configured on the new
  machine; if it isn't, they're reported as skipped and the **Trained** status
  still stands (the run is what marks it trained, not the file on disk). Nothing
  is ever overwritten — a dataset whose name already exists comes back with a
  `(restored)` suffix — and you get an honest final report of what was restored,
  renamed or skipped.

## The image bank (triage a big folder)

You exported 9 000 unsorted images from Telegram (or a scraper dumped a
mountain of files) and a dataset only needs the best 30–150 of them. The
**🗃️ Bank** tab is the triage funnel that gets you there — without ever
touching the folder itself:

1. **Create a bank** — give it a name and paste the folder path. The app
   inventories every image in place (subfolders included). Nothing is copied,
   nothing is modified; rejecting an image is a reversible status, never a file
   deletion. The folder stays LIVE: keep dropping images into it and they are
   picked up automatically the next time you open the bank list or the bank
   itself ("42 new image(s) found in the folder"), as undecided images ready
   for the next scan — your existing keep/reject decisions, scores and captions
   are never touched. Files you removed from the folder are reported at the top
   of the bank, never deleted from it, so an unplugged drive can't wipe your
   triage.
1bis. **🕸 Scrape the web into a bank** — you don't need a folder you prepared
   by hand. Unfold **🕸 Scrape the web into a bank** on the bank list, choose a
   destination (a **new bank**, or **add to an existing one**), then scan a
   gallery URL and pick images exactly as you would for a dataset. They are
   downloaded into that bank's own folder and inventoried on the spot.

   Two things are worth knowing, because they are the whole point:

   - **Nothing is filtered on the way in.** Scraping straight into a *dataset*
     applies training-grade gates (short side ≥ 768 px, ratio ≤ 3:1, perceptual
     de-duplication) *before* anything is stored. A bank is the step **before**
     that judgement: "too small", "near-duplicate" and "wrong framing" are
     verdicts its own passes produce, with thresholds you move. So the bank
     stores what it downloaded and lets you decide. If you already know what you
     are collecting, scraping straight into a dataset is still the shorter road.
   - **A second scrape resumes the same bank.** Pick *Add to an existing bank*
     and the new images join the pile — nothing is replaced, and no triage
     decision you already made is reset. Re-downloading the exact same file
     lands on the same name instead of piling up copies; that is file identity,
     not a duplicate verdict (the bank's own passes own that word).

   The rest of the funnel is unchanged: scan, cull, promote into a dataset.
2. **🔎 Scan quality** — a background pass (CPU only, a few minutes even on
   thousands of images) scores every file: sharpness, noise, flat/empty
   frames, resolution — and groups **near-duplicates**. The flags follow the
   thresholds in *Settings → Captioning & quality*; because the raw scores are
   stored, tuning a threshold re-sorts the bank instantly, no rescan. The same
   pass also answers two questions the file itself lies about — see
   *Is this image really what it says it is?* below.
3. **Cull** — use the filter chips (🌫 Blurry, 📺 Noisy, ⬜ Flat, 📐 Small,
   🧇 Soft detail, 🎞 Black bars, ≈ Duplicates) to review the worst
   offenders first. **🧹 Auto-reject
   flagged…** clears whole categories in one click (your manual ✓/✕ are never
   flipped). In the Duplicates view, resolve every group at once with **keep
   best** (highest resolution, then sharpest) or **keep first**, or pick the
   keeper by eye.
4. **👥 Group by person** — the face pass (needs the Quality tools from Setup)
   detects the dominant face of every remaining image and clusters the bank by
   person, *no reference photo needed*. Click a person card to see only them,
   select all, keep or reject. Embeddings are cached, so re-running after a
   cull is much faster.
5. **🏷️ Caption & 🔍 search** — caption the bank with the same engines your
   datasets use (JoyCaption / Ollama vision, your *Settings*). Hit **🏷️ Caption
   all** to describe every not-yet-captioned image, or select some first to
   caption just those. It runs in the background, frees the GPU like the other
   passes, and is Stop-able mid-run. The captions are plain descriptions (no
   trigger word, nothing omitted) whose real job is **search**: type into the
   🔍 box — `red dress`, `sunset`, a file name — and the grid filters to
   matching images, combinable with every other filter. It's the fast way to
   find shots in a 9 000-image dump.
6. **⬆ Promote** — the kept images are **copied** into the dataset you choose
   through the normal import path: normalized to webp, near-duplicates already
   in the dataset skipped. Any bank caption **rides along**, so a captioned
   selection starts already captioned in the dataset. From there they get
   everything datasets have — captions, watermark cleaning, face scoring against
   a reference, training.

Work the funnel in that order: quality first (cheap, catches the trash), then
subject, then selection. A promoted image keeps its ⬆ badge in the bank so you
always know what's been used where.

**🎨 Curate down to the right subset.** Culling removes the bad shots; curation
picks the *good* subset — and it's most of what makes a LoRA good. Once **✨
Score** has run (it caches a CLIP embedding per image), the **Curate** row under
the selection bar offers two selectors that cost no extra GPU time:

- **🎨 Pick diverse** — enter a number and it selects the images that best
  *cover the variety* of what you're looking at (varied angles, outfits, scenes),
  instead of that many near-identical frames. It's the antidote to a dump of
  4 000 shots of the same pose: ask for 60 and you get 60 that actually differ.
  **Skip the odd ones out** (the slider under the number) is why they are the
  *right* 60. "Most varied" is computed as "farthest from everything already
  picked", and the image that is farthest from everything in a collected bank is
  usually not a nice unusual shot of your subject — it's the meme, the screenshot,
  the botched frame, the one photo of somebody else. The slider discounts an image
  for being *alone in the bank*: at the default **50%** an image that resembles
  nothing else has to be far more interesting than a normal one to earn a slot,
  and at **100%** it is all but excluded. It never works the other way round —
  anything as typical as the median of the bank is left completely alone, so this
  cannot turn your 60 into 60 look-alikes. Set it to **0** for the pure-coverage
  behaviour the button had before this setting existed. On a very large bank the
  first click takes a few seconds (it reads every image's neighbourhood once);
  the button says *Sampling…* while it does.
- **⚖️ Balanced pick** — see [Pick a balanced set](#pick-a-balanced-set) below: the
  same sampling, but spread evenly over your **framings** instead of taken off
  the top of one ranking.
- **🎯 Similar to selected** — select **one** image as a reference, and it ranks
  everything by how much it looks like that image and selects the closest N — the
  fast way to pull one person or one look out of a mixed export.

Both honour whatever filter and 🔍 search are active ("the 60 most diverse of
*this* subfolder"), and both just **select** — the images light up and you review
them with the same ✓ Keep / ✕ Reject / ⬆ Promote bar. Nothing is auto-kept or
deleted, so a selection you don't like costs one click to clear.

**📐 Classify framing** tags every non-rejected image by *shot type* — face
close-up, bust, full body or back view — using the same detector the datasets
use. The result becomes a row of **📐 Framing** filter chips (compose with every
other filter and search), so balancing a character set's angles is a couple of
clicks. It's a GPU vision pass; add it to **🚀 Launch all** to have it run
overnight with the rest.

**📊 Coverage advice** (idea by [@antonp](https://github.com/perfectgf/lora-dataset-studio))
is a read-only panel next to the Curate row. From what you've **kept** (or every
non-rejected image before you've kept anything), it says in plain sentences what
leans and what's thin for a good LoRA — *"70% face shots, add body/back"*,
*"person #1 is 60% of the set — one subject or a mix?"*, *"only 8 kept, most
families want 20+"*. It's **advice only** — nothing is kept or rejected — and
pure maths on data the passes already computed, so it costs no GPU. The
framing-balance line needs the 📐 Framing pass to have run; without it the panel
still covers person mix, style spread and resolution and hints to run framing.

The advice becomes a gesture with **⚖️ Pick a balanced set…** at the bottom of
the panel — see [Pick a balanced set](#pick-a-balanced-set).

**🗑 Delete rejected from disk** (next to Promote) is the one exception to the
"your source folder is never modified" rule, and it's opt-in. Once you're happy
with your triage, it removes every image you marked ✕ rejected from its source
folder — the actual files, not just the status. It asks you to type **DELETE**
first, and tells you where the files will go before you confirm: your OS trash
when [`send2trash`](https://pypi.org/project/Send2Trash/) is installed, the
app's own Trash otherwise (recoverable until you empty it from Settings), and a
permanent delete only when neither can take the file. Kept and undecided images
are never touched, and a file it can't remove (locked, read-only) is reported
and left alone rather than aborting the batch.

⚠️ A bank doesn't own its folder, so two banks can point at nested folders and
list the **same files**. That's harmless while you triage — decisions live on
the bank — but deleting from disk in one bank removes those files from the other
too, along with every decision you made on them there. The app says so when you
create such a bank, and the confirmation names the other bank and how many of
its files are about to disappear.

**🚀 Launch all** does the whole funnel for you in one go. Tick which passes
run and how auto-reject behaves, hit Go, and walk away — it chains *scan →
auto-reject → score → find watermarks → group by person → classify framing →
(optional) caption* in that exact order. Two things make it safe to run overnight: a pass whose
tool isn't installed, or a moment when the GPU is busy with a training run, is
**skipped with a reason** instead of failing the whole run; and because
auto-reject runs *before* the heavy passes, scoring/watermarks/person only ever
process the survivors, never the images you just rejected. Captioning is the one
pass left **off by default** (it's the slowest GPU pass and a clean-up run
rarely needs a description on every shot). Stop it any time — and when you come
back, a saved report at the top of the bank tells you exactly what ran, what was
skipped and why, with the headline counts.

## Pick a balanced set

Advice is only half the gesture, so **📊 Coverage advice** ends with **⚖️ Pick a
balanced set…** (the same button sits in the **Curate** row). It answers a
question no per-image score can ask: *does my set cover what I want to be able to
generate?*

Ask **🎨 Pick diverse** for 20 images out of a bank that is 47% full body, 35%
bust, 12% face and 6% back views, and you get roughly those proportions — on a
synthetic reproduction of exactly that shape it returned **0 face shots and 0
back views**. The LoRA then renders one shot type well and the rest badly, and
nothing ever said so. **⚖️ Balanced pick** returns **5 face, 5 bust, 5 body, 5
back** out of the same pool, each bucket filled with the *same* most-varied
sampling — and the same **Skip the odd ones out** guard — that 🎨 Pick diverse
uses.

- **Balance on** — **Framing** by default. It is the axis that carries real
  information: on a one-subject bank, person groups are sparse and split into
  many small, arbitrary clusters, so balancing on them spreads a selection over
  noise. **Framing × person** is there for a dump that genuinely holds several
  subjects.
- **When an axis can't be satisfied**, it says so instead of quietly filling the
  gap: *"Only 3 back images exist in this filter — an even split wanted 15"*. The
  freed picks go to the buckets that have room, so asking for 60 still gives you
  60 — the deficit is reported as a number, never hidden. If even that isn't
  enough, it says how many you actually got and why.
- **The result is always stated** — *"Selected 60 of 60 requested, spread over
  framing: 15 face, 15 bust, 15 body, 15 back"* — as text, per bucket, next to
  what each bucket had available. There is no chart you have to read.
- **An unlabelled bank is the normal state**, not an error. Nothing has a framing
  until the 📐 Framing pass has run, so the button says which pass is missing and
  how many images it would bring in, rather than returning an empty or misleading
  selection. 🎨 Pick diverse keeps working without it.

Like the other selectors it honours the current filter and search, and it only
**selects** — nothing is kept, rejected or deleted.

## Is this image really what it says it is?

Two things a file will happily lie about, both measured by the ordinary
**🔎 Scan quality** pass — plain CPU work, no extra install, no GPU.

**Its size.** An image enlarged from 512 px to 2048 px still *reports* 2048, so
it walks into a dataset as a high-resolution shot and the LoRA learns
interpolated mush. The scan measures how far real detail actually goes and says
it in pixels on the image's details line: *"2048 px stored · ~512 px of real
detail"*. The worst offenders sit behind the **🧇 Soft detail** filter chip,
and *Settings → Captioning & quality → Real-detail minimum* moves the bar.

Treat it exactly like the sharpness score: **a shortlist, not a verdict.** A
photo with motion blur, a portrait with the background thrown out of focus, and
a heavily denoised phone shot all genuinely lack fine detail and all read the
same way as an enlargement — which is fine for choosing training images (a LoRA
learns as little from either), but it is not proof the image was ever resized.
Look before you mass-reject. Two honest limits: a *nearest-neighbour* enlargement
is invisible to it (blocky pixels are real high-frequency detail), and large
enlargements are under-stated, so the pixel figure ranks images rather than
recovering the original file's size.

**Where it came from.** The scan reads the file's own metadata and sorts the
bank with the **🔎 Origin** chips:

- **🤖 AI** — the file still carries generation metadata: a ComfyUI workflow
  in the PNG, A1111-style `parameters`, or the C2PA/XMP "generated" marker the
  commercial generators write. Certain when present.
- **📷 Camera** — the file still carries camera EXIF (make, model, exposure).
  Strong evidence it was actually photographed.
- **❔ Unknown** — nothing left to read. **This is the normal answer**, not a
  failure: scrapers, chat apps and social networks strip metadata on sight (on a
  36 000-image Telegram export, *every single file* landed here). It is not
  evidence the image is a real photo, and it is not evidence it is AI — it is
  the absence of evidence, which is why it is its own answer instead of being
  quietly folded into "not AI".

On an image whose metadata is gone, the details line may add a *hint* when the
dimensions are a standard generator size (1024×1024, 832×1216, 896×1152…) and
there is no camera EXIF. It says it is a hint; plenty of crops and downloads
land on round numbers too.

Two smaller facts come free with the same pass: **🎞 Black bars** flags flat
letterbox/pillarbox padding (video screenshots, stills padded into a square,
which survive a training crop), and the **JPEG quality** of the last save is
shown as-is — a low figure means the file has been through a re-encoding
pipeline, but it is far too common to be worth a filter.

A bank you already scanned picks all of this up on its next **🔎 Scan** — the
pass re-visits the images that predate these measurements on its own. You do not
need a full rescan.

## Find bank images by describing them

Under **Curate**, **🔤 Find by text…** ranks images by how close they are to a
phrase you type — `brunette outdoors, wide shot`, `red dress against a white
wall`, `close-up, harsh flash`. It reuses the embeddings **✨ Score** already
computed, so there is no extra model, no download and no GPU work; searching
while a LoRA trains is fine.

**It is a ranking, not a filter.** Every image scores *something* against every
phrase, so a result list always comes back full. The panel therefore reports the
similarity of the best and of the last result, and tells you how far apart they
are — *"all about equally close"*, *"the last ones are noticeably looser"*, or
*"the tail is much weaker than the top"*. That spread is the useful signal: it
says whether you can trust the bottom of the list.

**Do not read those numbers as percentages.** They are much lower than intuition
suggests. Measured on a real bank (48 images drawn from 8 unrelated datasets,
using the exact model the app uses — ViT-L/14, `openai` weights):

| | Range |
|---|---|
| Top-1 results verified correct by eye | **0.177 – 0.233** |
| Guaranteed-unrelated image/phrase pairs | median **0.112**, up to **0.197** |

So 0.22 is not "22% of a match" — it is roughly as good as this model ever gets.

**And this is why there is no similarity slider.** Look at the two rows again:
the unrelated *ceiling* (0.197) is **higher** than two genuinely correct answers
(0.177 and 0.178). The distributions overlap, so no cut-off separates "relevant"
from "unrelated" — anything below ~0.20 lets false positives through, anything
above ~0.18 throws away true matches. A threshold control would be a knob on a
boundary that does not exist, so the app gives you a result *count* instead and
shows the ranking honestly.

The app never compares your scores against those figures either. It measures
what a *typical* image of **your** bank scores for **your** phrase, and describes
the results relative to that — which is the only version of the question that
survives a different bank.

**On a bank that is mostly one subject** — the normal case here — expect the
ranking to flatten. Images of the same person score 0.60–0.89 against *each
other*, far above any text score, and a query's ability to discriminate
compresses by 30–70%. The summary will say *"barely above what any image here
scores — the order is a hint at best"* when that happens. Believe it: at that
point the first result is not meaningfully better than the tenth.

It searches **inside the current filter**, exactly like 🎨 Pick diverse and 🎯
Similar to selected. So "wide shots, in this subfolder, among the undecided" is
just a filter plus a phrase; nothing needs a second search grammar. Results land
as a normal selection you review with ✓ Keep / ✕ Reject / ⬆ Promote — nothing is
kept or deleted for you. **Clear search** returns to the full grid.

**Images that were never scored cannot be found by any phrase.** Rather than
letting them vanish, the summary counts them: *"3 of 27 images in this filter
have no ✨ Score embedding yet and could NOT be searched."* Run ✨ Score to
include them.

### What it is good at, and what it is not

CLIP reads a picture as a whole. It is reliable for **subjects, styles, framing,
setting, materials and colour**, and unreliable for three things in particular:

| Ask for | What you actually get | Measured |
|---|---|---|
| **Counting** — "two people" | Photos of people, any number. | On a two-person image, "two people" beat "one person" by **0.001** — pure noise. It separates "one" from "several" at best. |
| **Negation** — "without glasses" | *More* glasses, not fewer. | On a photo of an astronaut **wearing** a helmet: "with a helmet" **0.212**, "without a helmet" **0.217**, plain "an astronaut" **0.219**. The negation scored **higher** than the affirmation. |
| **Spatial relations** — "to the left of" | Both objects, in any arrangement. | — |

The negation case is the one to remember, because it fails *silently and
backwards*: CLIP does not penalise "without", it simply ignores the word. Someone
searching `woman without glasses` gets women **wearing** glasses and has no way
to tell the search misfired.

These are properties of the model, not bugs to report. The workaround is to
describe what *is* in the frame rather than what is absent — "bare face" works,
"without glasses" does not — and to check counting and left/right by eye.

One last caveat, seen in the same measurement: a result can be right on the broad
trait and wrong on the detail. A generic indoor query returned a genuinely indoor
shot that was not the *kind* of indoor scene the wording implied. Text search
brings the likeliest images to the front; the final call stays yours.

### Why the first search takes a moment

The text encoder is CLIP's other half, and loading it costs about **ten seconds**
on the CPU. The app therefore keeps it warm after the first search — subsequent
searches are effectively instant — and releases it once you close the panel or
after ten idle minutes, because it holds roughly 2.4 GB of RAM while it lives.
Every phrase you have already searched is also cached on disk, so re-typing one
is free even after a restart.

On a memory-tight machine you can set `bank_scoring.text_search_idle_minutes` to
`0`: nothing is ever kept warm, and each new phrase pays the ten seconds instead.

## Review a bank one image at a time

Filter chips and bulk actions clear the obvious trash, but the last call —
*is this shot good enough for the LoRA?* — is made one image at a time, and
squinting at a 140-pixel thumbnail is not how you make it. **▶ Review one by
one** (above the grid) opens the images of the **current filter** full size, one
after the other:

- **✓ Keep**, **✕ Reject**, **⏭ Skip** — each one saves and jumps straight to the
  next image. The keyboard is the point: **K** keep, **R** reject, **S** skip,
  **←/→** move without deciding, **Esc** to leave. A few hundred images go by in
  minutes.
- **⏭ Skip** decides nothing (the image stays undecided) but is not shown again
  in that run — it's "not now", not "no".
- **🎲 Random order** walks what's left in shuffled order instead of folder
  order. On a scraped dump of 3 000 photos, sequential order means 200
  near-identical frames in a row; random gives you a representative sample from
  the first click. Ticking or unticking it mid-run only re-orders what you have
  **not** seen yet — nothing you already judged comes back.
- Under the image, the facts the passes already computed (resolution, sharpness,
  aesthetic score, NSFW, quality flags, person and duplicate groups) so you can
  call it without leaving the lightbox.
- The counter is honest — *12 / 340* over the snapshot taken when you opened the
  review, so a decision that drops the image out of the filter can't make the
  run skip images or loop. Each decision is saved on the spot: close after fifty
  of them and all fifty are there.

The ▶ button on a tile starts the same review **at that image**. A plain click
on a tile still selects it for the bulk ✓/✕/⬆ bar, so both ways of working stay.

## Promote a shortlist into a new bank

**⬆ Promote…** has two destinations, and picking the right one saves you a mess.

- **📁 An existing dataset** — the end of the funnel. The images are normalized
  to webp, deduplicated against what the dataset already holds, and become
  training material.
- **🗃 A new image bank** — for when you are not there yet. A 9 000-image dump,
  200 candidates isolated out of it, and you want to keep working on those 200
  apart: give the new bank a name and the selection lands in it, **un-triaged**,
  with every bank tool available again (scan, dedup, framing, captions, review).
  Nothing is committed to training.

With images selected in the grid, those are the ones that go; with nothing
selected, every **kept** image does.

Either way this is a **copy**. Banks never share their files, deliberately: the
app rewrites images in place (a re-crop, a watermark clean), so two banks reading
one file would stop being two banks at the first edit. The dialog therefore
states, before you click, **how many megabytes** the copy costs — a measured
figure for that exact selection, not an average. For photographs it is usually a
footnote; the line is there for the day a bank holds something heavier.

Your source bank is untouched by all this. It keeps every image, now marked ⬆
promoted, and your original folder is never written to — the copies live in the
app's own data folder, and deleting the new bank takes them with it.

If the copy cannot be written — a full disk, a drive pulled out — the new bank is
**discarded** rather than left holding half the shortlist and looking finished.
You are told what happened and nothing has changed.
## Undo the last bulk decision

A bank lets you mark hundreds of images with one click: select the whole filter
and press ✕, apply an auto-reject at a threshold, collapse every duplicate group,
or run 🚀 Launch all. That is the point of a bank — and it is also the click you
most want back when the threshold was wrong or the filter was not the one you
thought.

After any of those, an **↩ Undo** bar appears above the grid saying what
happened and how many images it moved. Press it and every one of those images
goes back to exactly what it was: its previous ✓/✕/undecided state *and* the
reason it carried. Images the action never touched are not touched here either —
if you had already kept a photo by hand and the bulk reject flipped it, undo puts
it back to **kept**, not to undecided.

The bar does not disappear on a timer, and it survives a page reload: the
decision it takes back lives in the app's database, not in your browser tab. It
stays until you use it, dismiss it, or run another bulk action.

**Its limits, stated plainly.**

- **One step.** Only the most recent bulk action is remembered. Run a second one
  and it replaces the first — this is a net under the click you just made, not a
  history of your session.
- **Until the app restarts.** The memory is in the running app. Restart it and
  the offer is gone; the decisions themselves are safely saved, as always.
- **It never over-claims.** If some of the images have left the bank since (a
  re-scan noticed the files were gone), or if you changed some of them yourself
  in the meantime — in ▶ Review, or in another tab — those are *not* overwritten.
  The result tells you exactly how many it restored out of how many, how many
  are gone, and names the ones a newer decision now owns.

**What is deliberately NOT offered.** Two bank actions have no undo, because a
half-working one would be worse than none:

- **🗑 Delete rejected** sends your source files to the recycle bin and drops
  their rows with everything the passes had computed about them. Files in the
  recycle bin are yours to restore, from your file manager — the app cannot do
  it for you, and it will not pretend otherwise. This action also withdraws any
  pending ↩ offer, since the images it pointed at are the ones just removed.
- **⬆ Promote** copies images into a dataset (or a new bank) through the normal
  import path. Un-promoting would mean deleting images in a dataset you may have
  already captioned, cropped or trained on. Delete them there if you want them
  gone.

The 🔄 rotate button needs no undo entry: turn the other way and the image is
byte-for-byte the original again.
## Sort a grid to review faster

Filters answer *which images*; sorting answers *which one first*. Both grids
have a **Sort** control, and it changes nothing but the order — the same images
match, the counts stay put, and every bulk action keeps operating on exactly
what the filters left.

In a **bank** (View ▸ Sort, next to the tile size):

- **Resolution ↓ / ↑** — megapixels, so a 900×900 outranks a wider 1200×300.
- **Aesthetic ↓ / ↑** — the 1–10 rating from **✨ Score**. ↓ puts your keepers on
  the first page; ↑ puts the duds there, which is usually the faster way to prune.
- **Sharpness ↓ / ↑** — the Laplacian variance from **🔎 Scan quality**. ↑ brings
  the blurry misses to you instead of making you hunt for them.

In a **dataset** (above the grid, next to the decision chips): **Face similarity
↓ / ↑**, the ArcFace cosine against your reference photo computed by **🎭 Analyze
faces**. ↓ is "who looks most like my subject", ↑ is the shortlist to cut.

Two things worth knowing:

- **Images a pass never reached always go last**, in both directions. An
  un-analysed image has no score — putting it first would bury the very images
  you asked to see.
- **A sort you have no data for is greyed out** and says which pass to run,
  rather than pretending to reorder. Run the pass, and it lights up.

In a bank the ordering is done by the database over the *whole* filter, not just
the page you can see — so **Select all in filter** and **▶ Review one by one**
walk the same order you are looking at.

## Compare an improved image with the original

Two things in the app never overwrite an image — they add a **candidate** next
to it, and leave the choice to you:

- **✨ Upscale & improve** in the dataset lightbox (a manual Klein pass, 2 MP by
  default);
- the automatic **small-image rescue** of scraped images under 768 px.

Open that candidate full screen and it now carries **⧉ Compare with original**.
The view splits in two named panes — *Original* and *Improved* (or *Klein
rescue*) — **side by side on a wide screen, stacked on a phone**, where width is
the scarce axis and two half-width thumbnails would prove nothing.

Both panes are the same size and both images are fitted inside them, so they are
shown at **the same scale and the same framing** even though the candidate has
more pixels. That matters: an improve pass rescales to a megapixel budget, and
two images displayed at different scales cannot be compared honestly.

**Zoom is off inside the comparison**, and the hint under the image says so. At
100 % a 2 MP result and a 0.5 MP original cover different parts of the subject —
that is not a comparison. Leave the comparison (⊟) and the usual click-for-100 %
inspection is back, on whichever image you are looking at.

When you **✓ Keep** a completed **✨ Upscale & improve** candidate, LDS keeps
both files but returns its original to **Undecided** automatically — so the
improved image is the one selected for training. This happens in the lightbox
and with bulk **✓ Keep**, even if you selected both tiles. Nothing is deleted:
you can still compare them, and can mark the original **Keep** again later if
you deliberately want to train on both.

If the original was deleted, rejected and purged, or simply never recorded (very
old rows), there is no button — a short amber note says why instead, so a
missing control can't be mistaken for a bug. Everything else in the lightbox —
✂ Crop, ⇄ Mirror, ✨ Upscale & improve — is unchanged and still acts on the
image you opened.

## Tune the Bank filter thresholds

The filter chips (🌫 Blurry, 📐 Small, ≈ Duplicates…) are verdicts, and every
verdict comes from a number. Those numbers used to live only in
*Settings ▸ Captioning & quality*, three screens away from the bank you were
triaging. They are now also under the chips themselves: open **🎚 Filter
thresholds** above the grid.

It is the **same setting in both places** — one value, seen twice — so anything
you change here applies to **every bank**, and the panel says so at the top.

The twelve knobs are grouped by the question they answer: **Image quality**,
**Duplicates**, **Size & framing**, **Content**, **Style**. The first two are
open by default; the rest fold away, and a folded group tells you how many of
its values you have moved off the default.

Three things each control tells you that a bare number cannot:

- **Which way catches more.** "Stricter" is not a direction. *Duplicate
  distance* is a distance in hash bits — **raise** it to catch more
  near-duplicates. *Semantic duplicate similarity* is a similarity — **lower**
  it to catch more. They sit side by side and they move opposite ways, so each
  field spells its own direction out in a sentence next to the input.
- **When it takes effect.** Eight of them re-sort the bank the moment you save,
  because the scan stores raw measurements and the verdicts are recomputed on
  every read — no rescan, ever. The other four are baked into stored groups by a
  pass, so they carry a button that re-runs that pass on the spot. Re-grouping
  duplicates is cheap: it walks the stored hashes and decodes nothing.
- **How many images it would touch.** As you change a read-time value, the panel
  asks the server how many images that number *would* flag and shows
  `1 240 → 3 019 images flagged` before you save anything. Nothing is written
  until you press **Save**.

Every field has **↺ Reset to default** (it only appears when the value is not
the default), and the header carries **↺ Reset all to defaults**. The defaults
come from the server, so they are always the real shipped values.

### What editing an image costs it

Crop, ✂ Mirror, ↺ Rotate and the watermark cleaners **overwrite** the file the
trainer will later copy verbatim, so whatever they discard is discarded for good.
They all follow one rule: **keep the file's format and re-encode it without losing
pixels.** A PNG stays a PNG, a WebP is rewritten losslessly (crop it ten times and the tenth
is identical to the first), and the file keeps a name that matches what is inside
it. JPEG is the exception nobody can fix — it has no lossless mode — so a JPEG is
re-saved at the highest practical quality with no chroma subsampling rather than
converted to something heavier to protect pixels that were already lossy.

Two honest caveats:

- **A large crop still resamples.** A box longer than 1024 px is normalised *down*
  to a 1024 px long side, and only the *encoding* is lossless — that downscale
  never can be. A box at or under 1024 px is a pure cut, so it is lossless end to
  end, as is the watermark **✂ auto-crop**, which only cuts and never resizes.
- **Files get bigger.** A cropped photo that used to weigh ~200 KB now weighs
  ~950 KB. That is the price of not throwing pixels away. Thumbnails and the
  copies uploaded to a generation API are unaffected: they stay small on purpose.

### A crop is never enlarged

A crop used to be stretched *up* to a 1024 px long side as well: select 240×180
and the file stored was 1024×768. That enlargement invented no detail — shrinking
such a file back recovers the real crop almost exactly — and since the encoder
went lossless it cost roughly **6× the bytes** for nothing. A crop now keeps its
own size, and only comes *down* to 1024 px.

Two consequences worth stating plainly:

- **Your dataset can end up mixing image sizes.** That is fine — training buckets
  images by size — but a tile cropped out of a small area really does carry less
  detail than a native shot of the same framing, and it always did; it just used
  to look like 1024 px.
- **The composition meter says so.** The old ⚠ *Upscaled* line is now
  ⚠ *Under training resolution*. It fires on the same measurement and means the
  same thing it always meant: this framing bucket is filled by cropping far into
  photos rather than by native shots — add native shots for it. (Images imported
  with the automatic head-crop *are* still enlarged to 1024, so both shapes land
  under the same warning.)

Images cropped **before** this change keep the enlarged pixels they have.

Images you cropped **before** this changed keep the pixels they have — nothing is
re-processed retroactively, and re-cropping an already-degraded file cannot bring
back what the old encoder removed.

## Why a ↻ re-run button is greyed out

A bank runs **one pass at a time**. While a ✨ Score, a 🔎 Quality scan or a
🚀 Launch all is walking it, the ↻ buttons in this panel are disabled — and each
one says which pass is holding the bank and how far it has got, for example
*✨ Score pass is running on this bank — 137 / 412*. Wait for it to land, or
press **Stop** in the ⏳ progress bar at the top of the bank; the buttons come
back by themselves the moment the bank is free.

When a re-run does start, the button reports what the pass produced right where
you pressed it: **`Done — 12 duplicate groups · 34 images (was 9 · 26)`**. If
your new value groups exactly the same images it says so — *unchanged* — rather
than leaving you unable to tell a no-op from a pass that never ran.

## Rotate a sideways image

Scraped folders and phone exports are full of shots lying on their side. Both
places you meet an image can turn it a quarter turn, and neither charges you for
it. (Asked for by 1Tomber, GitHub issue #17.)

**In a dataset**, open the image (click its tile) and use **↺ Rotate left** /
**↻ Rotate right** in the bar under the picture, next to ⇄ Mirror. The file
keeps its name, its caption, its status and its format — a PNG stays a PNG, a
WEBP stays a WEBP. Four turns bring you back to exactly where you started:
measured on the shipped encoder, a PNG and a WEBP come back **byte-identical**
after going all the way round, so a mis-click costs nothing. The one exception
is a JPEG, which the format itself forces to be re-encoded on every save: at the
quality LDS writes (95, no chroma subsampling) that is around 46 dB PSNR — far
below anything visible, and it barely grows with more turns — but it is not
free, so it is worth knowing. Datasets normally hold WEBP, so this mostly
concerns files restored from an old backup.

Rotation is deliberately **not** part of ✂ Crop, even though that is where you
might look for it first. Cropping **resamples** the image — it rescales the box
you drew to a 1024 px long side — and resampling costs detail no matter how
carefully the result is then saved. A quarter turn resamples nothing at all: it
just moves existing pixels to new coordinates. Sending it through the crop lane
would make it pay a price it does not owe.

**In a bank**, your own folder is never written to — so a bank rotation does not
touch your files at all. The turn is remembered against the image and applied to
what the app shows you and to what it copies when you **⬆ Promote**; your
original keeps its exact bytes, whatever you do. Select the images and use
**↺ Rotate left** / **↻ Rotate right** in the selection bar to fix a whole
sideways batch at once, or turn one image without leaving **▶ Review** with the
↺ / ↻ buttons (keyboard: `[` and `]`). Rotating in Review never decides
anything — the image stays under your cursor so you can judge it once it is the
right way up.

One caveat worth stating: the analysis passes (👤 Subject, ✨ Score, 📐 Framing)
still read the original file, so turning an image does **not** re-run them. Turn
first, then run the passes if you want them to see it upright.

## Clean the watermarks a bank found

**🚩 Find watermarks** flags the images carrying an overlaid logo, URL or
@username. Removing them used to mean promoting the watermark into a dataset
first and cleaning it there; the bank now does it itself, in **two steps you
launch by hand** — cheapest and safest first:

1. **✂ Auto-crop** cuts off the marks sitting in a border strip. No model, no
   GPU, no invented pixel: it simply trims the band up to the mark, and only
   when the image stays big enough to train on. Anything it can't crop that way
   is left flagged, on purpose.
2. **🧽 Inpaint** repaints what's left. **LaMa** (fast, non-generative) handles
   small off-centre marks and leaves marks *on the subject* flagged; **Klein**
   (slower, via ComfyUI) also clears those. Each engine says what to install
   when it isn't ready, and the button stays off rather than failing mid-pass.

Each step shows how many images it still has to work on and how many it has
already handled, so you can see where the funnel stands. **Your source files are
never modified** — a cleaned image is a copy the app keeps beside the bank's
thumbnails. That copy is what the grid shows, and what **⬆ Promote** sends to
the dataset, so a cleaned bank produces a clean dataset. **↩ Undo cleaning**
just deletes those copies and flags the images again, and **👁 Before / after**
flips a sample between the cleaned version and your untouched original.

If a bank was scanned by an older version, its flagged images carry no recorded
mark position; the panel says so and one more **🚩 Find watermarks** run makes
them cleanable.


## Fix a watermark mask in a bank

The detector draws **one** box, and it is a guess: it can miss a second logo,
swallow half the face, or land beside the mark. Open **▶ Review**, walk to a
flagged image and press **🚩 Edit mask** (shortcut `M`) — the same zone editor
the datasets use, on the bank image, right there.

- **+ Add zone**, then drag on the photo to draw a rectangle over the mark. Up
  to 32 zones; drag a zone to move it, its corners to resize.
- **Delete zone** removes the selected one, **Reset to detected** throws your
  zones away and puts the detector's box back.
- Every edit saves as you draw. If a save fails it says so and offers a retry —
  the zones on screen are never silently unsaved.

What the two cleaning steps then do with your mask:

- **🧽 Inpaint repaints exactly the zones you drew** — all of them, including a
  zone sitting on the subject, which is precisely what a hand mask is for.
- **✂ Auto-crop skips a hand-masked image.** A crop can only cut one border
  band; it cannot express several zones or a mark on the subject, so cropping
  the old box would remove pixels you did not point at.
- **An empty mask cleans nothing.** Delete every zone and you have said "there
  is nothing to repaint here": neither step touches that image, and the panel
  says how many are in that state instead of leaving them looking unhandled.

A flagged image an older scan left *without* a box becomes cleanable as soon as
you draw the zones yourself — that drawing is the missing information. And as
everywhere else in a bank, **your own file is never modified**: cleaning writes
a separate copy. A rotated image is shown unrotated here, because the whole
watermark lane works on your original file, which the ↻ turn never changed.


## A bank and a dataset never share files

A dataset and an image bank can hand images to each other in both directions,
and both directions **copy**. That is not an implementation detail — it is the
rule the whole flow rests on:

- **Bank → dataset** (**⬆ Promote…**) writes new files into the dataset.
- **Dataset → bank** (**🗃 Import to bank**, on the dataset) copies the dataset's
  kept images into a folder of the bank's own. Both choices retain the
  Dataset-owned captions, keep/reject curation, framing, watermark and
  provenance. Its dialog defaults to **Reuse compatible final-file analysis**;
  **Start fresh analysis** skips only reuse of prior analysis, not that metadata.
  The AI **Face** and **Score** results are not reused after normalization or
  another transformation because they are no longer proved.

Neither ever *points* at the other's files. The reason is that the two containers
have opposite contracts. A dataset **owns** its images; a bank merely **points**
at a live folder it does not own — which is exactly why 🗑 **Delete rejected** is
allowed to remove files from it. Put a bank on a dataset's folder and that button
stops deleting your rejects and starts deleting the dataset's training images.

So the app refuses it. If you paste a dataset's image folder into **➕ Create
bank** — or into **📦 Move folder…** for an existing bank — you get a refusal
that names the dataset and points you at **🗃 Import to bank** instead. The check
looks through the disguises: a subfolder of the dataset, the folder *containing*
all datasets, a different letter case, forward slashes instead of backslashes,
and symlinks or Windows junctions that resolve to the same place.

**If you already have such a bank** (it was possible before this check existed),
nothing is repaired or deleted behind your back. Opening it shows a red banner
naming the dataset, and 🗑 Delete rejected is refused on that bank — everything
else keeps working, so you can finish triaging. When you are ready, either
**📦 Move folder…** to point the bank at a folder of its own, or remove the bank
(removing a bank never touches files).

The dataset's own folder is shown at the top of the dataset, with a **⧉ Copy**
button, so you never have to go hunting for it in a file manager — which is how
this trap was found in the first place.


## Move a bank folder to another disk

A bank points at a folder *in place*, but nothing it computes lives in that
folder: the quality scores, duplicate groups, face clusters, captions and every
keep/reject decision are stored against the image row, and each row remembers
its file *relative* to the bank's folder. So moving a 30 000-image bank to
another drive costs nothing — you just have to tell the app where it went.

You can do this in either order. **📦 Move folder…** sits in the bank's header
next to its path (and **📦** on the bank's card in the list), so you can open it
before touching anything to see what the app will ask for; it also appears inside
the warning shown once the app notices the folder is gone, if you moved first.
Paste or browse to the new folder
and press **🔍 Check this folder**. Nothing is written yet: the app walks the
candidate folder and tells you how many of *this bank's* images are in there and
how many are not. Paste it however you like — Windows' *Copy as path* wraps the
path in quotes, and a trailing `\` or forward slashes are equally fine; the field
then shows the folder the app actually resolved, so what you confirm is what it
will use.

- **All of them found** → confirm, and the bank is repointed with every score
  and decision intact.
- **Some found, some missing** → you can still confirm. Nothing is deleted:
  rows whose file didn't come along keep their analysis and simply read as
  missing until the file comes back.
- **None found** → refused. That folder is a *different* folder, not a moved
  one — the usual cause is picking the parent of the folder you moved.

The app never deletes a row on its own, and an analysis pass run while the files
are away no longer degrades them either: a file that is *absent* is not a file
that is *broken*, so the pass stops and tells you the folder appears to have
moved instead of marking thousands of images unusable.


## Make Score use a GPU Python you already have

The **✨ Score** pass (aesthetic · NSFW · style) runs in its own small Python
environment, and that environment deliberately carries **CPU-only PyTorch**: a
first install stays a few hundred megabytes instead of pulling ~2.5 GB of CUDA
wheels onto machines that may have no card at all.

On a machine that *does* have one, that default is expensive — CLIP measures
about **336 ms per image on the CPU against ~15 ms on a recent card**, so a
30 000-image bank is the difference between a coffee break and most of an
afternoon. The bank says so: when Score is about to run on the CPU on a machine
with an NVIDIA card, an amber note gives you the estimate and a button, **⚡ Use
a GPU Python I already have**.

That button is the point. If you train LoRAs or run ComfyUI, this machine
*already* has a PyTorch with working CUDA. Score can simply borrow it — no
download, no third environment to maintain.

The dialog lists the interpreters the app knows about (the environment it built
for scoring, ai-toolkit's, ComfyUI's, its own) and reports each one **package by
package**:

- **GPU ready** — everything the pass imports is there *and* PyTorch sees the
  card. Pick it and the next Score run is minutes instead of hours.
- **Missing packages** — the reason is named. The common one is an interpreter
  with a perfect CUDA PyTorch but no **OpenCLIP**: Score needs `open_clip` and
  `transformers`/`timm` too, so CUDA alone is not enough. Such an interpreter is
  **refused**, on purpose — accepting it would trade slow-but-working scoring for
  an import error an hour into the pass.
- **CPU only** — it can run the pass, it just has no usable CUDA.
- **No answer** — the path is not a working interpreter (moved venv, unplugged
  drive). Nothing changes.

**The app never installs anything into an environment it did not create.** Your
ai-toolkit venv runs your training and ComfyUI's runs your generation; a silent
`pip install` into either is not something a dataset tool gets to do. When a
package is missing the dialog shows you the exact command and leaves the choice
to you — run it in a terminal, then hit **↻ Check again** and the row updates.

**Not listed? That field is not a fallback.** Most machines have neither
ai-toolkit nor ComfyUI where the app looks — or at all — so entering a path
yourself is a first-class route, checked exactly the same way. Paste an
interpreter *or* the environment folder that contains it: a venv, a conda or
miniconda env, a uv venv, a portable bundle, the system Python, something on a
second disk. Spaces, accents and quotes around the path are fine ("Copy as path"
on Windows wraps it in quotes; that is handled). The layout is never assumed —
the app knocks on the shapes an environment can have and keeps whichever one
actually answers.

No version of PyTorch or CUDA is required. The only question asked is the one
that matters: do the packages import, and does PyTorch see a card. An old card
on cu118, a 50-series that only works on cu128, a nightly build — all fine.

**No NVIDIA card?** Then there is nothing to fix, and the app says so plainly
instead of suggesting a CUDA install you could not use. Borrowing an interpreter
is still offered, for one honest reason: if another Python here already has the
packages, you can skip installing them a second time. It will not be faster.

**Back to the app default** puts everything back exactly as it was. The choice is
reversible at any time, and the note under the passes always says which
interpreter is in use. If you never open this dialog, nothing changes: an install
that works today keeps working, untouched.

## The LoRA Canvas (every run on one board)

**Canvas** in the top bar opens a single board holding the training history of
every dataset you have. Each dataset gets a lane; inside a lane, each run is a
card and each save it wrote is a small pill underneath it. When a run continued
from an earlier one, the line between them starts at the *exact* checkpoint it
resumed from — so "where did this LoRA come from" is a thing you read, not a
thing you reconstruct.

**Choosing what is on the board.** Everything is on it by default. The
**Datasets** control above the board unticks what you do not want to see; the
choice is remembered. On a phone it opens folded, with the current state written
on the button ("3 of 7") so you always know what you are looking at.

**Moving around.** Drag the background to pan, use the wheel (or two fingers) to
zoom, and **Fit** puts the whole board back in view. The board only fits itself
automatically until you first touch it — after that a dataset finishing its load
never yanks your view away.

**Reading a run.** Click a run card to open **everything that run produced**:
its images grouped by the checkpoint that made them, most-trained step first, so
you can see where the LoRA stopped getting better without opening one pill at a
time. Underneath the images are the run's note, its per-checkpoint notes, and the
settings it trained with. **ⓘ Full details** opens the drawer where those notes
can be edited.

A run with many checkpoints opens with its three most-trained steps expanded and
the rest folded behind their image counts — tap a step to unfold it. When a run
holds more images than one panel should carry, the panel says so rather than
looking complete; the missing ones are still reachable from each checkpoint's own
pill and in the Test Studio.

Sometimes a step reads **Step unknown**. Those are older test images whose file
name identifies the run but not the checkpoint inside it, so they belong to the
run and to no pill. Images that identify nothing at all are still counted in the
footnote at the bottom of the panel — they live in the Test Studio.

**Shift-click two** run cards to compare their settings side by side, with the
differences highlighted — and because every dataset is on the same board, those
two runs no longer have to belong to the same dataset. Dragging a card to
rearrange the board never opens the panel.

**Arranging the board.** Drag a run card and it stays where you put it, across
reloads. On a phone, moving a card and scrolling the board are the same gesture,
so a card is picked up with a **long press** — rest your finger on it for a
moment and it lifts; a finger that slides straight away scrolls as usual.

Once you have moved anything in a lane, that whole lane stops rearranging itself:
a training run that finishes later lands in free space next to your layout
instead of pushing everything sideways, which is what would otherwise happen —
the automatic tree centres each run over its continuations, so one new branch
re-flows the lane around it. Lanes you have never touched keep following the
automatic tree, because there is no arrangement to protect there.

**✦ Tidy up** is the way back: it forgets every card you have moved on the lanes
currently shown and rebuilds the automatic tree. Positions are only ever a
display preference — moving a card never changes which run continued which, and
Tidy up never deletes a run, a checkpoint or a note.

**Generating from the board.** Every checkpoint pill carries a small **✓** box.
Tick one and the run settings open beside the board: the prompt, the seed, the
format, the steps, the engine settings — the Test Studio's own panel, not a
lookalike, so anything the Test Studio can do the board can do too.

What the board adds is that your picks do not have to belong to the same
dataset. Tick a checkpoint in one lane and two in another and they run together
on one shared prompt and one shared seed, which is the only honest way to
compare LoRAs against each other.

Two things it will tell you rather than fail at:

- **A checkpoint that is not in ComfyUI yet** is still pickable. The button then
  says what it is about to do — *"Deploy 2 checkpoints, then generate"* — and
  waits for you. Nothing is copied into your ComfyUI folder by a button that did
  not announce it, and if a copy fails, nothing generates: half a comparison
  answers a different question than the one you asked.
- **Two different families in one selection** (say Krea and Z-Image) is refused,
  and it says which two. This is not a restriction we chose: those families do
  not share a base model or a workflow, so there is no single run that can render
  both. Unpick one family and the button comes back.

**▶ Continue training from a checkpoint.** Clicking a pill's body opens its
actions — Download, Deploy, Details, Delete — and **▶ Continue from here**. It
opens the *same* launch dialog the Checkpoints panel and the Runs page open, on
*that exact save*: pick where it runs (**💻 Local** or **☁ Cloud**), how many
extra steps, and — folded under *Adjust settings* — the checkpoint cadence, the
preview prompts, the timestep weighting and the learning rate. Rank, base and
optimizer are locked to the checkpoint being continued; they are not things a
resume can change.

The dialog also names **what “resume” means**; it never silently guesses:

- **Full training state** is offered only for a local checkpoint carrying a
  complete, hash-verified state bundle. It restores the raw adapter parameters,
  optimizer, scheduler, scaler, EMA, Python/NumPy/Torch/CUDA random generators,
  dataloader order and cursor, bucket/crop geometry, the exact latent/text-cache
  bytes, and the exact next step. Exported image, caption and mask contents,
  dataset topology, base, network shape, training recipe, ai-toolkit revision,
  GPU identity and the complete installed Python-package map must still match.
  In this mode only the preview prompts can change. Save/preview cadence,
  learning rate and timestep weighting stay locked because changing any of them
  would change the trajectory the state belongs to.
- **LoRA weights only** is the explicit fallback and is available for legacy
  checkpoints. The chosen `.safetensors` is copied into a clean run folder;
  optimizer, scheduler, scaler, RNG and dataloader progress restart. The source
  run is renamed aside, not deleted, so all its saves remain recoverable.

Each checkpoint says why full state is unavailable when its bundle is missing,
incomplete, corrupt or incompatible. Cloud continuation currently supports
weights only; choosing full state there is refused before a pod is created.
State bundles are published atomically and the newest two are retained alongside
the public checkpoints, so a crash during capture cannot masquerade as a usable
exact save.

One deliberately conservative boundary remains: low-level Torch/CUDA backend
flags changed externally after LDS performs its runtime preflight are not yet
part of the compatibility fingerprint. Do not change deterministic/TF32/cuDNN
flags between the original process and an exact continuation.

Read the step field as **extra** steps, not a total: the line beside it spells
out where you land ("→ target step 3500") and so does the button. Resuming step
2500 of a run that ended at 3500 is the whole point of opening this from a pill
— a later epoch can be over-cooked, and the earlier one is often the better
LoRA.

The board is the only screen that offers **both lanes for both kinds of run**: a
checkpoint is just a file, so a run trained on your machine can be finished on a
rented GPU, and a cloud run's epoch can be finished here. What is *not* always
possible is stated rather than hidden — a lane you cannot use stays visible,
greyed, with its reason:

- *"Local training needs ai-toolkit"* / *"A training is already running on this
  machine"* — local training is single-flight for the whole machine.
- *"Cloud training needs a vast.ai API key"*, an active cloud run on the same
  dataset, or the concurrency limit.
- *"This save is no longer on this machine"* — the lanes that need the file are
  closed. For a **cloud** run the cloud lane still works, because a fresh pod is
  seeded from that run's own staging rather than from your disk. For a local run
  there is no copy anywhere, and both lanes say so.

If the save vanished between the board being drawn and the click, the launch is
refused with the steps that *are* available, named — never a silent failure.

**The gallery under a checkpoint.** Images pile up. A checkpoint that has
produced more than one shows a small **× N** badge; clicking it opens everything
that checkpoint ever made, newest first — from the board, from the Test Studio,
from a comparison run, it does not matter. Regenerating no longer replaces what
was there.

Which image belongs to which checkpoint is recorded when the image is generated.
Images made before that was recorded are matched back where the evidence allows
it (the run tag the deploy stamps into the LoRA's name); those that cannot be
traced are **counted and left out** rather than shown under a checkpoint they
might not belong to. The gallery says how many those are — they are still in the
Test Studio, they simply have no node to sit under.

**What a generated image was made with.** Open any image from a gallery and the
full-screen view lays its record out beside it: the three facts you look for
first (**step**, **seed**, **LoRA strength**) as chips, then the settings that
actually decided the picture — sampler, scheduler, CFG, sampling steps, the base
model, the LoRA file, any always-on LoRAs, the format, the face-similarity score
— and the prompt last. The prompt folds when it is long instead of pushing
everything else off the screen, and both the **seed** and the **prompt** copy in
one click. A run that predates a given setting simply shows no row for it: an
absent line is honest, a dash is not.

**📌 Pinning an image onto the board.** Comparing two checkpoints means looking
at their pictures *at the same time*, which a full-screen viewer cannot do. So
**📌** drops an image onto the board as a node of its own, joined to the
checkpoint that produced it by the same connector the board uses for "this run
continued from that checkpoint".

There are two ways in, and the first one is the one to remember: **every
thumbnail in a run or checkpoint gallery carries a 📌 in its bottom-right
corner** — one tap, no need to open the image at all. It is hidden while you are
in **Select** mode (that mode is for arming a delete, and a second target there
is a mis-tap waiting to happen). The same action is also in the full-screen
viewer, spelled out as **📌 Pin to canvas**, for when you have already opened a
picture and decide it belongs on the board.

- **Move it** by dragging (on a phone: a long press picks it up, exactly like a
  run card). **Resize it** from the corner handle. **Close it** with **✕**.
- Closing forgets nothing. Pin the same image again and it comes back **exactly
  where you left it, at exactly the size you left it** — that is the point of the
  feature, not a side effect. The geometry lives with your card positions, on
  your machine's LoRA Dataset Studio rather than in one browser, so it follows
  the dataset.
- **Keyboard:** focus a pinned image (Tab), then the arrow keys move it,
  Shift+arrows move it faster, **+** / **−** resize it and **Esc** closes it.
- If the image is later **deleted**, its node quietly leaves the board — a node
  showing a picture that no longer exists would be worse than no node. If the
  *checkpoint* is gone but the image is not, the picture stays and simply loses
  its connecting line.
- Unticking a dataset takes its lane off the board, pinned images included; they
  come back with the lane, untouched.
- **✦ Tidy up** does not throw pinned images away — it re-flows them into the
  same tidy band **📌 Pin all** uses, so a rebuild of the automatic tree can no
  longer park a picture on top of a run card.
- The **✕**, the **🔍** and the resize corner keep a finger-sized target **at
  every zoom level**: they are drawn at a constant size on screen rather than at
  the board's, so a board fitted to twenty runs is still one you can tap.

**🖼🖼 Fuse pinned images side by side.** Comparing two renders across a gap and
two frames is comparing two frames. **Drop one pinned image onto another and
they become a single node**, pictures edge to edge with nothing drawn between
them. There is **no limit**: drop a third, a tenth, they all join the strip.

- **Where it lands.** While you drag, the picture you are about to join lights up
  with a dashed outline, a bar marks the exact slot yours would take, and a label
  says how many pictures the group would then hold. Let go anywhere else and it
  is an ordinary move — nothing fuses by surprise.
- **Which side.** Drop on the left half of a picture to land before it, on the
  right half to land after it. The same gesture **re-orders** a group: drag a
  member out and back onto the slot you want.
- **Move the whole group** by its **title bar** (`⠿ N images`), which is also
  where its **✕** lives. That bar is the only thing that moves a group, on
  purpose: dragging a *picture* inside a group means something else entirely.
- **Take one back out** by dragging it **off the group**. That is the whole rule
  — while it is still over the strip nothing has happened, and letting go there
  puts it back. Once it is clear of the strip it becomes a node of its own again,
  **at the size it had before it joined**, wherever you dropped it. Joining a
  group never rewrites a picture's own size; the strip only borrows it.
- **The pictures that stay do not move.** Take the first one out and the strip
  keeps its place and its height; the rest simply close the gap. A group left
  with a single picture stops being a group.
- **Which ✕ am I about to press?** At rest a group is nothing but photographs.
  Hover (or Tab to) one and *that* picture lights up and shows its own step
  label, its 🔍 and its ✕ — the group's own ✕ is the one on the title bar, and it
  carries the count (`✕3`) precisely so the two can never be confused. Closing a
  group closes all of its pictures, undoes the group, and each one keeps its own
  remembered size; re-pinning one from its gallery brings back **that one**, not
  the strip.
- **Every picture in a strip is the same height**, each scaled to keep its own
  shape — that is what makes the band continuous instead of a row of letterboxed
  tiles. Resize the group from its corner and the whole strip scales.
- **A strip has no width limit, and that is the honest consequence of "no
  limit".** Ten pictures side by side is ten times as wide as one; the board
  zooms and pans, so **✦ Fit** is the answer. It deliberately does *not* wrap
  onto a second row — a strip that quietly stopped being a strip at some
  invisible threshold would be worse than a wide one. On a phone, expect to zoom.
- **✦ Tidy up leaves groups alone.** It rebuilds the automatic tree and re-flows
  the pictures you have *not* grouped; a strip is something you assembled on
  purpose, and taking it apart is not tidying. The way out is the group's ✕, or
  dragging its pictures back off it.

**📌 Pin all — the whole lot in one gesture.** When a generation launched from
the board finishes, the green bar says how many images are ready and names the
checkpoints they joined. **📌 Pin all N to the board** puts every one of them on
the board without opening a single gallery.

- **Where they land.** In a band under the lane, **one column per checkpoint**,
  each column under the checkpoint that produced it — so a lot spanning four runs
  reads as four groups, and each picture still draws its own line back to its
  pill. The band starts below everything already on the lane, which is what makes
  the guarantee a real one: **nothing is ever placed on top of a run card, a
  checkpoint pill or a picture you positioned yourself.**
- **Big lots become a contact sheet.** A pair of renders lands full size; twenty
  or thirty land as thumbnails, which is the size you actually compare that many
  pictures at. Each one is still resizable afterwards like any other node.
- **What is already on the board is left alone.** An image you have already
  pinned is neither moved nor duplicated, and the button counts only what is
  left — once everything is up, the button is simply not there any more. An
  image you *closed* is offered again, and comes back where you closed it when
  that spot is free.
- **Nothing is stacked in silence.** One click places at most 40 pictures; if the
  run made more, the bar says how many were left out and where to get them
  (their checkpoint gallery). The count of what was actually pinned is announced
  for screen readers too.
- **↩ Undo** takes exactly the images that click added straight back off the
  board, and nothing else.

**Which checkpoints you can generate from, at a glance.** Every checkpoint pill
carries its deployment state on its **left edge**: a **solid sky bar** means the
checkpoint is deployed to ComfyUI and can be generated from right now; a **dashed
grey bar** means the file is on your disk but not deployed yet. Not deployed does
*not* mean missing — the save is there, it simply has no copy in ComfyUI, and
ticking it before **🎨 Generate** makes the launch deploy it for you. The shape
(solid versus dashed) carries as much of the message as the colour does, a legend
sits above the board, and hovering a pill spells it out in words.

The graph embedded in a dataset's *Checkpoints & LoRAs* panel is unchanged and
still holds the per-checkpoint actions (download, deploy, continue from here,
inline previews). The canvas is a second way in, not a replacement.

## Tips that save runs

- Trust the composition meter over your instinct — a set that "looks varied"
  is usually still face-heavy.
- Fix every leak the badge reports before training; one "a woman with long
  blonde hair" caption quietly competes with your trigger.
- Don't chase steps. Train the auto count, then let the Test Studio find the
  *earliest* checkpoint that nails the identity — it keeps the most prompt
  flexibility.
- The next chapter — **Building a good dataset** — explains *why* behind every
  rule above. Read it once before your first serious run.
