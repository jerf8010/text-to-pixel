import { derive } from '../core/generator.js';
import { h, replaceChildren } from './dom.js';
import { drawPixels, pixelsToBlob } from './pixel-render.js';

const DOWNLOAD_SCALE = 16; // a 16x16 image becomes a 256x256 PNG
const CANVAS_SCALE = 16;
const WATCHED = ['status', 'result', 'error', 'selectedModelIds', 'nVariants', 'vocabulary', 'models'];

function errorTitle(code) {
  switch (code) {
    case 'network':
      return 'Cannot reach the server';
    case 'unknown_words':
    case 'prompt_too_long':
    case 'empty_prompt':
      return 'Check your description';
    case 'internal_error':
      return 'The model failed to generate an image';
    default:
      return 'Something went wrong';
  }
}

/** Shows the generated images, one card per model, side by side. */
export function mountResultsPanel(store, actions) {
  const headExtra = document.getElementById('results-head-extra');
  const notice = document.getElementById('notice');
  const body = document.getElementById('results-body');
  const section = document.getElementById('results');

  let seedCopied = false;
  let last = {};

  async function copySeed(seed) {
    try {
      await navigator.clipboard.writeText(String(seed));
      seedCopied = true;
      actions.notify('Seed copied to the clipboard.');
      draw(store.get());
      setTimeout(() => {
        seedCopied = false;
        draw(store.get());
      }, 1500);
    } catch {
      actions.notify('Could not copy the seed. Select it and copy it manually.');
    }
  }

  async function download(modelId, seed, index, pixels, palette) {
    try {
      const blob = await pixelsToBlob(pixels, palette, DOWNLOAD_SCALE);
      const url = URL.createObjectURL(blob);
      const link = h('a', { href: url, download: `pixelgen-${modelId}-${seed}-${index + 1}.png` });
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000); // give the browser time to start the download
      actions.notify('Image downloaded.');
    } catch {
      actions.notify('Could not create the PNG.');
    }
  }

  function drawEmpty() {
    return h(
      'div',
      { class: 'empty' },
      h('div', { class: 'empty-art', 'aria-hidden': 'true' }),
      h('h3', {}, 'Nothing generated yet'),
      h(
        'p',
        { class: 'muted' },
        'Write a description or pick an example, then press ',
        h('strong', {}, 'Generate'),
        '. With both models selected you will see their drawings side by side.',
      ),
    );
  }

  function drawLoading(state) {
    const slots = Array.from({ length: state.nVariants }, () => h('div', { class: 'skeleton' }));
    return [
      h('p', { class: 'sr-only', role: 'status' }, 'Generating images…'),
      h(
        'div',
        { class: 'columns', 'aria-hidden': 'true' },
        state.selectedModelIds.map((id) =>
          h(
            'article',
            { class: 'card' },
            h('header', {}, h('h3', {}, state.models.find((m) => m.id === id)?.label ?? id)),
            h('div', { class: state.nVariants === 1 ? 'images single' : 'images' }, slots.map((s) => s.cloneNode())),
          ),
        ),
      ),
    ];
  }

  function drawError(state) {
    const error = state.error;
    return h(
      'div',
      { class: 'banner', role: 'alert' },
      h('h3', {}, errorTitle(error.code)),
      h('p', {}, error.message),
      h(
        'button',
        { type: 'button', class: 'secondary', disabled: !derive(state).canGenerate, onclick: () => actions.generate() },
        'Try again',
      ),
    );
  }

  function drawResult(state) {
    const result = state.result;
    const palette = state.vocabulary.palette;
    return [
      h('p', { class: 'prompt' }, `“${result.prompt}”`),
      h(
        'div',
        { class: 'columns' },
        result.results.map((model) =>
          h(
            'article',
            { class: 'card', 'aria-labelledby': `model-${model.id}` },
            h(
              'header',
              {},
              h('h3', { id: `model-${model.id}` }, model.label),
              h('span', { class: 'time' }, `${model.elapsed_ms.toFixed(0)} ms`),
            ),
            h(
              'div',
              { class: model.images.length === 1 ? 'images single' : 'images' },
              model.images.map((image, i) => {
                const canvas = h('canvas', {
                  class: 'pixel',
                  role: 'img',
                  'aria-label': `${model.label}, variant ${i + 1} for: ${result.prompt}`,
                });
                drawPixels(canvas, image.pixels, palette, CANVAS_SCALE);
                return h(
                  'figure',
                  {},
                  canvas,
                  h(
                    'figcaption',
                    {},
                    h(
                      'button',
                      {
                        type: 'button',
                        class: 'link',
                        onclick: () => download(model.id, result.seed, i, image.pixels, palette),
                      },
                      'Download PNG',
                    ),
                  ),
                );
              }),
            ),
          ),
        ),
      ),
    ];
  }

  function drawHeadExtra(state) {
    if (state.status !== 'done' || !state.result) {
      replaceChildren(headExtra);
      return;
    }
    const seed = state.result.seed;
    replaceChildren(
      headExtra,
      h('span', { class: 'muted' }, 'seed'),
      h('code', {}, String(seed)),
      h('button', { type: 'button', class: 'link', onclick: () => copySeed(seed) }, seedCopied ? 'Copied' : 'Copy'),
      h('button', { type: 'button', class: 'link', onclick: () => actions.useSeed(seed) }, 'Reuse'),
    );
  }

  function draw(state) {
    section.setAttribute('aria-busy', String(state.status === 'loading'));
    drawHeadExtra(state);
    switch (state.status) {
      case 'loading':
        replaceChildren(body, drawLoading(state));
        break;
      case 'error':
        replaceChildren(body, drawError(state));
        break;
      case 'done':
        replaceChildren(body, drawResult(state));
        break;
      default:
        replaceChildren(body, drawEmpty());
    }
  }

  function render(state) {
    notice.textContent = state.notice ?? '';
    // Typing in the prompt must not rebuild the results (and their canvases).
    if (WATCHED.every((key) => state[key] === last[key])) return;
    last = Object.fromEntries(WATCHED.map((key) => [key, state[key]]));
    draw(state);
  }

  store.subscribe(render);
  render(store.get());
}
