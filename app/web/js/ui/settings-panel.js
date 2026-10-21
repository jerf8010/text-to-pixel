import { derive } from '../core/generator.js';
import { h, replaceChildren } from './dom.js';

function formatParams(count) {
  return count >= 1_000_000 ? `${(count / 1_000_000).toFixed(2)}M` : `${Math.round(count / 1000)}K`;
}

/** Step 2: choose which models to run, plus the advanced sampling options. */
export function mountSettingsPanel(store, actions) {
  const $ = (id) => document.getElementById(id);
  const modelList = $('models');
  const modelsNote = $('models-note');
  const variants = $('variants');
  const temperature = $('temperature');
  const temperatureValue = $('temperature-value');
  const seed = $('seed');

  variants.addEventListener('change', () => actions.setVariants(Number(variants.value)));
  temperature.addEventListener('input', () => actions.setTemperature(Number(temperature.value)));
  seed.addEventListener('input', () => actions.setSeedText(seed.value));

  let builtFor = null;
  const checkboxes = new Map();

  function buildModels(models) {
    checkboxes.clear();
    replaceChildren(
      modelList,
      h('legend', { class: 'sr-only' }, 'Models to run'),
      models.map((model) => {
        const input = h('input', { type: 'checkbox', onchange: () => actions.toggleModel(model.id) });
        checkboxes.set(model.id, input);
        return h(
          'label',
          { class: 'model' },
          input,
          h(
            'span',
            { class: 'model-text' },
            h('span', { class: 'model-name' }, model.label),
            h('span', { class: 'muted' }, model.description),
            h('span', { class: 'model-meta' }, `${formatParams(model.n_params)} parameters`),
          ),
        );
      }),
    );
  }

  function render(state) {
    if (builtFor !== state.models) {
      buildModels(state.models);
      builtFor = state.models;
    }
    for (const [id, input] of checkboxes) {
      const selected = state.selectedModelIds.includes(id);
      input.checked = selected;
      input.closest('.model').classList.toggle('selected', selected);
    }

    const count = state.selectedModelIds.length;
    modelsNote.className = count === 0 ? 'problem' : 'muted';
    modelsNote.setAttribute('role', count === 0 ? 'alert' : 'note');
    modelsNote.textContent =
      count === 0
        ? 'Select at least one model.'
        : count > 1
          ? 'Both models use the same seed, so you can compare them fairly.'
          : '';

    if (variants.value !== String(state.nVariants)) variants.value = String(state.nVariants);
    if (Number(temperature.value) !== state.temperature) temperature.value = String(state.temperature);
    temperatureValue.textContent = state.temperature.toFixed(1);
    if (seed.value !== state.seedText) seed.value = state.seedText;
    seed.setAttribute('aria-invalid', String(!derive(state).seedValid));
  }

  store.subscribe(render);
  render(store.get());
}
