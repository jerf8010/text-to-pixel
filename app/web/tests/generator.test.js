import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { createActions, derive, initialState } from '../js/core/generator.js';
import { createStore } from '../js/core/store.js';

const MODELS = [
  { id: 'decoder-only', label: 'Decoder-only', kind: 'decoder_only', n_params: 1, description: '' },
  { id: 'encoder-decoder', label: 'Encoder-decoder', kind: 'encoder_decoder', n_params: 2, description: '' },
];
const VOCABULARY = {
  words: ['a', 'small', 'large', 'red', 'blue', 'circle', 'square', 'left', 'of'],
  groups: {},
  max_words: 10,
  palette: [[0, 0, 0]],
  examples: [],
};
const RESULT = { prompt: 'a small red circle', seed: 7, results: [] };

/** A fake API; `generate` records its calls and can be made to fail or hang. */
function fakeApi(overrides = {}) {
  const calls = [];
  return {
    calls,
    getModels: async () => MODELS,
    getVocabulary: async () => VOCABULARY,
    generate: async (payload, signal) => {
      calls.push({ payload, signal });
      return RESULT;
    },
    ...overrides,
  };
}

function memoryStorage(initial = {}) {
  const data = { ...initial };
  return {
    data,
    getItem: (key) => (key in data ? data[key] : null),
    setItem: (key, value) => {
      data[key] = value;
    },
  };
}

async function loaded({ api = fakeApi(), storage = memoryStorage() } = {}) {
  const store = createStore(initialState);
  const actions = createActions({ store, api, storage });
  await actions.load();
  return { store, actions, api, storage };
}

describe('load', () => {
  it('loads the catalogue and selects every model by default', async () => {
    const { store } = await loaded();
    const s = store.get();
    assert.equal(s.bootState, 'ready');
    assert.deepEqual(s.selectedModelIds, ['decoder-only', 'encoder-decoder']);
  });

  it('reports a failure so the UI can offer a retry', async () => {
    const api = fakeApi({
      getModels: async () => {
        throw new Error('Cannot reach the server.');
      },
    });
    const { store } = await loaded({ api });
    assert.equal(store.get().bootState, 'error');
    assert.equal(store.get().bootError, 'Cannot reach the server.');
  });

  it('restores saved settings and ignores unknown models', async () => {
    const saved = JSON.stringify({ prompt: 'a red circle', models: ['encoder-decoder', 'gone'], nVariants: 3, temperature: 0.5 });
    const { store } = await loaded({ storage: memoryStorage({ 'pixelgen.settings.v1': saved }) });
    const s = store.get();
    assert.equal(s.prompt, 'a red circle');
    assert.deepEqual(s.selectedModelIds, ['encoder-decoder']);
    assert.equal(s.nVariants, 3);
    assert.equal(s.temperature, 0.5);
  });

  it('starts from defaults when the saved data is corrupted', async () => {
    const { store } = await loaded({ storage: memoryStorage({ 'pixelgen.settings.v1': '{not json' }) });
    assert.equal(store.get().prompt, '');
    assert.equal(store.get().selectedModelIds.length, 2);
  });
});

describe('derive', () => {
  it('explains why Generate is blocked', async () => {
    const { store, actions } = await loaded();
    assert.equal(derive(store.get()).blockedReason, 'Write a description to continue.');

    actions.setPrompt('a small red cirle');
    assert.match(derive(store.get()).blockedReason, /Not in the vocabulary/);

    actions.setPrompt('a small red circle');
    assert.equal(derive(store.get()).canGenerate, true);

    actions.toggleModel('decoder-only');
    actions.toggleModel('encoder-decoder');
    assert.equal(derive(store.get()).blockedReason, 'Select at least one model.');
  });

  it('validates the seed', async () => {
    const { store, actions } = await loaded();
    actions.setPrompt('a small red circle');
    actions.setSeedText('12');
    assert.equal(derive(store.get()).seed, 12);
    actions.setSeedText('abc');
    assert.equal(derive(store.get()).seedValid, false);
    assert.equal(derive(store.get()).canGenerate, false);
    actions.setSeedText('');
    assert.equal(derive(store.get()).seed, null);
  });
});

describe('prompt editing', () => {
  it('appends words and replaces misspelt ones', async () => {
    const { store, actions } = await loaded();
    actions.appendWord('a');
    actions.appendWord('small');
    assert.equal(store.get().prompt, 'a small');
    actions.setPrompt('a small cirle cirle');
    actions.replaceWord('cirle', 'circle');
    assert.equal(store.get().prompt, 'a small circle circle');
  });

  it('keeps model order when toggling', async () => {
    const { store, actions } = await loaded();
    actions.toggleModel('decoder-only');
    actions.toggleModel('decoder-only');
    assert.deepEqual(store.get().selectedModelIds, ['decoder-only', 'encoder-decoder']);
  });
});

describe('generate', () => {
  it('sends the canonical prompt and stores the result', async () => {
    const { store, actions, api } = await loaded();
    actions.setPrompt('  A Small RED circle. ');
    actions.setSeedText('7');
    await actions.generate();
    assert.deepEqual(api.calls[0].payload, {
      prompt: 'a small red circle',
      models: ['decoder-only', 'encoder-decoder'],
      n_variants: 1,
      temperature: 1.0,
      seed: 7,
    });
    assert.equal(store.get().status, 'done');
    assert.equal(store.get().result, RESULT);
  });

  it('does nothing while the prompt is invalid', async () => {
    const { store, actions, api } = await loaded();
    actions.setPrompt('a green hexagon');
    await actions.generate();
    assert.equal(api.calls.length, 0);
    assert.equal(store.get().status, 'idle');
  });

  it('keeps the error so the UI can show it', async () => {
    const failure = Object.assign(new Error('boom'), { code: 'internal_error' });
    const { store, actions } = await loaded({
      api: fakeApi({
        generate: async () => {
          throw failure;
        },
      }),
    });
    actions.setPrompt('a small red circle');
    await actions.generate();
    assert.equal(store.get().status, 'error');
    assert.equal(store.get().error, failure);
  });

  it('ignores the answer of a request that was superseded', async () => {
    let release;
    const slow = new Promise((resolve) => {
      release = resolve;
    });
    let n = 0;
    const api = fakeApi({
      generate: async () => {
        n += 1;
        return n === 1 ? slow : { ...RESULT, seed: 2 };
      },
    });
    const { store, actions } = await loaded({ api });
    actions.setPrompt('a small red circle');

    const first = actions.generate();
    store.set({ status: 'idle' }); // simulate the user pressing Generate again
    await actions.generate();
    release({ ...RESULT, seed: 1 });
    await first;

    assert.equal(store.get().result.seed, 2);
  });
});

describe('save', () => {
  it('persists the settings worth remembering', async () => {
    const { actions, storage } = await loaded();
    actions.setPrompt('a red circle');
    actions.setVariants(2);
    actions.save();
    const saved = JSON.parse(storage.data['pixelgen.settings.v1']);
    assert.equal(saved.prompt, 'a red circle');
    assert.equal(saved.nVariants, 2);
  });
});
