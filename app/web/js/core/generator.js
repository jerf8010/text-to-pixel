/**
 * State and behaviour of the generator screen.
 *
 * The UI modules only read the state and call these actions; they never touch HTTP or
 * storage, which keeps this file testable without a browser.
 */
import { tokenizePrompt, validatePrompt } from './prompt-validation.js';

export const MAX_SEED = 2 ** 31 - 1;
const MAX_VARIANTS = 4;
const STORAGE_KEY = 'pixelgen.settings.v1';

export const initialState = {
  bootState: 'loading', // 'loading' | 'ready' | 'error'
  bootError: null,
  vocabulary: null,
  models: [],

  prompt: '',
  selectedModelIds: [],
  nVariants: 1,
  temperature: 1.0,
  seedText: '', // empty means "pick a random seed"
  exampleGroup: 0,

  status: 'idle', // 'idle' | 'loading' | 'done' | 'error'
  result: null,
  error: null,
  notice: null, // short message announced to screen readers
};

/** Values computed from the state: validation, seed and whether Generate is allowed. */
export function derive(state) {
  const vocabulary = state.vocabulary;
  const check = vocabulary
    ? validatePrompt(state.prompt, vocabulary.words, vocabulary.max_words)
    : null;

  const seedText = state.seedText.trim();
  const seedValid = seedText === '' || (/^\d+$/.test(seedText) && Number(seedText) <= MAX_SEED);
  const seed = seedText !== '' && seedValid ? Number(seedText) : null;

  let blockedReason = null;
  if (!check) blockedReason = 'The models are still loading.';
  else if (check.issue === 'empty') blockedReason = 'Write a description to continue.';
  else if (!check.valid) blockedReason = check.message;
  else if (state.selectedModelIds.length === 0) blockedReason = 'Select at least one model.';
  else if (!seedValid) blockedReason = `The seed must be a whole number between 0 and ${MAX_SEED}.`;

  return {
    check,
    seedValid,
    seed,
    blockedReason,
    canGenerate: blockedReason === null && state.status !== 'loading',
  };
}

/**
 * @param {{
 *   store: ReturnType<typeof import('./store.js').createStore>,
 *   api: { getModels(): Promise<any>, getVocabulary(): Promise<any>, generate(payload: any, signal?: AbortSignal): Promise<any> },
 *   storage?: { getItem(key: string): string | null, setItem(key: string, value: string): void },
 * }} deps
 */
export function createActions({ store, api, storage = null }) {
  let pending = null; // AbortController of the request in flight

  function readSaved() {
    try {
      const raw = storage?.getItem(STORAGE_KEY);
      if (!raw) return null;
      const data = JSON.parse(raw);
      const valid =
        typeof data?.prompt === 'string' &&
        Array.isArray(data.models) &&
        typeof data.nVariants === 'number' &&
        typeof data.temperature === 'number';
      return valid ? data : null;
    } catch {
      return null; // storage unavailable or corrupted: start from defaults
    }
  }

  function restore(models) {
    const known = new Set(models.map((m) => m.id));
    const saved = readSaved();
    const chosen = saved ? saved.models.filter((id) => known.has(id)) : [];
    const patch = { selectedModelIds: chosen.length > 0 ? chosen : models.map((m) => m.id) };
    if (saved) {
      patch.prompt = saved.prompt;
      patch.nVariants = Math.min(MAX_VARIANTS, Math.max(1, Math.round(saved.nVariants)));
      patch.temperature = Math.min(2, Math.max(0.1, saved.temperature));
    }
    return patch;
  }

  return {
    /** Load models and vocabulary from the API. Safe to call again to retry. */
    async load() {
      store.set({ bootState: 'loading', bootError: null });
      try {
        const [models, vocabulary] = await Promise.all([api.getModels(), api.getVocabulary()]);
        store.set({ models, vocabulary, ...restore(models), bootState: 'ready' });
      } catch (error) {
        store.set({ bootError: error.message, bootState: 'error' });
      }
    },

    async generate() {
      const state = store.get();
      const { check, seed, canGenerate } = derive(state);
      if (!canGenerate) return;

      pending?.abort();
      const controller = new AbortController();
      pending = controller;
      store.set({ status: 'loading', error: null });
      try {
        const result = await api.generate(
          {
            prompt: check.canonical,
            models: state.selectedModelIds,
            n_variants: state.nVariants,
            temperature: state.temperature,
            seed,
          },
          controller.signal,
        );
        if (pending === controller) store.set({ result, status: 'done' });
      } catch (error) {
        if (error?.name === 'AbortError' || pending !== controller) return; // superseded
        store.set({ error, status: 'error' });
      }
    },

    // --- Prompt editing ------------------------------------------------------------
    setPrompt: (prompt) => store.set({ prompt }),
    clearPrompt: () => store.set({ prompt: '' }),
    appendWord: (word) => store.set((s) => ({ prompt: `${s.prompt.trim()} ${word}`.trim() })),

    /** Replace every occurrence of a misspelt word, keeping the rest of the prompt. */
    replaceWord: (unknown, suggestion) =>
      store.set((s) => ({
        prompt: tokenizePrompt(s.prompt)
          .map((word) => (word === unknown ? suggestion : word))
          .join(' '),
      })),

    // --- Settings ------------------------------------------------------------------
    toggleModel(id) {
      store.set((s) => {
        const chosen = new Set(s.selectedModelIds);
        if (!chosen.delete(id)) chosen.add(id);
        // Keep the configured order so results always appear in the same columns.
        return { selectedModelIds: s.models.map((m) => m.id).filter((m) => chosen.has(m)) };
      });
    },
    setVariants: (n) => store.set({ nVariants: n }),
    setTemperature: (temperature) => store.set({ temperature }),
    setSeedText: (seedText) => store.set({ seedText }),
    setExampleGroup: (exampleGroup) => store.set({ exampleGroup }),
    useSeed: (seed) =>
      store.set({
        seedText: String(seed),
        notice: `Seed ${seed} will be used for the next generation.`,
      }),
    notify: (notice) => store.set({ notice }),

    /** Persist the settings worth remembering between visits (per-browser convenience). */
    save() {
      const s = store.get();
      if (s.bootState !== 'ready') return;
      try {
        storage?.setItem(
          STORAGE_KEY,
          JSON.stringify({
            prompt: s.prompt,
            models: s.selectedModelIds,
            nVariants: s.nVariants,
            temperature: s.temperature,
          }),
        );
      } catch {
        /* private mode or quota: the app works without persistence */
      }
    },
  };
}
