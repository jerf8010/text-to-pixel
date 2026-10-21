import { Injectable, computed, effect, inject, signal } from '@angular/core';
import { Subscription, forkJoin } from 'rxjs';

import { ApiService } from '../api/api.service';
import { ApiError, GenerateResponse, MAX_SEED, ModelInfo, Vocabulary } from '../api/api.models';
import { tokenizePrompt, validatePrompt } from '../prompt/prompt-validation';

const STORAGE_KEY = 'pixelgen.settings.v1';
const MAX_VARIANTS = 4;

export type BootState = 'loading' | 'ready' | 'error';
export type GenerationStatus = 'idle' | 'loading' | 'done' | 'error';

interface SavedSettings {
  prompt: string;
  models: string[];
  nVariants: number;
  temperature: number;
}

/**
 * Single source of truth for the generator screen.
 *
 * Components read signals and call intent-named methods; they never touch HTTP or
 * storage themselves, which keeps them small and easy to test.
 */
@Injectable({ providedIn: 'root' })
export class GeneratorStore {
  private readonly api = inject(ApiService);
  private pending: Subscription | null = null;

  // --- Catalogue loaded at startup -------------------------------------------------
  readonly bootState = signal<BootState>('loading');
  readonly bootError = signal<string | null>(null);
  readonly vocabulary = signal<Vocabulary | null>(null);
  readonly models = signal<ModelInfo[]>([]);

  // --- User input ------------------------------------------------------------------
  readonly prompt = signal('');
  readonly selectedModelIds = signal<string[]>([]);
  readonly nVariants = signal(1);
  readonly temperature = signal(1.0);
  /** Raw text of the seed field; empty means "pick a random seed". */
  readonly seedText = signal('');

  // --- Generation state ------------------------------------------------------------
  readonly status = signal<GenerationStatus>('idle');
  readonly result = signal<GenerateResponse | null>(null);
  readonly error = signal<ApiError | null>(null);

  // --- Derived state ---------------------------------------------------------------
  readonly check = computed(() => {
    const vocabulary = this.vocabulary();
    return vocabulary ? validatePrompt(this.prompt(), vocabulary.words, vocabulary.max_words) : null;
  });

  readonly seedValid = computed(() => {
    const text = this.seedText().trim();
    return text === '' || (/^\d+$/.test(text) && Number(text) <= MAX_SEED);
  });

  /** Seed to send, or `null` to let the server choose one. */
  readonly seed = computed(() => {
    const text = this.seedText().trim();
    return text !== '' && this.seedValid() ? Number(text) : null;
  });

  /** Why the Generate button is disabled, or `null` when it can be used. */
  readonly blockedReason = computed<string | null>(() => {
    const check = this.check();
    if (!check) return 'The models are still loading.';
    if (check.issue === 'empty') return 'Write a description to continue.';
    if (!check.valid) return check.message;
    if (this.selectedModelIds().length === 0) return 'Select at least one model.';
    if (!this.seedValid()) return `The seed must be a whole number between 0 and ${MAX_SEED}.`;
    return null;
  });

  readonly canGenerate = computed(() => this.blockedReason() === null && this.status() !== 'loading');

  constructor() {
    // Remember the last settings between visits (a per-browser convenience only).
    effect(() => {
      if (this.bootState() !== 'ready') return;
      this.save({
        prompt: this.prompt(),
        models: this.selectedModelIds(),
        nVariants: this.nVariants(),
        temperature: this.temperature(),
      });
    });
  }

  /** Load models and vocabulary from the API. Safe to call again to retry. */
  load(): void {
    this.bootState.set('loading');
    this.bootError.set(null);
    forkJoin({ models: this.api.getModels(), vocabulary: this.api.getVocabulary() }).subscribe({
      next: ({ models, vocabulary }) => {
        this.models.set(models);
        this.vocabulary.set(vocabulary);
        this.restore(models);
        this.bootState.set('ready');
      },
      error: (error: ApiError) => {
        this.bootError.set(error.message);
        this.bootState.set('error');
      },
    });
  }

  generate(): void {
    const check = this.check();
    if (!check || !this.canGenerate()) return;

    this.pending?.unsubscribe();
    this.status.set('loading');
    this.error.set(null);
    this.pending = this.api
      .generate({
        prompt: check.canonical,
        models: this.selectedModelIds(),
        n_variants: this.nVariants(),
        temperature: this.temperature(),
        seed: this.seed(),
      })
      .subscribe({
        next: (response) => {
          this.result.set(response);
          this.status.set('done');
        },
        error: (error: ApiError) => {
          this.error.set(error);
          this.status.set('error');
        },
      });
  }

  // --- Prompt editing --------------------------------------------------------------
  setPrompt(text: string): void {
    this.prompt.set(text);
  }

  clearPrompt(): void {
    this.prompt.set('');
  }

  appendWord(word: string): void {
    this.prompt.update((text) => `${text.trim()} ${word}`.trim());
  }

  /** Replace every occurrence of a misspelt word, keeping the rest of the prompt. */
  replaceWord(unknown: string, suggestion: string): void {
    this.prompt.update((text) =>
      tokenizePrompt(text)
        .map((word) => (word === unknown ? suggestion : word))
        .join(' '),
    );
  }

  // --- Settings --------------------------------------------------------------------
  toggleModel(id: string): void {
    const chosen = new Set(this.selectedModelIds());
    if (!chosen.delete(id)) chosen.add(id);
    // Keep the configured order so results always appear in the same columns.
    this.selectedModelIds.set(this.models().map((m) => m.id).filter((m) => chosen.has(m)));
  }

  modelLabel(id: string): string {
    return this.models().find((m) => m.id === id)?.label ?? id;
  }

  // --- Persistence -----------------------------------------------------------------
  private restore(models: ModelInfo[]): void {
    const known = new Set(models.map((m) => m.id));
    const saved = this.read();
    const chosen = saved?.models.filter((id) => known.has(id)) ?? [];
    this.selectedModelIds.set(chosen.length > 0 ? chosen : models.map((m) => m.id));
    if (saved) {
      this.prompt.set(saved.prompt);
      this.nVariants.set(Math.min(MAX_VARIANTS, Math.max(1, Math.round(saved.nVariants))));
      this.temperature.set(Math.min(2, Math.max(0.1, saved.temperature)));
    }
  }

  private read(): SavedSettings | null {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const data = JSON.parse(raw) as Partial<SavedSettings>;
      if (
        typeof data.prompt !== 'string' ||
        !Array.isArray(data.models) ||
        typeof data.nVariants !== 'number' ||
        typeof data.temperature !== 'number'
      ) {
        return null;
      }
      return data as SavedSettings;
    } catch {
      return null; // storage unavailable or corrupted: start from defaults
    }
  }

  private save(settings: SavedSettings): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* private mode or quota: the app works without persistence */
    }
  }
}
