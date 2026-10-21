/** Types that mirror the JSON contract of the backend (`pixelgen_api/schemas.py`). */

export interface ModelInfo {
  id: string;
  label: string;
  kind: string;
  n_params: number;
  description: string;
}

export interface ExampleGroup {
  title: string;
  description: string;
  prompts: string[];
}

export interface Vocabulary {
  words: string[];
  groups: Record<string, string[]>;
  max_words: number;
  /** RGB color of every palette index. */
  palette: number[][];
  examples: ExampleGroup[];
}

export interface GenerateRequest {
  prompt: string;
  models: string[];
  n_variants: number;
  temperature: number;
  seed: number | null;
}

export interface GeneratedImage {
  /** Palette indices, `pixels[row][column]`. */
  pixels: number[][];
}

export interface ModelResult {
  id: string;
  label: string;
  images: GeneratedImage[];
  elapsed_ms: number;
}

export interface GenerateResponse {
  /** Normalised prompt that was actually used. */
  prompt: string;
  /** Seed that produced the result; send it back to reproduce it. */
  seed: number;
  results: ModelResult[];
}

/** Uniform error returned by the API (and synthesised for network failures). */
export interface ApiError {
  code: string;
  message: string;
  details: Record<string, unknown>;
}

export const MAX_SEED = 2 ** 31 - 1;
