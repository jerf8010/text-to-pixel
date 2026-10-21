/**
 * Client-side prompt validation. It mirrors `pixelgen_api/services/prompt.py` so the user
 * gets instant feedback; the server remains the authority and re-validates every request.
 */

export interface WordCheck {
  text: string;
  known: boolean;
  /** Close vocabulary words, only filled for unknown words. */
  suggestions: string[];
}

export type PromptIssue = 'empty' | 'too_long' | 'unknown_words';

export interface PromptCheck {
  words: WordCheck[];
  /** First problem found, or `null` when the prompt can be sent. */
  issue: PromptIssue | null;
  /** Explanation to show the user; `null` when there is nothing to say (e.g. empty prompt). */
  message: string | null;
  /** Lower-case prompt with single spaces. */
  canonical: string;
  valid: boolean;
}

/** Lower-case `text`, drop punctuation and split it into words. */
export function tokenizePrompt(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}_\s]/gu, ' ')
    .split(/\s+/)
    .filter((word) => word.length > 0);
}

/** Levenshtein distance between two short strings. */
export function editDistance(a: string, b: string): number {
  let previous = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const current = [i];
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      current[j] = Math.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost);
    }
    previous = current;
  }
  return previous[b.length];
}

/** Up to `limit` vocabulary words close to `word`, nearest first. */
export function suggestWords(word: string, vocabulary: readonly string[], limit = 3): string[] {
  const maxDistance = Math.max(1, Math.floor(word.length / 3));
  return vocabulary
    .map((candidate) => ({ candidate, distance: editDistance(word, candidate) }))
    .filter(({ distance }) => distance <= maxDistance)
    .sort((x, y) => x.distance - y.distance || x.candidate.localeCompare(y.candidate))
    .slice(0, limit)
    .map(({ candidate }) => candidate);
}

/** Check a prompt against the model vocabulary and length limit. */
export function validatePrompt(
  text: string,
  vocabulary: readonly string[],
  maxWords: number,
): PromptCheck {
  const tokens = tokenizePrompt(text);
  const known = new Set(vocabulary);
  const words: WordCheck[] = tokens.map((token) =>
    known.has(token)
      ? { text: token, known: true, suggestions: [] }
      : { text: token, known: false, suggestions: suggestWords(token, vocabulary) },
  );
  const canonical = tokens.join(' ');

  if (tokens.length === 0) {
    return { words, issue: 'empty', message: null, canonical, valid: false };
  }
  if (tokens.length > maxWords) {
    return {
      words,
      issue: 'too_long',
      message: `The description has ${tokens.length} words; the maximum is ${maxWords}.`,
      canonical,
      valid: false,
    };
  }
  const unknown = [...new Set(words.filter((w) => !w.known).map((w) => w.text))];
  if (unknown.length > 0) {
    return {
      words,
      issue: 'unknown_words',
      message: `Not in the vocabulary: ${unknown.join(', ')}.`,
      canonical,
      valid: false,
    };
  }
  return { words, issue: null, message: null, canonical, valid: true };
}
