import { editDistance, suggestWords, tokenizePrompt, validatePrompt } from './prompt-validation';

const VOCAB = ['a', 'small', 'large', 'red', 'blue', 'circle', 'square', 'left', 'of'];

describe('tokenizePrompt', () => {
  it('lower-cases, strips punctuation and collapses spaces', () => {
    expect(tokenizePrompt('  A Small,  RED circle! ')).toEqual(['a', 'small', 'red', 'circle']);
  });

  it('returns no words for blank text', () => {
    expect(tokenizePrompt(' ... ')).toEqual([]);
  });
});

describe('editDistance', () => {
  it('counts single edits', () => {
    expect(editDistance('circle', 'circle')).toBe(0);
    expect(editDistance('circel', 'circle')).toBe(2);
    expect(editDistance('cirle', 'circle')).toBe(1);
  });
});

describe('suggestWords', () => {
  it('proposes the nearest vocabulary word first', () => {
    expect(suggestWords('cirle', VOCAB)[0]).toBe('circle');
  });

  it('proposes nothing for unrelated words', () => {
    expect(suggestWords('hexagon', VOCAB)).toEqual([]);
  });
});

describe('validatePrompt', () => {
  it('accepts a prompt made of known words', () => {
    const check = validatePrompt('A small red circle.', VOCAB, 6);
    expect(check.valid).toBe(true);
    expect(check.canonical).toBe('a small red circle');
    expect(check.issue).toBeNull();
  });

  it('stays silent about an empty prompt', () => {
    const check = validatePrompt('  ', VOCAB, 6);
    expect(check.valid).toBe(false);
    expect(check.issue).toBe('empty');
    expect(check.message).toBeNull();
  });

  it('rejects prompts that are too long', () => {
    const check = validatePrompt('a small red circle left of a', VOCAB, 4);
    expect(check.issue).toBe('too_long');
    expect(check.valid).toBe(false);
  });

  it('flags unknown words and suggests fixes', () => {
    const check = validatePrompt('a small red cirle', VOCAB, 6);
    expect(check.issue).toBe('unknown_words');
    const unknown = check.words.filter((w) => !w.known);
    expect(unknown.map((w) => w.text)).toEqual(['cirle']);
    expect(unknown[0].suggestions).toContain('circle');
  });
});
