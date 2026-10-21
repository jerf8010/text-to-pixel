import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  editDistance,
  suggestWords,
  tokenizePrompt,
  validatePrompt,
} from '../js/core/prompt-validation.js';

const VOCAB = ['a', 'small', 'large', 'red', 'blue', 'circle', 'square', 'left', 'of'];

describe('tokenizePrompt', () => {
  it('lower-cases, strips punctuation and collapses spaces', () => {
    assert.deepEqual(tokenizePrompt('  A Small,  RED circle! '), ['a', 'small', 'red', 'circle']);
  });

  it('returns no words for blank text', () => {
    assert.deepEqual(tokenizePrompt(' ... '), []);
  });
});

describe('editDistance', () => {
  it('counts single edits', () => {
    assert.equal(editDistance('circle', 'circle'), 0);
    assert.equal(editDistance('circel', 'circle'), 2);
    assert.equal(editDistance('cirle', 'circle'), 1);
  });
});

describe('suggestWords', () => {
  it('proposes the nearest vocabulary word first', () => {
    assert.equal(suggestWords('cirle', VOCAB)[0], 'circle');
  });

  it('proposes nothing for unrelated words', () => {
    assert.deepEqual(suggestWords('hexagon', VOCAB), []);
  });
});

describe('validatePrompt', () => {
  it('accepts a prompt made of known words', () => {
    const check = validatePrompt('A small red circle.', VOCAB, 6);
    assert.equal(check.valid, true);
    assert.equal(check.canonical, 'a small red circle');
    assert.equal(check.issue, null);
  });

  it('stays silent about an empty prompt', () => {
    const check = validatePrompt('  ', VOCAB, 6);
    assert.equal(check.valid, false);
    assert.equal(check.issue, 'empty');
    assert.equal(check.message, null);
  });

  it('rejects prompts that are too long', () => {
    const check = validatePrompt('a small red circle left of a', VOCAB, 4);
    assert.equal(check.issue, 'too_long');
    assert.equal(check.valid, false);
  });

  it('flags unknown words and suggests fixes', () => {
    const check = validatePrompt('a small red cirle', VOCAB, 6);
    assert.equal(check.issue, 'unknown_words');
    const unknown = check.words.filter((w) => !w.known);
    assert.deepEqual(unknown.map((w) => w.text), ['cirle']);
    assert.ok(unknown[0].suggestions.includes('circle'));
  });
});
