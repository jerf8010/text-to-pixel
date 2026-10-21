import { derive } from '../core/generator.js';
import { h, replaceChildren } from './dom.js';

/** Display order and labels of the vocabulary groups sent by the API. */
const GROUPS = [
  ['sizes', 'Size'],
  ['colors', 'Color'],
  ['shapes', 'Shape'],
  ['other', 'Position and linking words'],
];

/**
 * Step 1: write the prompt, with live validation, word shortcuts and examples.
 * The static structure lives in index.html; this module fills and updates it.
 */
export function mountPromptPanel(store, actions) {
  const $ = (id) => document.getElementById(id);
  const textarea = $('prompt');
  const wordCount = $('word-count');
  const clearButton = $('clear-prompt');
  const wordList = $('words');
  const feedback = $('feedback');
  const vocabGroups = $('vocab-groups');
  const exampleTabs = $('example-tabs');
  const exampleDescription = $('example-description');
  const exampleList = $('example-list');

  textarea.addEventListener('input', () => actions.setPrompt(textarea.value));
  textarea.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      actions.generate();
    }
  });
  clearButton.addEventListener('click', () => {
    actions.clearPrompt();
    textarea.focus();
  });

  let builtFor = null; // vocabulary the static parts were built for
  let examplesKey = null; // what the example widgets currently show

  function buildVocabulary(vocabulary) {
    replaceChildren(
      vocabGroups,
      GROUPS.filter(([key]) => vocabulary.groups[key]?.length).map(([key, label]) =>
        h(
          'div',
          { class: 'group' },
          h('span', { class: 'group-name' }, label),
          h(
            'div',
            { class: 'chips' },
            vocabulary.groups[key].map((word) =>
              h('button', { type: 'button', class: 'chip', onclick: () => actions.appendWord(word) }, word),
            ),
          ),
        ),
      ),
    );
  }

  function renderExamples(vocabulary, selected) {
    const groups = vocabulary.examples;
    const group = groups[selected];
    replaceChildren(
      exampleTabs,
      groups.map((g, i) =>
        h(
          'button',
          {
            type: 'button',
            class: 'chip',
            'aria-pressed': String(i === selected),
            onclick: () => actions.setExampleGroup(i),
          },
          g.title,
        ),
      ),
    );
    exampleDescription.textContent = group?.description ?? '';
    replaceChildren(
      exampleList,
      (group?.prompts ?? []).map((prompt) =>
        h(
          'li',
          {},
          h('button', { type: 'button', class: 'example', onclick: () => actions.setPrompt(prompt) }, prompt),
        ),
      ),
    );
  }

  function renderFeedback(check) {
    const unknown = new Map();
    for (const word of check?.words ?? []) {
      if (!word.known && !unknown.has(word.text)) unknown.set(word.text, word);
    }
    if (unknown.size > 0) {
      replaceChildren(
        feedback,
        h(
          'div',
          { class: 'problem' },
          [...unknown.values()].map((word) =>
            h(
              'p',
              {},
              `“${word.text}” is not in the vocabulary. `,
              word.suggestions.length > 0
                ? [
                    'Did you mean ',
                    word.suggestions.map((s) =>
                      h('button', { type: 'button', class: 'chip', onclick: () => actions.replaceWord(word.text, s) }, s),
                    ),
                    ' ?',
                  ]
                : 'Pick words from the lists below.',
            ),
          ),
        ),
      );
    } else if (check?.issue === 'too_long') {
      replaceChildren(feedback, h('p', { class: 'problem' }, check.message));
    } else {
      replaceChildren(feedback);
    }
  }

  function render(state) {
    const { vocabulary } = state;
    if (!vocabulary) return;
    if (builtFor !== vocabulary) {
      buildVocabulary(vocabulary);
      builtFor = vocabulary;
      examplesKey = null; // rebuild the examples too
    }
    if (examplesKey !== state.exampleGroup) {
      renderExamples(vocabulary, state.exampleGroup);
      examplesKey = state.exampleGroup;
    }

    const { check } = derive(state);
    const words = check.words;
    const unknownCount = words.filter((w) => !w.known).length;

    if (textarea.value !== state.prompt) textarea.value = state.prompt; // set by an example or chip
    textarea.setAttribute('aria-invalid', String(unknownCount > 0 || check.issue === 'too_long'));
    wordCount.textContent = `${words.length} / ${vocabulary.max_words} words`;
    wordCount.classList.toggle('over', words.length > vocabulary.max_words);
    clearButton.hidden = !state.prompt;

    wordList.hidden = words.length === 0;
    replaceChildren(
      wordList,
      words.map((word) =>
        h(
          'li',
          { class: word.known ? null : 'unknown' },
          word.text,
          word.known ? null : h('span', { class: 'sr-only' }, ' (not in the vocabulary)'),
        ),
      ),
    );
    renderFeedback(check);
  }

  store.subscribe(render);
  render(store.get());
}
