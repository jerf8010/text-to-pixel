import { derive } from '../core/generator.js';
import { h, replaceChildren } from './dom.js';

/** The primary action: the Generate button and the reason it may be disabled. */
export function mountActionBar(store, actions) {
  const button = document.getElementById('generate');
  const hint = document.getElementById('action-hint');

  button.addEventListener('click', () => actions.generate());

  function render(state) {
    const { canGenerate, blockedReason } = derive(state);
    button.disabled = !canGenerate;
    replaceChildren(
      button,
      state.status === 'loading'
        ? [h('span', { class: 'spinner', 'aria-hidden': 'true' }), ' Generating…']
        : 'Generate',
    );
    hint.textContent = blockedReason ?? '';
  }

  store.subscribe(render);
  render(store.get());
}
