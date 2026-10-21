import { api } from './core/api.js';
import { createActions, initialState } from './core/generator.js';
import { createStore } from './core/store.js';
import { mountActionBar } from './ui/action-bar.js';
import { mountPromptPanel } from './ui/prompt-panel.js';
import { mountResultsPanel } from './ui/results-panel.js';
import { mountSettingsPanel } from './ui/settings-panel.js';

/** localStorage can be missing or throw (private mode, blocked site data). */
function safeStorage() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

const store = createStore(initialState);
const actions = createActions({ store, api, storage: safeStorage() });

const boot = document.getElementById('boot');
const bootError = document.getElementById('boot-error');
const bootMessage = document.getElementById('boot-error-message');
const layout = document.getElementById('layout');
document.getElementById('retry').addEventListener('click', () => actions.load());

function renderBoot(state) {
  boot.hidden = state.bootState !== 'loading';
  bootError.hidden = state.bootState !== 'error';
  layout.hidden = state.bootState !== 'ready';
  bootMessage.textContent = state.bootError ?? '';
}

// The panels render only once the vocabulary is loaded; mount them when it arrives.
let mounted = false;
store.subscribe((state) => {
  renderBoot(state);
  if (state.bootState === 'ready' && !mounted) {
    mounted = true;
    mountPromptPanel(store, actions);
    mountSettingsPanel(store, actions);
    mountResultsPanel(store, actions);
    mountActionBar(store, actions);
  }
  actions.save();
});

renderBoot(store.get());
actions.load();
