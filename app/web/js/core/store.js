/**
 * A tiny observable store: one immutable state object, replaced on every change.
 * Subscribers are called with the new state right after each `set`.
 */
export function createStore(initial) {
  let state = initial;
  const listeners = new Set();

  return {
    get: () => state,

    /** Merge a patch (or the result of `patch(state)`) into the state and notify. */
    set(patch) {
      state = { ...state, ...(typeof patch === 'function' ? patch(state) : patch) };
      listeners.forEach((listener) => listener(state));
    },

    /** Register a listener; returns the function that removes it. */
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
