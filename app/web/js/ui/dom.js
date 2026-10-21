/**
 * Small DOM helpers. Text is always inserted with `textContent` / text nodes, never as HTML,
 * so prompts and server messages can never inject markup.
 */

const PROPERTIES = new Set(['value', 'checked', 'selected', 'disabled', 'hidden']);

/**
 * Create an element.
 * @param {string} tag
 * @param {Record<string, any>} [props] `class`, `on<event>` handlers, properties and attributes.
 * @param {...(Node | string | null | false | undefined | Array<any>)} children
 */
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (PROPERTIES.has(key)) node[key] = value;
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

/** Replace the content of `node` with `children`. */
export function replaceChildren(node, ...children) {
  node.replaceChildren();
  append(node, children);
}
