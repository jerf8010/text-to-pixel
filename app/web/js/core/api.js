/** Thin client for the pixelgen HTTP API. Every failure is delivered as an `ApiError`. */

const BASE_URL = '/api';

/** Uniform error: the API's `{error: {code, message, details}}` or a synthesised one. */
export class ApiError extends Error {
  constructor({ code, message, details = {} }) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.details = details;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(BASE_URL + path, options);
  } catch (error) {
    if (error?.name === 'AbortError') throw error; // the caller cancelled on purpose
    throw new ApiError({
      code: 'network',
      message: 'Cannot reach the server. Check that the API is running on port 8000.',
    });
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    /* not JSON (for example an HTML error page) */
  }

  if (!response.ok) {
    const error = body?.error;
    if (error?.code && error?.message) throw new ApiError(error);
    throw new ApiError({
      code: `http_${response.status}`,
      message: `The server answered with an error (${response.status}).`,
    });
  }
  return body;
}

export const api = {
  getModels: () => request('/models'),
  getVocabulary: () => request('/vocabulary'),
  generate: (payload, signal) =>
    request('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    }),
};
