import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, catchError, throwError } from 'rxjs';

import { ApiError, GenerateRequest, GenerateResponse, ModelInfo, Vocabulary } from './api.models';

const BASE_URL = '/api';

/** Convert anything thrown by `HttpClient` into the uniform `ApiError` the UI works with. */
export function toApiError(error: unknown): ApiError {
  if (error instanceof HttpErrorResponse) {
    if (error.status === 0) {
      return {
        code: 'network',
        message: 'Cannot reach the server. Check that the API is running on port 8000.',
        details: {},
      };
    }
    const body: unknown = error.error;
    if (typeof body === 'object' && body !== null && 'error' in body) {
      const payload = (body as { error: Partial<ApiError> }).error;
      if (payload?.code && payload.message) {
        return { code: payload.code, message: payload.message, details: payload.details ?? {} };
      }
    }
    return {
      code: `http_${error.status}`,
      message: `The server answered with an error (${error.status}).`,
      details: {},
    };
  }
  return { code: 'unexpected', message: 'Something unexpected happened.', details: {} };
}

/** Error-stream handler: re-throws every failure as an `ApiError`. */
const rethrowAsApiError = (error: unknown) => throwError(() => toApiError(error));

/** Thin client for the pixelgen HTTP API. Every error is delivered as an `ApiError`. */
@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);

  getModels(): Observable<ModelInfo[]> {
    return this.http.get<ModelInfo[]>(`${BASE_URL}/models`).pipe(catchError(rethrowAsApiError));
  }

  getVocabulary(): Observable<Vocabulary> {
    return this.http.get<Vocabulary>(`${BASE_URL}/vocabulary`).pipe(catchError(rethrowAsApiError));
  }

  generate(request: GenerateRequest): Observable<GenerateResponse> {
    return this.http
      .post<GenerateResponse>(`${BASE_URL}/generate`, request)
      .pipe(catchError(rethrowAsApiError));
  }
}
