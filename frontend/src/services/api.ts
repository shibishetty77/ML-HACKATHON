import { ApiError } from '@/types/api';

const DEFAULT_TIMEOUT_MS = 15000;
const STORAGE_KEY = 'pathsnap_api_base_url';

export function getApiBaseUrl(): string {
  if (typeof window !== 'undefined') {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && saved.trim()) {
      return saved.trim().replace(/\/+$/, '');
    }
  }
  const envUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, '');
  }
  return 'http://localhost:8000';
}

export function setApiBaseUrl(url: string): void {
  if (typeof window !== 'undefined') {
    const cleaned = url.trim().replace(/\/+$/, '');
    localStorage.setItem(STORAGE_KEY, cleaned);
    window.dispatchEvent(new Event('pathsnap_api_url_changed'));
  }
}

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<T> {
  const baseUrl = getApiBaseUrl();
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const url = `${baseUrl}${cleanEndpoint}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorBody: unknown = null;
      try {
        errorBody = await response.json();
      } catch {
        errorBody = await response.text();
      }

      const errorMessage =
        (typeof errorBody === 'object' && errorBody !== null && 'error' in errorBody
          ? String((errorBody as { error: unknown }).error)
          : null) ||
        (typeof errorBody === 'object' && errorBody !== null && 'detail' in errorBody
          ? String((errorBody as { detail: unknown }).detail)
          : null) ||
        `HTTP Error ${response.status}: ${response.statusText}`;

      const err: ApiError = {
        message: errorMessage,
        statusCode: response.status,
        details: errorBody,
        isNetworkError: false,
      };
      throw err;
    }

    const data = (await response.json()) as T;
    return data;
  } catch (error: unknown) {
    clearTimeout(timeoutId);

    if ((error as { name?: string }).name === 'AbortError') {
      const timeoutError: ApiError = {
        message: `Request timed out after ${timeoutMs / 1000}s. Check if Laptop 1 is running and reachable at ${baseUrl}.`,
        isNetworkError: true,
      };
      throw timeoutError;
    }

    if (error && typeof error === 'object' && 'statusCode' in error) {
      throw error;
    }

    const networkError: ApiError = {
      message: `Failed to connect to ${baseUrl}. Check local Wi-Fi / Hotspot connectivity, IP address, and firewall permissions.`,
      isNetworkError: true,
      details: error,
    };
    throw networkError;
  }
}
