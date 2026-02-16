/**
 * WebSocket connection configuration with layered URL resolution.
 *
 * Priority order:
 *   1. localStorage override (user-set via UI or console)
 *   2. VITE_WS_URL env variable (build-time override)
 *   3. Auto-detect from window.location (zero-config default)
 */

export const WS_PATH = '/api/v1/ws/orchestrated';
export const DEFAULT_API_PORT = 8000;
export const STORAGE_KEY = 'atlas-ws-url';

/** Standard HTTP/HTTPS ports that indicate a reverse proxy is in use. */
const STANDARD_PORTS = ['80', '443', ''];

/**
 * Derive the WebSocket URL from the browser's current location.
 * - Maps http → ws, https → wss
 * - Substitutes the dev-server port with DEFAULT_API_PORT,
 *   unless the page is served on a standard port (reverse proxy).
 */
export function getAutoDetectedUrl(): string {
  const { protocol, hostname, port } = window.location;
  const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';

  // Behind a reverse proxy (standard port) → omit port, proxy routes to backend
  const wsPort = STANDARD_PORTS.includes(port) ? '' : `:${DEFAULT_API_PORT}`;

  return `${wsProtocol}//${hostname}${wsPort}${WS_PATH}`;
}

/**
 * Get the WebSocket URL using layered resolution:
 *   localStorage → env variable → auto-detect
 */
export function getWebSocketUrl(): string {
  // 1. localStorage override (highest priority)
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) return stored;

  // 2. Build-time env variable
  const envUrl = import.meta.env.VITE_WS_URL;
  if (envUrl) return envUrl;

  // 3. Auto-detect from current page URL
  return getAutoDetectedUrl();
}

/** Persist a custom WebSocket URL to localStorage. */
export function setWebSocketUrl(url: string): void {
  localStorage.setItem(STORAGE_KEY, url);
}

/** Remove the localStorage override, reverting to auto-detect. */
export function clearCustomUrl(): void {
  localStorage.removeItem(STORAGE_KEY);
}

/** Check whether a user-set localStorage override exists. */
export function hasCustomUrl(): boolean {
  return localStorage.getItem(STORAGE_KEY) !== null;
}
