const BASE = import.meta.env.VITE_API_BASE || "/api";
export const DEMO = false;

export function setToken() {}
export function setMarket() {}
export function getMarket() {
  return "kr";
}
export function setUnauthorizedHandler() {}

async function request(path, { method = "GET", body } = {}) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { "content-type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || data.detail || "요청을 처리하지 못했습니다.");
  }
  return data;
}

export const api = {
  login: async () => ({ access_token: "", user: { name: "Lab", email: "local" } }),
  signup: async () => ({ access_token: "", user: { name: "Lab", email: "local" } }),
  me: async () => ({ name: "Lab", email: "local" }),
  config: () => request("/config"),
  runs: () => request("/runs"),
  startExperiment: (payload) =>
    request("/experiments", { method: "POST", body: payload }),
  job: (id) => request(`/jobs/${id}`),
};

export async function readText(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} 로딩 실패`);
  return res.text();
}

export async function readJson(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} 로딩 실패`);
  return res.json();
}
