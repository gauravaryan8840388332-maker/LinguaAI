import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = Number(process.env.PORT || 3000);
const AZURE_API_VERSION = process.env.AZURE_API_VERSION || "2023-04-01";
const MAX_TEXT_CHARS = 125000;
const SYNC_KINDS = new Set(["SentimentAnalysis", "KeyPhraseExtraction", "LanguageDetection", "EntityRecognition"]);
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg"
};

function sendJson(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 1024 * 1024) throw Object.assign(new Error("Request body exceeds 1MB."), { status: 413 });
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw Object.assign(new Error("Request body must be valid JSON."), { status: 400 });
  }
}

function normalizeEndpoint(endpoint) {
  if (!endpoint || typeof endpoint !== "string") throw Object.assign(new Error("Azure endpoint is required."), { status: 400 });
  let parsed;
  try {
    parsed = new URL(endpoint.trim());
  } catch {
    throw Object.assign(new Error("Azure endpoint must be a valid HTTPS URL."), { status: 400 });
  }
  if (parsed.protocol !== "https:") throw Object.assign(new Error("Azure endpoint must use HTTPS."), { status: 400 });
  parsed.pathname = parsed.pathname.replace(/\/+$/, "");
  parsed.search = "";
  parsed.hash = "";
  return parsed.toString().replace(/\/$/, "");
}

function getAzureConfig(body) {
  const endpoint = normalizeEndpoint(body.endpoint || process.env.AZURE_LANGUAGE_ENDPOINT);
  const key = body.key || process.env.AZURE_LANGUAGE_KEY;
  if (!key || typeof key !== "string") throw Object.assign(new Error("Azure API key is required."), { status: 400 });
  return { endpoint, key };
}

function validateText(text) {
  if (!text || typeof text !== "string" || text.trim().length < 15) {
    throw Object.assign(new Error("Text must contain at least 15 characters."), { status: 400 });
  }
  if (text.length > MAX_TEXT_CHARS) {
    throw Object.assign(new Error(`Text exceeds the ${MAX_TEXT_CHARS.toLocaleString()} character limit.`), { status: 400 });
  }
  return text.trim();
}

async function readAzureError(response, fallback) {
  const raw = await response.text();
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw);
    return parsed?.error?.message || parsed?.message || fallback;
  } catch {
    return raw.slice(0, 400) || fallback;
  }
}

async function azureFetch(url, key, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": key,
    ...(options.headers || {})
  };
  Object.keys(headers).forEach((header) => {
    if (headers[header] === undefined || headers[header] === null) delete headers[header];
  });
  const response = await fetch(url, { ...options, headers });
  if (!response.ok) {
    const message = await readAzureError(response, `Azure request failed with HTTP ${response.status}.`);
    throw Object.assign(new Error(message), { status: response.status });
  }
  return response;
}

async function runSyncAnalysis({ endpoint, key, text, kind }) {
  if (!SYNC_KINDS.has(kind)) throw Object.assign(new Error("Unsupported analysis kind."), { status: 400 });
  const document = kind === "LanguageDetection" ? { id: "1", text } : { id: "1", language: "en", text };
  const response = await azureFetch(`${endpoint}/language/:analyze-text?api-version=${AZURE_API_VERSION}`, key, {
    method: "POST",
    body: JSON.stringify({ kind, parameters: { modelVersion: "latest" }, analysisInput: { documents: [document] } })
  });
  const data = await response.json();
  const documentResult = data?.results?.documents?.[0];
  if (!documentResult) throw Object.assign(new Error("Azure returned no document result."), { status: 502 });
  return documentResult;
}

async function runSummary({ endpoint, key, text }) {
  const submit = await azureFetch(`${endpoint}/language/analyze-text/jobs?api-version=${AZURE_API_VERSION}`, key, {
    method: "POST",
    body: JSON.stringify({
      displayName: "LinguaAI Summary",
      analysisInput: { documents: [{ id: "1", language: "en", text }] },
      tasks: [{ kind: "ExtractiveSummarization", parameters: { sentenceCount: 3 } }]
    })
  });
  const operationLocation = submit.headers.get("operation-location");
  if (!operationLocation) throw Object.assign(new Error("Azure did not return an operation-location header."), { status: 502 });

  for (let attempt = 0; attempt < 24; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, attempt < 2 ? 1000 : 1500));
    const poll = await azureFetch(operationLocation, key, { method: "GET", headers: { "Content-Type": undefined } });
    const data = await poll.json();
    if (data.status === "succeeded") return data.tasks?.items?.[0]?.results?.documents?.[0] || null;
    if (data.status === "failed" || data.status === "cancelled") {
      const message = data.errors?.[0]?.message || `Summarization ${data.status}.`;
      throw Object.assign(new Error(message), { status: 502 });
    }
  }
  throw Object.assign(new Error("Summarization timed out while polling Azure."), { status: 504 });
}

async function handleAnalyze(req, res) {
  try {
    const body = await readBody(req);
    const { endpoint, key } = getAzureConfig(body);
    const text = validateText(body.text);
    const mode = body.mode || "all";
    const tasks = [];
    const add = (name, promise) => tasks.push(promise.then((data) => [name, { ok: true, data }]).catch((error) => [name, { ok: false, error: error.message }]));

    if (mode === "all" || mode === "sentiment") add("sentiment", runSyncAnalysis({ endpoint, key, text, kind: "SentimentAnalysis" }));
    if (mode === "all" || mode === "keyphrase") add("keyphrase", runSyncAnalysis({ endpoint, key, text, kind: "KeyPhraseExtraction" }));
    if (mode === "all" || mode === "language") add("language", runSyncAnalysis({ endpoint, key, text, kind: "LanguageDetection" }));
    if (mode === "all" || mode === "ner") add("ner", runSyncAnalysis({ endpoint, key, text, kind: "EntityRecognition" }));
    if (mode === "all" || mode === "summary") add("summary", runSummary({ endpoint, key, text }));
    if (!tasks.length) throw Object.assign(new Error("Unsupported analysis mode."), { status: 400 });

    const results = Object.fromEntries(await Promise.all(tasks));
    const hasSuccess = Object.values(results).some((item) => item.ok);
    sendJson(res, hasSuccess ? 200 : 502, { mode, completedAt: new Date().toISOString(), results });
  } catch (error) {
    sendJson(res, error.status || 500, { error: { message: error.message || "Unexpected server error." } });
  }
}

async function serveStatic(req, res) {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const requested = url.pathname === "/" ? "/index.html" : decodeURIComponent(url.pathname);
  const filePath = path.normalize(path.join(__dirname, requested));
  if (!filePath.startsWith(__dirname)) {
    res.writeHead(403).end("Forbidden");
    return;
  }
  try {
    const data = await readFile(filePath);
    res.writeHead(200, { "Content-Type": MIME[path.extname(filePath).toLowerCase()] || "application/octet-stream" });
    res.end(data);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
}

createServer((req, res) => {
  if (req.method === "POST" && req.url?.startsWith("/api/analyze")) {
    handleAnalyze(req, res);
    return;
  }
  if (req.method === "GET" || req.method === "HEAD") {
    serveStatic(req, res);
    return;
  }
  res.writeHead(405, { Allow: "GET, HEAD, POST" }).end("Method not allowed");
}).listen(PORT, () => {
  console.log(`LinguaAI running at http://localhost:${PORT}`);
});
