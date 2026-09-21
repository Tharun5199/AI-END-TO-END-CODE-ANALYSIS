const repoUrlInput = document.getElementById("repoUrl");
const analyzeBtn = document.getElementById("analyzeBtn");
const repoStatus = document.getElementById("repoStatus");
const configBanner = document.getElementById("configBanner");
const chatPanel = document.getElementById("chatPanel");
const chatHeader = document.getElementById("chatHeader");
const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

let history = []; // list of [role, text] -- only completed question/answer pairs
let busy = false;

/* ------------------------------------------------------------------ */
/* Minimal, XSS-safe Markdown renderer (no external libraries).        */
/* Everything is HTML-escaped FIRST; only a known set of tags is added */
/* back afterwards, so model output can never inject markup/scripts.   */
/* ------------------------------------------------------------------ */
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInline(text) {
  const codeSpans = [];
  text = text.replace(/`([^`\n]+)`/g, (_, code) => {
    codeSpans.push(code);
    return `\u0000C${codeSpans.length - 1}\u0000`;
  });
  text = text
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(
      /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
  return text.replace(/\u0000C(\d+)\u0000/g, (_, i) => `<code>${codeSpans[Number(i)]}</code>`);
}

function renderMarkdown(src) {
  const codeBlocks = [];
  src = String(src || "").replace(/\r\n/g, "\n");

  // 1. Pull fenced code blocks out before anything else touches them.
  src = src.replace(/```([\w+#.-]*)[^\n]*\n([\s\S]*?)(?:```|$)/g, (_, lang, code) => {
    const cls = lang ? ` class="lang-${escapeHtml(lang)}"` : "";
    codeBlocks.push(`<pre><code${cls}>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return `\n\u0000B${codeBlocks.length - 1}\u0000\n`;
  });

  // 2. Block-level parsing on the escaped text.
  const lines = escapeHtml(src).split("\n");
  const out = [];
  let para = [];
  let list = null;
  let table = [];

  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${renderInline(para.join("<br>"))}</p>`);
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      const items = list.items.map((item) => `<li>${renderInline(item)}</li>`).join("");
      out.push(`<${list.type}>${items}</${list.type}>`);
      list = null;
    }
  };
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.map((r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
    const hasHeader = rows.length > 1 && rows[1].every((c) => /^:?-{2,}:?$/.test(c));
    if (hasHeader) {
      const head = rows[0].map((c) => `<th>${renderInline(c)}</th>`).join("");
      const body = rows
        .slice(2)
        .map((r) => `<tr>${r.map((c) => `<td>${renderInline(c)}</td>`).join("")}</tr>`)
        .join("");
      out.push(`<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`);
    } else {
      out.push(`<p>${table.map(renderInline).join("<br>")}</p>`);
    }
    table = [];
  };
  const flushAll = () => {
    flushPara();
    flushList();
    flushTable();
  };

  for (const line of lines) {
    const isTableRow = /^\s*\|.*\|\s*$/.test(line);
    if (!isTableRow && table.length) flushTable();

    const block = line.match(/^\u0000B(\d+)\u0000$/);
    if (block) {
      flushAll();
      out.push(codeBlocks[Number(block[1])]);
      continue;
    }
    if (isTableRow) {
      flushPara();
      flushList();
      table.push(line);
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushAll();
      const level = Math.min(heading[1].length + 2, 6);
      out.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }
    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const type = ul ? "ul" : "ol";
      if (!list || list.type !== type) {
        flushList();
        list = { type, items: [] };
      }
      list.items.push((ul || ol)[1]);
      continue;
    }
    if (!line.trim()) {
      flushAll();
      continue;
    }
    if (list && /^\s{2,}\S/.test(line)) {
      list.items[list.items.length - 1] += " " + line.trim();
      continue;
    }
    flushList();
    para.push(line);
  }
  flushAll();
  return out.join("");
}

/* ------------------------------------------------------------------ */
/* UI helpers                                                          */
/* ------------------------------------------------------------------ */
function setStatus(text, kind) {
  repoStatus.textContent = text;
  repoStatus.classList.toggle("error", kind === "error");
  repoStatus.classList.toggle("ok", kind === "ok");
}

function setChatEnabled(enabled) {
  chatInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
}

function appendMessage(role, text, sources) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const roleLabel = document.createElement("div");
  roleLabel.className = "role";
  roleLabel.textContent = role === "user" ? "You" : "AutoCode Analyzer";
  wrap.appendChild(roleLabel);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    bubble.classList.add("markdown");
    bubble.innerHTML = renderMarkdown(text); // safe: renderMarkdown escapes all input first
  } else {
    bubble.textContent = text;
  }
  wrap.appendChild(bubble);

  if (sources && sources.length) {
    const sourcesWrap = document.createElement("div");
    sourcesWrap.className = "sources";
    sources.forEach((src) => {
      const chip = document.createElement("span");
      chip.className = "source-chip";
      chip.textContent = src;
      sourcesWrap.appendChild(chip);
    });
    wrap.appendChild(sourcesWrap);
  }

  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return wrap;
}

function appendError(text) {
  const el = appendMessage("assistant", text);
  el.classList.add("error");
  return el;
}

function showRepoLoaded(repo) {
  chatHeader.textContent = `Analyzing ${repo.repo_name} - ${repo.num_files} files, ${repo.num_chunks} chunks`;
  chatPanel.hidden = false;
  setChatEnabled(true);
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = {};
  try {
    data = await res.json();
  } catch (_) {
    data = { error: `Server returned HTTP ${res.status}.` };
  }
  return { ok: res.ok, data };
}

/* ------------------------------------------------------------------ */
/* Actions                                                             */
/* ------------------------------------------------------------------ */
async function loadStatus() {
  try {
    const res = await fetch("/status");
    const data = await res.json();
    if (!data.llm_configured) {
      configBanner.textContent = `Setup needed: ${data.config_problem}`;
      configBanner.hidden = false;
    }
    if (data.repo) {
      repoUrlInput.value = data.repo.repo_url;
      showRepoLoaded(data.repo);
      setStatus("A repository is already loaded on the server - ask away, or analyze another one.", "ok");
    }
  } catch (_) {
    /* status is best-effort; the app still works without it */
  }
}

async function analyzeRepo() {
  if (busy) return;
  const githubUrl = repoUrlInput.value.trim();
  if (!githubUrl) {
    setStatus("Enter a GitHub repository URL first.", "error");
    return;
  }

  busy = true;
  analyzeBtn.disabled = true;
  setChatEnabled(false);
  const started = Date.now();
  const tick = () =>
    setStatus(`Cloning and indexing the repository... ${Math.round((Date.now() - started) / 1000)}s`);
  tick();
  const timer = setInterval(tick, 1000);

  try {
    const { ok, data } = await postJSON("/ingest", { github_url: githubUrl });
    if (!ok) {
      setStatus(data.error || "Failed to analyze repository.", "error");
      if (!chatPanel.hidden) setChatEnabled(true); // the previously loaded repo is still usable
      return;
    }
    const note = data.truncated ? " (large repo: only the first files up to MAX_FILES were indexed)" : "";
    setStatus(`Indexed ${data.num_files} files into ${data.num_chunks} chunks in ${data.seconds}s${note}. Ask away!`, "ok");
    history = [];
    chatLog.innerHTML = "";
    showRepoLoaded(data);
    chatInput.focus();
  } catch (_) {
    setStatus("Network error - is the server still running?", "error");
  } finally {
    clearInterval(timer);
    analyzeBtn.disabled = false;
    busy = false;
  }
}

async function sendQuestion(question) {
  if (busy) return;
  busy = true;
  setChatEnabled(false);
  analyzeBtn.disabled = true;

  appendMessage("user", question);
  const typingEl = appendMessage("assistant", "Thinking...");
  typingEl.querySelector(".bubble").classList.add("typing");

  try {
    const { ok, data } = await postJSON("/chat", { question, history });
    typingEl.remove();
    if (!ok) {
      appendError(data.error || "Something went wrong.");
      return; // failed turns are NOT added to history
    }
    appendMessage("assistant", data.answer, data.sources);
    history.push(["human", question], ["ai", data.answer]);
  } catch (_) {
    typingEl.remove();
    appendError("Network error - is the server still running?");
  } finally {
    busy = false;
    setChatEnabled(true);
    analyzeBtn.disabled = false;
    chatInput.focus();
  }
}

analyzeBtn.addEventListener("click", analyzeRepo);
repoUrlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") analyzeRepo();
});

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question || busy) return;
  chatInput.value = "";
  sendQuestion(question);
});

loadStatus();
