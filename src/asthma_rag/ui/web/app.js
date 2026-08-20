(() => {
  "use strict";

  const chatEl = document.getElementById("chat");
  const composer = document.getElementById("composer");
  const input = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const micBtn = document.getElementById("micBtn");
  const examplesEl = document.getElementById("examples");
  const railEl = document.getElementById("rail");
  const railToggle = document.getElementById("railToggle");
  const statusPill = document.getElementById("statusPill");
  const safetyBtn = document.getElementById("safetyBtn");
  const safetyDialog = document.getElementById("safetyDialog");
  const voiceSelect = document.getElementById("voiceSelect");
  const speakToggle = document.getElementById("speakToggle");
  const ttsPlayer = document.getElementById("ttsPlayer");

  const confidenceBlock = document.getElementById("confidenceBlock");
  const sourcesBlock = document.getElementById("sourcesBlock");
  const sourceList = document.getElementById("sourceList");
  const videoBlock = document.getElementById("videoBlock");
  const videoFrame = document.getElementById("videoFrame");

  let history = [];

  // ---------------------------------------------------------------- utils

  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  /** Minimal, safe markdown-ish renderer: escapes HTML first, then applies
   * a small set of transforms (bold, links, line breaks, simple lists). */
  function renderMarkdown(text) {
    let safe = escapeHtml(text);
    safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    safe = safe.replace(/(https?:\/\/[^\s)]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
    safe = safe.split(/\n{2,}/).map((block) => {
      const lines = block.split("\n").filter(Boolean);
      const isList = lines.length > 1 && lines.every((l) => /^[-*]\s+/.test(l.trim()));
      if (isList) {
        const items = lines.map((l) => `<li>${l.trim().replace(/^[-*]\s+/, "")}</li>`).join("");
        return `<ul>${items}</ul>`;
      }
      return `<p>${block.replace(/\n/g, "<br>")}</p>`;
    }).join("");
    return safe;
  }

  function scrollToBottom() {
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function clearEmptyState() {
    const empty = chatEl.querySelector(".empty-state");
    if (empty) empty.remove();
  }

  // ---------------------------------------------------------------- health

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      statusPill.classList.remove("ok", "warn", "error");
      const textEl = statusPill.querySelector(".status-text");
      if (data.status === "ok") {
        statusPill.classList.add("ok");
        textEl.textContent = "ready";
      } else {
        statusPill.classList.add("warn");
        textEl.textContent = `missing ${data.missing_keys.join(", ")}`;
      }
    } catch {
      statusPill.classList.add("error");
      statusPill.querySelector(".status-text").textContent = "offline";
    }
  }

  async function loadVoices() {
    try {
      const res = await fetch("/api/voices");
      const data = await res.json();
      voiceSelect.innerHTML = "";
      for (const v of data.voices) {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = v.label;
        if (v.id === "hannah") opt.selected = true;
        voiceSelect.appendChild(opt);
      }
    } catch {
      voiceSelect.innerHTML = '<option value="hannah">Hannah (clear)</option>';
    }
  }

  // ---------------------------------------------------------------- render

  function addMessage(role, text, opts = {}) {
    clearEmptyState();
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}${opts.refused ? " refused" : ""}`;

    if (role === "assistant" && (opts.routeLabel || opts.confidence)) {
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      if (opts.routeLabel) {
        const tag = document.createElement("span");
        tag.className = "route-tag";
        tag.textContent = opts.routeLabel;
        meta.appendChild(tag);
      }
      if (opts.confidence) {
        const tag = document.createElement("span");
        tag.className = `confidence-tag ${opts.confidence.toLowerCase().split(" ")[0]}`;
        tag.textContent = opts.confidence;
        meta.appendChild(tag);
      }
      wrap.appendChild(meta);
    }

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "assistant") {
      bubble.innerHTML = renderMarkdown(text);
    } else {
      bubble.textContent = text;
    }
    wrap.appendChild(bubble);

    if (opts.videoHtml) {
      const v = document.createElement("div");
      v.className = "msg-video";
      v.innerHTML = opts.videoHtml;
      wrap.appendChild(v);
    }

    chatEl.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function addThinking() {
    clearEmptyState();
    const wrap = document.createElement("div");
    wrap.className = "msg assistant thinking";
    wrap.id = "thinkingMsg";
    wrap.innerHTML = `<div class="bubble"><span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span></div>`;
    chatEl.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function removeThinking() {
    const el = document.getElementById("thinkingMsg");
    if (el) el.remove();
  }

  function updateRail(data) {
    confidenceBlock.innerHTML = "<h2>Last answer</h2>";
    const card = document.createElement("div");
    card.className = "confidence-card";
    const route = document.createElement("div");
    route.className = "route";
    route.textContent = data.route_label;
    card.appendChild(route);

    if (data.confidence) {
      const tag = document.createElement("span");
      tag.className = `confidence-tag ${data.confidence.toLowerCase().split(" ")[0]}`;
      tag.textContent = `Confidence: ${data.confidence}`;
      card.appendChild(tag);
    }
    if (data.judge_reason) {
      const reason = document.createElement("div");
      reason.className = "reason";
      reason.textContent = data.judge_reason;
      card.appendChild(reason);
    }
    confidenceBlock.appendChild(card);

    if (data.sources && data.sources.length) {
      sourcesBlock.hidden = false;
      sourceList.innerHTML = "";
      for (const s of data.sources) {
        const li = document.createElement("li");
        li.className = "source-item";
        const score = typeof s.rerank_score === "number" ? s.rerank_score.toFixed(2) : "—";
        li.innerHTML = `<div class="doc-name">${escapeHtml(s.doc_name)}</div>
          <div class="doc-meta">p.${escapeHtml(String(s.page))} · chunk ${escapeHtml(String(s.chunk_id))} · relevance ${score}</div>`;
        sourceList.appendChild(li);
      }
    } else if (data.web_sources && data.web_sources.length) {
      sourcesBlock.hidden = false;
      sourceList.innerHTML = "";
      for (const s of data.web_sources) {
        const li = document.createElement("li");
        li.className = "source-item";
        li.innerHTML = `<div class="doc-name"><a href="${s.url}" target="_blank" rel="noopener">${escapeHtml(s.title)}</a></div>
          <div class="doc-meta">${escapeHtml(s.published_date || "date unknown")}</div>`;
        sourceList.appendChild(li);
      }
    } else {
      sourcesBlock.hidden = true;
    }

    if (data.video_html) {
      videoBlock.hidden = false;
      videoFrame.innerHTML = data.video_html;
    } else {
      videoBlock.hidden = true;
      videoFrame.innerHTML = "";
    }
  }

  // ---------------------------------------------------------------- send

  async function sendMessage(text, displayText) {
    const question = text.trim();
    if (!question) return;

    addMessage("user", displayText || question);
    input.value = "";
    autoGrow();
    sendBtn.disabled = true;
    document.querySelector(".app").classList.add("thinking");
    addThinking();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question }),
      });
      removeThinking();

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        addMessage("assistant", err.detail || "Something broke on the way to an answer. Check the server logs and try again.", { refused: true });
        return;
      }

      const data = await res.json();
      addMessage("assistant", data.answer, {
        routeLabel: data.route_label,
        confidence: data.confidence,
        refused: data.refused,
        videoHtml: data.video_html,
      });
      updateRail(data);

      if (speakToggle.checked && data.answer) {
        speakText(data.answer);
      }
    } catch (e) {
      removeThinking();
      addMessage("assistant", "Couldn't reach the server. Confirm it's running and try again.", { refused: true });
    } finally {
      sendBtn.disabled = false;
      document.querySelector(".app").classList.remove("thinking");
    }
  }

  async function speakText(text) {
    try {
      const res = await fetch("/api/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice: voiceSelect.value }),
      });
      if (!res.ok) return;
      const blob = await res.blob();
      ttsPlayer.src = URL.createObjectURL(blob);
      ttsPlayer.play().catch(() => {});
    } catch {
      /* silent: TTS is a convenience, not core functionality */
    }
  }

  // ---------------------------------------------------------------- events

  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(input.value);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input.value);
    }
  });

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
  }
  input.addEventListener("input", autoGrow);

  examplesEl.addEventListener("click", (e) => {
    const btn = e.target.closest(".chip");
    if (!btn) return;
    sendMessage(btn.dataset.q);
  });

  safetyBtn.addEventListener("click", () => safetyDialog.showModal());

  railToggle.addEventListener("click", () => {
    const open = railEl.classList.toggle("open");
    railToggle.setAttribute("aria-expanded", String(open));
  });

  // ---------------------------------------------------------------- voice

  let mediaRecorder = null;
  let chunks = [];

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
      mediaRecorder.onstop = onRecordingStop;
      mediaRecorder.start();
      micBtn.setAttribute("aria-pressed", "true");
    } catch {
      addMessage("assistant", "Microphone access was denied or unavailable. Type your question instead.", { refused: true });
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    }
    micBtn.setAttribute("aria-pressed", "false");
  }

  async function onRecordingStop() {
    const blob = new Blob(chunks, { type: "audio/webm" });
    if (blob.size < 500) return; // too short, likely an accidental tap

    addThinking();
    document.querySelector(".app").classList.add("thinking");
    try {
      const form = new FormData();
      form.append("audio", blob, "question.webm");
      const res = await fetch("/api/transcribe", { method: "POST", body: form });
      removeThinking();
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        addMessage("assistant", err.detail || "Transcription failed. Try again or type your question.", { refused: true });
        return;
      }
      const data = await res.json();
      if (data.text && data.text.trim()) {
        sendMessage(data.text, `🎤 ${data.text}`);
      }
    } catch {
      removeThinking();
      addMessage("assistant", "Transcription failed. Try again or type your question.", { refused: true });
    } finally {
      document.querySelector(".app").classList.remove("thinking");
    }
  }

  micBtn.addEventListener("click", () => {
    if (micBtn.getAttribute("aria-pressed") === "true") {
      stopRecording();
    } else {
      startRecording();
    }
  });

  // ---------------------------------------------------------------- init

  function showEmptyState() {
    const wrap = document.createElement("div");
    wrap.className = "empty-state";
    wrap.innerHTML = `<h1>Ask about asthma with confidence</h1>
      <p>Diagnosis, controller therapy, inhaler technique, day-to-day management — every answer is grounded in indexed guideline text and checked by a safety judge before it reaches you.</p>`;
    chatEl.appendChild(wrap);
  }

  showEmptyState();
  checkHealth();
  loadVoices();
  input.focus();
})();
