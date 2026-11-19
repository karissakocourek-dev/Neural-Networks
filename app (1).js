// Wearable Activity Detective - exhibit logic and opt-in WebLLM chatbot.
// All numbers shown on the page come from results.json and gallery.json,
// which are written by train_and_evaluate.py.

const CHAT_MODEL = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
const WEBLLM_URL = "https://esm.run/@mlc-ai/web-llm";

const state = {
  results: null,
  windows: [],
  classes: [],
  guideText: "",
  queue: [],
  current: null,
  answered: false,
  seen: 0,
  youRight: 0,
  modelRight: 0,
  personFilter: null,
};

const el = (id) => document.getElementById(id);
const pct = (v) => (v * 100).toFixed(1) + "%";

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error("Could not load " + path);
  return response.json();
}

async function start() {
  try {
    const [results, gallery, guide] = await Promise.all([
      loadJson("results.json"),
      loadJson("gallery.json"),
      fetch("site_guide.md").then((r) => r.text()),
    ]);
    state.results = results;
    state.windows = gallery.windows;
    state.classes = gallery.classes;
    state.guideText = guide;
  } catch (error) {
    el("guess-section").insertAdjacentHTML(
      "beforeend",
      "<p>The exhibit data could not be loaded. If you opened this file directly from your computer, " +
        "use the GitHub Pages link instead, because browsers block local file loading.</p>"
    );
    return;
  }

  el("guide-text").textContent = state.guideText;
  buildGuessButtons();
  buildSpotlight();
  buildResults();
  el("filter").addEventListener("change", () => {
    state.personFilter = null;
    refillQueue();
    nextWindow();
  });
  el("next").addEventListener("click", nextWindow);
  el("spotlight-btn").addEventListener("click", showFeaturedMistake);
  refillQueue();
  nextWindow();
  setupChat();
}

// ---------- Section 1: guess the walk ----------

function buildGuessButtons() {
  const row = el("guess-row");
  state.classes.forEach((name, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    button.addEventListener("click", () => guess(index));
    row.appendChild(button);
  });
}

function shuffled(list) {
  const copy = list.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function refillQueue() {
  let pool = state.windows;
  if (el("filter").value === "wrong") pool = pool.filter((w) => w.truth !== w.pred);
  if (state.personFilter !== null) {
    pool = state.windows.filter((w) => w.subject === state.personFilter && w.truth !== w.pred);
  }
  state.queue = shuffled(pool);
}

function nextWindow() {
  if (state.queue.length === 0) refillQueue();
  showWindow(state.queue.pop());
}

function showWindow(win) {
  state.current = win;
  state.answered = false;
  el("reveal").hidden = true;
  document.querySelectorAll("#guess-row button").forEach((b) => (b.disabled = false));
  drawTrace(win);
}

function guess(index) {
  if (state.answered || !state.current) return;
  state.answered = true;
  const win = state.current;
  state.seen += 1;
  if (index === win.truth) state.youRight += 1;
  if (win.pred === win.truth) state.modelRight += 1;
  document.querySelectorAll("#guess-row button").forEach((b) => (b.disabled = true));

  const reveal = el("reveal");
  reveal.hidden = false;
  reveal.textContent = "";

  const verdict = document.createElement("p");
  verdict.className = "verdict " + (index === win.truth ? "right" : "wrong");
  verdict.textContent =
    (index === win.truth ? "You were right. " : "Not this time. ") +
    "The person was: " + state.classes[win.truth] + ". " +
    "The network said: " + state.classes[win.pred] +
    (win.pred === win.truth ? " (correct)." : " (wrong).");
  reveal.appendChild(verdict);

  const bars = document.createElement("div");
  bars.className = "bars";
  state.classes.forEach((name, i) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.textContent = name + (i === win.truth ? " (truth)" : "");
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill" + (i === win.truth ? " truth" : "");
    fill.style.width = (win.proba[i] * 100).toFixed(1) + "%";
    track.appendChild(fill);
    const value = document.createElement("span");
    value.textContent = pct(win.proba[i]);
    row.append(label, track, value);
    bars.appendChild(row);
  });
  reveal.appendChild(bars);

  const who = document.createElement("p");
  who.className = "note";
  who.textContent = "This window came from test person " + win.subject +
    ", who was not used for training or tuning.";
  reveal.appendChild(who);

  updateTally();
}

function updateTally() {
  el("tally").textContent =
    "Windows seen: " + state.seen + ". You got " + state.youRight +
    " right. The network got " + state.modelRight + " right on the same windows.";
}

function drawTrace(win) {
  const canvas = el("trace");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const lo = -2;
  const hi = 2;
  const padL = 44, padR = 12, padT = 12, padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const yOf = (v) => padT + plotH * (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo));
  const xOf = (i) => padL + (plotW * i) / 127;

  ctx.strokeStyle = "#e3e6e8";
  ctx.fillStyle = "#5b6770";
  ctx.font = "12px system-ui, sans-serif";
  ctx.lineWidth = 1;
  for (let v = lo; v <= hi; v += 1) {
    ctx.beginPath();
    ctx.moveTo(padL, yOf(v));
    ctx.lineTo(w - padR, yOf(v));
    ctx.stroke();
    ctx.fillText(v + " g", 6, yOf(v) + 4);
  }
  [0, 0.64, 1.28, 1.92, 2.56].forEach((t) => {
    ctx.fillText(t.toFixed(2) + " s", padL + (plotW * t) / 2.56 - 14, h - 8);
  });

  [["x", "#1a5fb4"], ["y", "#c2410c"], ["z", "#15803d"]].forEach(([axis, color]) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    win[axis].forEach((v, i) => (i === 0 ? ctx.moveTo(xOf(i), yOf(v)) : ctx.lineTo(xOf(i), yOf(v))));
    ctx.stroke();
  });
}

// ---------- Section 2: featured mistake ----------

function findWindow(id) {
  return state.windows.find((w) => w.id === id);
}

function showFeaturedMistake() {
  const mistake = state.results.highlighted_mistake;
  if (!mistake) return;
  const win = findWindow(mistake.id);
  if (win) showWindow(win);
  el("guess-section").scrollIntoView({ behavior: "smooth" });
}

function buildSpotlight() {
  const m = state.results.highlighted_mistake;
  const box = el("spotlight-text");
  if (!m) {
    box.textContent = "The network made no mistakes on the test set.";
    return;
  }
  const lines = [
    "The network was " + pct(m.confidence) + " sure that a window from test person " + m.subject +
      " was " + m.predicted.toLowerCase() + ". The person was actually " + m.truth.toLowerCase() + ".",
    "This is the wrong answer it was most confident about. Use the button in section 1 to look at the trace yourself.",
    "What might have helped (these are ideas, not tested fixes): this person's walking may simply look different " +
      "from the " + state.results.n_train_subjects + " people the network trained on, so more and more varied training volunteers could help. " +
      "Rescaling each person's data by their own typical movement might help too. " +
      "Training on the raw signal instead of the 561 summary numbers is another option. " +
      "Whatever the cause, a confident score of " + pct(m.confidence) + " on a wrong answer shows the network's " +
      "confidence should not be treated as proof.",
  ];
  box.textContent = "";
  lines.forEach((text) => {
    const p = document.createElement("p");
    p.textContent = text;
    box.appendChild(p);
  });
}

// ---------- Section 3: results ----------

function buildResults() {
  const r = state.results;
  const summary = el("results-summary");
  const intro = document.createElement("p");
  intro.textContent =
    "The test set is " + r.n_test_windows + " windows from " + r.n_test_subjects +
    " people who were kept out of training and tuning.";
  summary.appendChild(intro);

  const table = document.createElement("table");
  const rows = [
    ["Model", "Test accuracy"],
    ["Baseline: always guess \"" + r.baseline_predicts + "\"", pct(r.test_accuracy_baseline_most_frequent)],
    ["Neural network (one hidden layer of " + r.chosen.hidden.join(", ") + " units)", pct(r.test_accuracy_network)],
    ["Extra comparison: logistic regression", pct(r.test_accuracy_logistic_regression)],
  ];
  rows.forEach((cells, index) => {
    const tr = document.createElement("tr");
    cells.forEach((text) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      cell.textContent = text;
      tr.appendChild(cell);
    });
    table.appendChild(tr);
  });
  summary.appendChild(table);

  const note = document.createElement("p");
  note.className = "note";
  note.textContent =
    "On the validation people used for tuning, the chosen network scored " + pct(r.chosen.val_accuracy) +
    ". On the test people it scored " + pct(r.test_accuracy_network) +
    ". The network beats the baseline by a wide margin, but the simpler logistic regression did better on this test set.";
  summary.appendChild(note);

  const personBars = el("person-bars");
  Object.entries(r.per_subject_test_accuracy)
    .sort((a, b) => a[1] - b[1])
    .forEach(([subject, accuracy]) => {
      const row = document.createElement("div");
      row.className = "person-row";
      const label = document.createElement("span");
      label.textContent = "Person " + subject;
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill" + (accuracy < 0.9 ? " low" : "");
      fill.style.width = (accuracy * 100).toFixed(1) + "%";
      track.appendChild(fill);
      const value = document.createElement("span");
      value.textContent = pct(accuracy);
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Show";
      button.addEventListener("click", () => {
        state.personFilter = Number(subject);
        el("filter").value = "wrong";
        refillQueue();
        if (state.queue.length === 0) {
          state.personFilter = null;
          refillQueue();
        }
        nextWindow();
        el("guess-section").scrollIntoView({ behavior: "smooth" });
      });
      row.append(label, track, value, button);
      personBars.appendChild(row);
    });

  const cm = r.confusion_matrix_rows_true_cols_pred;
  const confusion = document.createElement("table");
  const head = document.createElement("tr");
  ["True activity \\ Network said", ...r.classes].forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    head.appendChild(th);
  });
  confusion.appendChild(head);
  cm.forEach((counts, i) => {
    const tr = document.createElement("tr");
    const first = document.createElement("td");
    first.textContent = r.classes[i];
    tr.appendChild(first);
    counts.forEach((count) => {
      const td = document.createElement("td");
      td.textContent = count;
      tr.appendChild(td);
    });
    confusion.appendChild(tr);
  });
  el("confusion").appendChild(confusion);
}

// ---------- Section 4: opt-in WebGPU chatbot ----------

let engine = null;
const chatHistory = [];

function chatSystemPrompt() {
  return (
    "You are the guide for the Wearable Activity Detective exhibit. " +
    "Answer only from the SITE GUIDE below. Keep answers short, in plain words. " +
    "If the guide does not answer the question, reply exactly: " +
    "\"The site guide doesn't answer that.\" Do not guess or add outside facts.\n\n" +
    "SITE GUIDE:\n" + state.guideText
  );
}

function addMessage(kind, text) {
  const p = document.createElement("p");
  p.className = "msg " + kind;
  p.textContent = text;
  el("chat-log").appendChild(p);
  el("chat-log").scrollTop = el("chat-log").scrollHeight;
  return p;
}

function setupChat() {
  el("enable-chat").addEventListener("click", enableChat);
  el("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    askQuestion();
  });
}

async function enableChat() {
  const status = el("chat-status");
  if (!("gpu" in navigator)) {
    status.textContent = "This browser does not support WebGPU, so chat is unavailable. The exhibit still works. " +
      "Try a recent desktop Chrome or Edge.";
    return;
  }
  el("enable-chat").disabled = true;
  el("chat-progress").hidden = false;
  status.textContent = "Loading the chat model. The first load downloads several hundred MB.";
  try {
    const webllm = await import(WEBLLM_URL);
    engine = await webllm.CreateMLCEngine(CHAT_MODEL, {
      initProgressCallback: (report) => {
        el("chat-progress").value = report.progress || 0;
        status.textContent = report.text || "Loading...";
      },
    });
    el("chat-progress").hidden = true;
    status.textContent = "Chat is ready.";
    el("chat-area").hidden = false;
    addMessage("system", "Ask about the data, the model, or how well it works.");
  } catch (error) {
    el("chat-progress").hidden = true;
    el("enable-chat").disabled = false;
    status.textContent = "Chat could not start (" + (error && error.message ? error.message : "unknown error") +
      "). The exhibit still works without it.";
  }
}

async function askQuestion() {
  const input = el("chat-text");
  const question = input.value.trim();
  if (!question || !engine) return;
  input.value = "";
  el("chat-send").disabled = true;
  addMessage("user", "You: " + question);
  const bubble = addMessage("bot", "");
  chatHistory.push({ role: "user", content: question });
  const messages = [{ role: "system", content: chatSystemPrompt() }, ...chatHistory.slice(-6)];
  try {
    const chunks = await engine.chat.completions.create({
      messages,
      temperature: 0.1,
      max_tokens: 300,
      stream: true,
    });
    let reply = "";
    for await (const chunk of chunks) {
      reply += chunk.choices[0]?.delta?.content || "";
      bubble.textContent = reply;
      el("chat-log").scrollTop = el("chat-log").scrollHeight;
    }
    chatHistory.push({ role: "assistant", content: reply });
  } catch (error) {
    bubble.textContent = "Sorry, the chat model hit an error. The exhibit above still works.";
  }
  el("chat-send").disabled = false;
}

start();
