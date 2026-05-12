const API = "";  // same-origin (FastAPI mounts frontend)

const $ = (sel) => document.querySelector(sel);

const els = {
  form:           $("#search-form"),
  query:          $("#query"),
  searchBtn:      $("#search-btn"),
  results:        $("#results"),
  topPick:        $("#top-pick"),
  shortlist:      $("#shortlist"),
  feedbackSec:    $("#feedback-section"),
  feedbackForm:   $("#feedback-form"),
  correctSong:    $("#correct-song"),
  feedbackStatus: $("#feedback-status"),
  notInList:      $("#not-in-list"),
  stats:          $("#stats"),
  retrainBtn:     $("#retrain-btn"),
  retrainStatus:  $("#retrain-status"),
};

let lastResult       = null;   // { query, top_pick, shortlist }
let feedbackLocked   = false;  // prevent double-submission per result


// -----------------------
// Search
// -----------------------
els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = els.query.value.trim();
  if (!query) return;

  els.searchBtn.disabled = true;
  els.searchBtn.textContent = "Searching…";
  els.feedbackStatus.textContent = "";
  els.feedbackStatus.classList.remove("error");
  feedbackLocked = false;

  try {
    const r = await fetch(`${API}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, shortlist_size: 5 }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    lastResult = await r.json();
    renderResults(lastResult);
  } catch (err) {
    els.feedbackStatus.textContent = `Search failed: ${err.message}`;
    els.feedbackStatus.classList.add("error");
  } finally {
    els.searchBtn.disabled = false;
    els.searchBtn.textContent = "Search";
  }
});


function renderResults({ top_pick, shortlist }) {
  els.results.classList.remove("hidden");
  els.feedbackSec.classList.remove("hidden");
  els.notInList.open = false;

  // Top pick
  els.topPick.innerHTML = `
    <div class="label">Top pick — click if correct</div>
    <h2 class="song-name">${escapeHtml(top_pick.song)}</h2>
    <p class="composer">${escapeHtml(top_pick.composer)}</p>
    <p class="lyrics-preview">${escapeHtml(truncate(top_pick.lyrics, 240))}</p>
  `;
  els.topPick.onclick = () => selectAnswer(top_pick.song, els.topPick);

  // Shortlist
  els.shortlist.innerHTML = "";
  shortlist.forEach((s) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="song-name">${escapeHtml(s.song)}</div>
      <div class="composer">${escapeHtml(s.composer)}</div>
    `;
    li.onclick = () => selectAnswer(s.song, li);
    els.shortlist.appendChild(li);
  });
}


// -----------------------
// Feedback (click on top pick or shortlist item)
// -----------------------
function selectAnswer(songName, clickedEl) {
  if (feedbackLocked) return;
  feedbackLocked = true;

  // Visual feedback
  clickedEl.classList.add("selected");
  document.querySelectorAll("#shortlist li, #top-pick").forEach((el) => {
    if (el !== clickedEl) el.classList.add("disabled");
  });

  submitFeedback(songName);
}


els.feedbackForm.addEventListener("submit", (e) => {
  e.preventDefault();
  if (feedbackLocked) return;
  const correct = els.correctSong.value.trim();
  if (!correct) return;
  feedbackLocked = true;
  document.querySelectorAll("#shortlist li, #top-pick").forEach((el) =>
    el.classList.add("disabled"),
  );
  submitFeedback(correct);
});


async function submitFeedback(correctSong) {
  if (!lastResult) return;
  els.feedbackStatus.textContent = "Saving…";
  els.feedbackStatus.classList.remove("error");

  try {
    const r = await fetch(`${API}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query:        lastResult.query,
        correct_song: correctSong,
        shortlist:    lastResult.shortlist,
        ml_pick:      lastResult.top_pick.song,
      }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();

    let msg;
    if (data.ml_correct) {
      msg = "Logged — ML got it right.";
    } else if (data.correct_in_shortlist) {
      msg = "Logged — correct was in shortlist but not top.";
    } else {
      msg = "Logged — correct wasn't in shortlist.";
    }

    els.feedbackStatus.textContent = msg + " Updating model…";
    els.correctSong.value = "";
    els.notInList.open = false;
    loadStats();

    // Auto-retrain in the background — each click moves the model
    autoRetrain();
  } catch (err) {
    els.feedbackStatus.textContent = `Feedback failed: ${err.message}`;
    els.feedbackStatus.classList.add("error");
    feedbackLocked = false;  // allow retry
    document.querySelectorAll("#shortlist li, #top-pick").forEach((el) =>
      el.classList.remove("disabled", "selected"),
    );
  }
}


async function autoRetrain() {
  try {
    const r = await fetch(`${API}/retrain`, { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    const baseMsg = els.feedbackStatus.textContent.replace(/ Updating model…$/, "");
    els.feedbackStatus.textContent = data.trained
      ? baseMsg + " Model updated."
      : baseMsg + " Need more samples to retrain.";
    loadStats();
  } catch (err) {
    els.feedbackStatus.textContent += ` (retrain failed: ${err.message})`;
  }
}


// -----------------------
// Stats
// -----------------------
async function loadStats() {
  try {
    const r = await fetch(`${API}/stats`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const s = await r.json();
    els.stats.innerHTML = `
      <div class="stat-card">
        <div class="stat-value">${s.total}</div>
        <div class="stat-label">Queries logged</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${s.ml_accuracy_pct ?? 0}%</div>
        <div class="stat-label">ML accuracy</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${s.shortlist_recall_pct ?? 0}%</div>
        <div class="stat-label">Shortlist recall</div>
      </div>
    `;
  } catch {
    els.stats.innerHTML = `<p class="muted">Stats unavailable</p>`;
  }
}

loadStats();


// -----------------------
// Manual retrain button (still available for full re-train)
// -----------------------
els.retrainBtn.addEventListener("click", async () => {
  els.retrainBtn.disabled = true;
  const original = els.retrainBtn.textContent;
  els.retrainBtn.textContent = "Retraining…";
  els.retrainStatus.textContent = "";

  try {
    const r = await fetch(`${API}/retrain`, { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    els.retrainStatus.textContent = data.trained
      ? "Model retrained successfully."
      : "Not enough usable feedback yet to retrain.";
  } catch (err) {
    els.retrainStatus.textContent = `Retrain failed: ${err.message}`;
  } finally {
    els.retrainBtn.disabled = false;
    els.retrainBtn.textContent = original;
  }
});


// -----------------------
// Utilities
// -----------------------
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function truncate(str, n) {
  if (!str) return "";
  return str.length > n ? str.slice(0, n).trimEnd() + "…" : str;
}
