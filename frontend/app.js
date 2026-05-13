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
  correctSong:    $("#correct-song"),
  typeaheadList:  $("#typeahead-results"),
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
  els.correctSong.value = "";
  els.typeaheadList.classList.add("hidden");

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

  // Swap the top-pick label to a confirmed state
  const labelEl = els.topPick.querySelector(".label");
  if (labelEl) {
    labelEl.textContent = clickedEl === els.topPick
      ? "✓ Marked correct"
      : "Top pick (you marked another song correct)";
  }

  // Append a check badge to whatever the user clicked
  if (!clickedEl.querySelector(".picked-badge")) {
    const badge = document.createElement("span");
    badge.className = "picked-badge";
    badge.textContent = "✓ Your pick";
    clickedEl.appendChild(badge);
  }

  submitFeedback(songName);
}


// -----------------------
// Typeahead lookup against /songs
// -----------------------
let typeaheadTimer = null;
let typeaheadController = null;

els.correctSong.addEventListener("input", () => {
  const q = els.correctSong.value.trim();
  clearTimeout(typeaheadTimer);
  if (!q) {
    els.typeaheadList.classList.add("hidden");
    return;
  }
  typeaheadTimer = setTimeout(() => fetchTypeahead(q), 150);
});

els.correctSong.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    els.typeaheadList.classList.add("hidden");
  }
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".typeahead")) {
    els.typeaheadList.classList.add("hidden");
  }
});


async function fetchTypeahead(q) {
  if (typeaheadController) typeaheadController.abort();
  typeaheadController = new AbortController();
  try {
    const r = await fetch(
      `${API}/songs?q=${encodeURIComponent(q)}&limit=10`,
      { signal: typeaheadController.signal },
    );
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const matches = await r.json();
    renderTypeahead(matches);
  } catch (err) {
    if (err.name === "AbortError") return;
    els.typeaheadList.innerHTML = `<li class="empty">Lookup failed</li>`;
    els.typeaheadList.classList.remove("hidden");
  }
}


function renderTypeahead(matches) {
  if (matches.length === 0) {
    els.typeaheadList.innerHTML = `<li class="empty">No matches</li>`;
  } else {
    els.typeaheadList.innerHTML = matches
      .map(
        (m) => `
          <li data-song="${escapeHtml(m.song)}">
            <div class="song-name">${escapeHtml(m.song)}</div>
            <div class="composer">${escapeHtml(m.composer)}</div>
          </li>`,
      )
      .join("");
    els.typeaheadList.querySelectorAll("li[data-song]").forEach((li) => {
      li.addEventListener("click", () => {
        const song = li.getAttribute("data-song");
        els.correctSong.value = song;
        els.typeaheadList.classList.add("hidden");
        if (feedbackLocked) return;
        feedbackLocked = true;
        document.querySelectorAll("#shortlist li, #top-pick").forEach((el) =>
          el.classList.add("disabled"),
        );
        submitFeedback(song);
      });
    });
  }
  els.typeaheadList.classList.remove("hidden");
}


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
