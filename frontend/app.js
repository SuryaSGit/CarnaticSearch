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
  feedbackYes:    $("#feedback-yes"),
  feedbackOther:  $("#feedback-other"),
  feedbackForm:   $("#feedback-form"),
  correctSong:    $("#correct-song"),
  feedbackStatus: $("#feedback-status"),
  stats:          $("#stats"),
  retrainBtn:     $("#retrain-btn"),
  retrainStatus:  $("#retrain-status"),
};

let lastResult = null;  // { query, top_pick, shortlist }


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
  els.feedbackForm.classList.add("hidden");

  els.topPick.innerHTML = `
    <div class="label">Top pick</div>
    <h2 class="song-name">${escapeHtml(top_pick.song)}</h2>
    <p class="composer">${escapeHtml(top_pick.composer)}</p>
    <p class="lyrics-preview">${escapeHtml(truncate(top_pick.lyrics, 240))}</p>
  `;

  els.shortlist.innerHTML = shortlist
    .map(
      (s) => `
        <li>
          <div class="song-name">${escapeHtml(s.song)}</div>
          <div class="composer">${escapeHtml(s.composer)}</div>
        </li>`,
    )
    .join("");
}


// -----------------------
// Feedback
// -----------------------
els.feedbackYes.addEventListener("click", () => {
  if (!lastResult) return;
  submitFeedback(lastResult.top_pick.song);
});

els.feedbackOther.addEventListener("click", () => {
  els.feedbackForm.classList.toggle("hidden");
  els.correctSong.focus();
});

els.feedbackForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const correct = els.correctSong.value.trim();
  if (!correct) return;
  submitFeedback(correct);
});

async function submitFeedback(correctSong) {
  if (!lastResult) return;
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
    if (data.ml_correct) {
      els.feedbackStatus.textContent = "Saved — ML got it right.";
    } else if (data.correct_in_shortlist) {
      els.feedbackStatus.textContent = "Saved — useful training signal (correct was in shortlist but not the top pick).";
    } else {
      els.feedbackStatus.textContent = "Saved — correct song wasn't in the shortlist.";
    }
    els.feedbackStatus.classList.remove("error");
    els.feedbackForm.classList.add("hidden");
    els.correctSong.value = "";
    loadStats();
  } catch (err) {
    els.feedbackStatus.textContent = `Feedback failed: ${err.message}`;
    els.feedbackStatus.classList.add("error");
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
// Retrain
// -----------------------
els.retrainBtn.addEventListener("click", async () => {
  els.retrainBtn.disabled = true;
  const originalText = els.retrainBtn.textContent;
  els.retrainBtn.textContent = "Retraining…";
  els.retrainStatus.textContent = "";

  try {
    const r = await fetch(`${API}/retrain`, { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    els.retrainStatus.textContent = data.trained
      ? "Model retrained successfully — new picks will use the updated ranker."
      : "Not enough usable feedback yet to retrain.";
  } catch (err) {
    els.retrainStatus.textContent = `Retrain failed: ${err.message}`;
  } finally {
    els.retrainBtn.disabled = false;
    els.retrainBtn.textContent = originalText;
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
