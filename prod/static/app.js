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

let lastResult     = null;   // { query, top_pick, shortlist }
let lastPick       = null;   // { song, element } — what's currently marked correct, if anything
let feedbackInFlight = false;  // gate concurrent network calls only


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
  lastPick = null;

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
    <div class="label">Top pick</div>
    <div class="song-info">
      <h2 class="song-name">${escapeHtml(top_pick.song)}</h2>
      <p class="composer">${escapeHtml(top_pick.composer)}</p>
      <p class="lyrics-preview">${escapeHtml(truncate(top_pick.lyrics, 240))}</p>
    </div>
    <button class="confirm-btn" type="button">✓ This is my song</button>
  `;
  els.topPick.onclick = () => openKarnatik(top_pick);
  els.topPick.querySelector(".confirm-btn").onclick = (e) => {
    e.stopPropagation();
    selectAnswer(top_pick.song, els.topPick);
  };

  // Shortlist
  els.shortlist.innerHTML = "";
  shortlist.forEach((s) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="song-info">
        <div class="song-name">${escapeHtml(s.song)}</div>
        <div class="composer">${escapeHtml(s.composer)}</div>
      </div>
      <button class="confirm-btn" type="button">✓ This is my song</button>
    `;
    li.onclick = () => openKarnatik(s);
    li.querySelector(".confirm-btn").onclick = (e) => {
      e.stopPropagation();
      selectAnswer(s.song, li);
    };
    els.shortlist.appendChild(li);
  });
}


function openKarnatik(song) {
  const q = `karnatik.com ${song.song} ${song.composer}`;
  const url = `https://www.google.com/search?q=${encodeURIComponent(q)}`;
  window.open(url, "_blank", "noopener,noreferrer");
}


// -----------------------
// Feedback (click on top pick or shortlist item)
// -----------------------
async function selectAnswer(songName, clickedEl) {
  if (feedbackInFlight) return;
  if (lastPick && lastPick.song === songName) return;  // already marked

  feedbackInFlight = true;
  els.feedbackStatus.textContent = "Saving…";
  els.feedbackStatus.classList.remove("error");

  try {
    // Switch case: revoke the previous pick first so counts don't double up
    if (lastPick) {
      await revokePrevious();
      lastPick = null;
    }

    // Apply the new pick
    const result = await submitFeedback(songName);
    setPickedVisual(clickedEl);
    lastPick = { song: songName, element: clickedEl };
    appendUndoLink(result.message);
  } finally {
    feedbackInFlight = false;
  }
}


async function undoPick() {
  if (feedbackInFlight || !lastPick) return;
  feedbackInFlight = true;
  els.feedbackStatus.textContent = "Undoing…";
  els.feedbackStatus.classList.remove("error");
  try {
    await revokePrevious();
    clearPickedVisual();
    lastPick = null;
    els.feedbackStatus.textContent = "Pick undone.";
  } finally {
    feedbackInFlight = false;
  }
}


async function revokePrevious() {
  if (!lastPick || !lastResult) return;
  try {
    await fetch(`${API}/feedback/revoke`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: lastResult.query,
        correct_song: lastPick.song,
      }),
    });
  } catch (err) {
    // Non-fatal — proceed with new pick anyway, the worst case is one
    // duplicated entry that fuzzy-counting still handles.
    console.warn("revoke failed:", err);
  }
  // Clear visuals on the previous card before re-applying
  clearPickedVisual();
}


function setPickedVisual(clickedEl) {
  // Card pick: highlight the chosen card, dim unpicked confirm buttons.
  // Typeahead pick (clickedEl null): no card visual — the Undo link in
  // the feedback status is the only affordance.
  if (clickedEl != null) {
    document.querySelectorAll(".confirm-btn").forEach((btn) => {
      const ownerEl = btn.closest("li") || btn.closest("#top-pick");
      if (ownerEl === clickedEl) {
        btn.textContent = "✓ Marked correct";
        btn.classList.add("confirmed");
        btn.classList.remove("dimmed");
      } else {
        btn.textContent = "✓ This is my song";
        btn.classList.remove("confirmed");
        btn.classList.add("dimmed");
      }
    });
    clickedEl.classList.add("selected");
    const labelEl = els.topPick.querySelector(".label");
    if (labelEl) {
      labelEl.textContent = clickedEl === els.topPick
        ? "Top pick — confirmed"
        : "Top pick (you marked another song correct)";
    }
  }
}


function appendUndoLink(statusText) {
  els.feedbackStatus.innerHTML = "";
  els.feedbackStatus.append(document.createTextNode(statusText + " "));
  const link = document.createElement("a");
  link.href = "#";
  link.className = "undo-link";
  link.textContent = "Undo";
  link.onclick = (e) => { e.preventDefault(); undoPick(); };
  els.feedbackStatus.append(link);
}


function clearPickedVisual() {
  document.querySelectorAll(".confirm-btn").forEach((btn) => {
    btn.textContent = "✓ This is my song";
    btn.classList.remove("confirmed", "dimmed");
    btn.disabled = false;
  });
  document.querySelectorAll(".selected").forEach((el) => el.classList.remove("selected"));
  document.querySelectorAll(".undo-btn").forEach((btn) => btn.remove());
  const labelEl = els.topPick.querySelector(".label");
  if (labelEl) labelEl.textContent = "Top pick";
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
        // Route through selectAnswer so undo/switch behavior is consistent.
        // No card to highlight (the song is being picked from the database
        // typeahead, not from the shortlist), so we pass null.
        selectAnswer(song, null);
      });
    });
  }
  els.typeaheadList.classList.remove("hidden");
}


async function submitFeedback(correctSong) {
  if (!lastResult) throw new Error("No active search");

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
      msg = "Marked correct (ML had it).";
    } else if (data.correct_in_shortlist) {
      msg = "Marked correct (was in shortlist, not top).";
    } else {
      msg = "Marked correct (wasn't in shortlist).";
    }

    els.correctSong.value = "";
    els.notInList.open = false;
    loadStats();

    // Auto-retrain in the background — each click moves the model
    autoRetrain();

    return { message: msg };
  } catch (err) {
    els.feedbackStatus.textContent = `Feedback failed: ${err.message}`;
    els.feedbackStatus.classList.add("error");
    throw err;
  }
}


async function autoRetrain() {
  // Silent: retrain in the background and refresh stats. We avoid touching
  // the feedback status so the Undo link stays intact.
  try {
    const r = await fetch(`${API}/retrain`, { method: "POST" });
    if (r.ok) loadStats();
  } catch {
    // ignore — stats will reflect actual state on next load
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
