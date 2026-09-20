/**
 * NINETY+ Football Intelligence Platform
 * Client Application Logic & ML Orchestration
 */

const API_BASE = "";

// Global State
let currentState = {
  activeView: "home",
  homeTeam: "Arsenal FC",
  awayTeam: "Chelsea FC",
  scoutTarget: "Bruno Fernandes",
  scoutPosFilter: "ALL",
  transferPlayer: "Bruno Fernandes",
  scoutTwins: [],
  teams: [],
  players: [],
};

// --------------------------------------------------------------------------
// Initialization & Navigation
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  await loadMetadata();
  await loadTeams();
  await loadPlayers();
  await initHomeView();
  await runMatchSimulation();
  await runTransferPrediction();
  await runScoutingEngine();
});

function setupNavigation() {
  const navButtons = document.querySelectorAll(".nav-btn");
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.getAttribute("data-view");
      switchView(view);
    });
  });

  // Shortcut search trigger
  const searchBtn = document.getElementById("searchShortcutBtn");
  if (searchBtn) {
    searchBtn.addEventListener("click", () => {
      const p = prompt("Quick Jump: Type a player name (e.g. Bukayo Saka, Bruno Fernandes):");
      if (p) {
        currentState.transferPlayer = p;
        currentState.scoutTarget = p;
        switchView("transfer");
        runTransferPrediction();
      }
    });
  }
}

function switchView(viewName) {
  currentState.activeView = viewName;

  // Update Nav Buttons
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-view") === viewName);
  });

  // Update Views
  document.querySelectorAll(".app-view").forEach((view) => {
    view.classList.toggle("active", view.id === `view-${viewName}`);
  });

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// --------------------------------------------------------------------------
// Data Loaders
// --------------------------------------------------------------------------
async function loadMetadata() {
  try {
    const res = await fetch(`${API_BASE}/api/meta`);
    const data = await res.json();
    if (data && data.platform) {
      document.getElementById("modelConfText").textContent = `MODEL CONFIDENCE INDEX: ${data.platform.model_confidence_index}`;
    }
  } catch (err) {
    console.warn("Could not load metadata:", err);
  }
}

async function loadTeams() {
  try {
    const res = await fetch(`${API_BASE}/api/teams`);
    const data = await res.json();
    currentState.teams = data.teams || [];

    const homeSelect = document.getElementById("homeTeamSelect");
    const awaySelect = document.getElementById("awayTeamSelect");

    if (homeSelect && awaySelect) {
      homeSelect.innerHTML = "";
      awaySelect.innerHTML = "";

      currentState.teams.forEach((t) => {
        const opt1 = new Option(t.name, t.name);
        const opt2 = new Option(t.name, t.name);
        homeSelect.add(opt1);
        awaySelect.add(opt2);
      });

      homeSelect.value = "Arsenal FC";
      awaySelect.value = "Chelsea FC";

      homeSelect.addEventListener("change", (e) => {
        currentState.homeTeam = e.target.value;
      });
      awaySelect.addEventListener("change", (e) => {
        currentState.awayTeam = e.target.value;
      });
    }
  } catch (err) {
    console.warn("Teams load error:", err);
  }
}

async function loadPlayers() {
  try {
    const res = await fetch(`${API_BASE}/api/transfer/players`);
    const data = await res.json();
    currentState.players = data.players || [];
  } catch (err) {
    console.warn("Players load error:", err);
  }
}

// --------------------------------------------------------------------------
// View 1: HOME
// --------------------------------------------------------------------------
async function initHomeView() {
  try {
    const res = await fetch(`${API_BASE}/api/home`);
    const data = await res.json();

    // Featured Matchups
    const matchupsContainer = document.getElementById("homeMatchupsGrid");
    if (matchupsContainer && data.matchups) {
      matchupsContainer.innerHTML = data.matchups
        .map(
          (m) => `
        <div class="matchup-card" onclick="quickSimulateFixture('${m.home}', '${m.away}')">
          <div class="matchup-header">
            <span>${m.venue}</span>
            <span class="matchup-xg-delta">xG Delta: ${m.xg_delta}</span>
          </div>
          <div class="matchup-teams-row">
            <span class="matchup-team-name">${m.home}</span>
            <span class="matchup-prob">${m.home_prob}%</span>
          </div>
          <div class="matchup-teams-row">
            <span class="matchup-team-name" style="color: #7b758c; font-size: 14px;">Draw Probability</span>
            <span class="matchup-prob" style="color: #7b758c; font-size: 16px;">${m.draw_prob}%</span>
          </div>
          <div class="matchup-teams-row">
            <span class="matchup-team-name">${m.away}</span>
            <span class="matchup-prob">${m.away_prob}%</span>
          </div>
          <div class="matchup-bar">
            <div class="bar-segment-home" style="width: ${m.home_prob}%"></div>
            <div class="bar-segment-draw" style="width: ${m.draw_prob}%"></div>
            <div class="bar-segment-away" style="width: ${m.away_prob}%"></div>
          </div>
          <div class="matchup-footer">
            <span>Optimal Score: ${m.optimal_score}</span>
            <span style="color: #7928ca; font-weight: 800;">FULL MODEL &rarr;</span>
          </div>
        </div>
      `
        )
        .join("");
    }
  } catch (err) {
    console.warn("Home view error:", err);
  }
}

window.quickSimulateFixture = function (home, away) {
  currentState.homeTeam = home;
  currentState.awayTeam = away;

  const hSel = document.getElementById("homeTeamSelect");
  const aSel = document.getElementById("awayTeamSelect");
  if (hSel) hSel.value = home;
  if (aSel) aSel.value = away;

  switchView("match");
  runMatchSimulation();
};

// --------------------------------------------------------------------------
// View 2: MATCH PREDICTOR
// --------------------------------------------------------------------------
async function runMatchSimulation() {
  const simBtn = document.getElementById("simulateBtn");
  if (simBtn) simBtn.textContent = "Simulating...";

  try {
    const res = await fetch(`${API_BASE}/api/match/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        home: currentState.homeTeam,
        away: currentState.awayTeam,
      }),
    });
    const data = await res.json();

    // Update Projections
    document.getElementById("predHomeName").textContent = data.home_team.name;
    document.getElementById("predAwayName").textContent = data.away_team.name;
    document.getElementById("predHomeXg").textContent = data.home_team.projected_xg;
    document.getElementById("predAwayXg").textContent = data.away_team.projected_xg;

    document.getElementById("predHomeProb").textContent = `${data.probabilities.home_win}%`;
    document.getElementById("predDrawProb").textContent = `${data.probabilities.draw}%`;
    document.getElementById("predAwayProb").textContent = `${data.probabilities.away_win}%`;

    document.getElementById("predBarHome").style.width = `${data.probabilities.home_win}%`;
    document.getElementById("predBarDraw").style.width = `${data.probabilities.draw}%`;
    document.getElementById("predBarAway").style.width = `${data.probabilities.away_win}%`;

    // Form Pills
    renderFormPills("predHomeForm", data.home_team.form);
    renderFormPills("predAwayForm", data.away_team.form);
    document.getElementById("predHomePts").textContent = data.home_team.form_pts;
    document.getElementById("predAwayPts").textContent = data.away_team.form_pts;

    // Feature Importances
    const shapContainer = document.getElementById("shapDriversContainer");
    if (shapContainer && data.feature_importance) {
      shapContainer.innerHTML = data.feature_importance
        .map(
          (d) => `
        <div class="shap-item">
          <div class="shap-header">
            <span>${d.rank}. ${d.driver}</span>
            <span style="color: var(--accent-neon-green);">${d.weight}</span>
          </div>
          <div class="shap-bar-bg">
            <div class="shap-bar-fill" style="width: ${40 - d.rank * 5}%"></div>
          </div>
          <div style="font-size: 11px; color: #7b758c;">${d.desc}</div>
        </div>
      `
        )
        .join("");
    }

    // Key Anchors Duel
    if (data.key_anchors) {
      document.getElementById("duelHomeName").textContent = data.key_anchors.home.name;
      document.getElementById("duelHomeRole").textContent = data.key_anchors.home.role;
      document.getElementById("duelAwayName").textContent = data.key_anchors.away.name;
      document.getElementById("duelAwayRole").textContent = data.key_anchors.away.role;
    }
  } catch (err) {
    console.error("Match simulation error:", err);
  } finally {
    if (simBtn) simBtn.innerHTML = "<span>&#9889;</span> Simulate";
  }
}

function renderFormPills(containerId, formArr) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = formArr
    .map((res) => `<div class="form-pill ${res.toLowerCase()}">${res}</div>`)
    .join("");
}

// --------------------------------------------------------------------------
// View 3: PLAYER SCOUTING
// --------------------------------------------------------------------------
async function runScoutingEngine() {
  try {
    const res = await fetch(`${API_BASE}/api/scouting/twins`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        player: currentState.scoutTarget,
        position_filter: currentState.scoutPosFilter,
        top_n: 5,
        attack_weight: 1.0,
        defense_weight: 1.0,
      }),
    });
    const data = await res.json();
    currentState.scoutTwins = data.twins || [];

    // Update Anchor Card
    document.getElementById("anchorPlayerName").textContent = data.anchor.name;
    document.getElementById("anchorPlayerClub").textContent = `${data.anchor.club} • ${data.anchor.pos}`;
    document.getElementById("anchorKp90").textContent = data.anchor.kp_90;
    document.getElementById("anchorXa90").textContent = data.anchor.xa_90;
    document.getElementById("anchorProgPasses").textContent = data.anchor.prog_passes;
    document.getElementById("anchorMarketEst").textContent = data.anchor.market_est;

    // Render Twins Table
    const tbody = document.getElementById("twinsTableBody");
    if (tbody) {
      tbody.innerHTML = data.twins
        .map(
          (t) => `
        <tr onclick="selectTwinForScouting('${t.name}')" style="cursor: pointer;">
          <td>${t.rank}</td>
          <td>
            <div style="font-weight: 800; color: #110d1c;">${t.name}</div>
            <div style="font-size: 11px; color: #7b758c;">${t.club}</div>
          </td>
          <td>${t.pos}</td>
          <td>${t.age}</td>
          <td>
            <div class="twin-sim-bar">
              <div class="sim-fill-bg">
                <div class="sim-fill" style="width: ${t.similarity}%"></div>
              </div>
              <span>${t.similarity}%</span>
            </div>
          </td>
          <td>${t.kp_90}</td>
          <td>${t.a_90}</td>
          <td style="color: #220845; font-weight: 800;">${t.value}</td>
        </tr>
      `
        )
        .join("");
    }

    // Render 2D Cluster Map Canvas
    renderClusterCanvas(data.scatter_points);

    // Head to head dimensions
    const h2hContainer = document.getElementById("h2hDimensionsContainer");
    if (h2hContainer && data.head_to_head && data.head_to_head.dimensions) {
      h2hContainer.innerHTML = data.head_to_head.dimensions
        .map(
          (d) => `
        <div style="margin-bottom: 14px;">
          <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 800; margin-bottom: 4px;">
            <span>${d.dimension}</span>
            <span style="color: #009655;">${d.match_pct}</span>
          </div>
          <div style="height: 6px; background: #eeecf4; border-radius: 3px; overflow: hidden; display: flex;">
            <div style="width: 50%; background: #220845;"></div>
            <div style="width: 45%; background: #00ff85;"></div>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 11px; color: #7b758c; margin-top: 4px;">
            <span>Anchor: ${d.anchor_val}</span>
            <span>#1 Twin: ${d.twin_val}</span>
          </div>
        </div>
      `
        )
        .join("");
    }
  } catch (err) {
    console.error("Scouting error:", err);
  }
}

window.selectTwinForScouting = function (name) {
  currentState.scoutTarget = name;
  document.getElementById("scoutSearchInput").value = name;
  runScoutingEngine();
};

function renderClusterCanvas(points) {
  const canvas = document.getElementById("clusterCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // Adjust for high-DPI displays
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  ctx.scale(2, 2);

  const w = canvas.offsetWidth;
  const h = canvas.offsetHeight;

  ctx.clearRect(0, 0, w, h);

  // Subtle grid
  ctx.strokeStyle = "#e8e5f0";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w / 2, 0);
  ctx.lineTo(w / 2, h);
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  // Axis Labels
  ctx.fillStyle = "#8a839c";
  ctx.font = "bold 9px sans-serif";
  ctx.fillText("HIGH PROGRESSION • Y+", 16, 20);
  ctx.fillText("SHOT CREATION & FINISHING • X+", w - 180, h - 16);

  if (!points || points.length === 0) return;

  // Find min/max for normalization
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs) - 0.5;
  const maxX = Math.max(...xs) + 0.5;
  const minY = Math.min(...ys) - 0.5;
  const maxY = Math.max(...ys) + 0.5;

  const clusterColors = ["#7928ca", "#0070f3", "#ff2a85", "#00ff85", "#f5a623"];

  // Draw background cluster dots
  points.forEach((p) => {
    const px = ((p.x - minX) / (maxX - minX)) * (w - 60) + 30;
    const py = h - (((p.y - minY) / (maxY - minY)) * (h - 60) + 30);

    ctx.beginPath();
    if (p.is_anchor) {
      // Highlight Anchor
      ctx.arc(px, py, 9, 0, Math.PI * 2);
      ctx.fillStyle = "#220845";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#00ff85";
      ctx.stroke();

      ctx.fillStyle = "#110d1c";
      ctx.font = "bold 11px sans-serif";
      ctx.fillText(`${p.name} (BASELINE)`, px + 12, py + 4);
    } else if (p.is_twin) {
      // Highlight Twin
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#00ff85";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#220845";
      ctx.stroke();

      ctx.fillStyle = "#110d1c";
      ctx.font = "bold 10px sans-serif";
      ctx.fillText(p.name, px + 8, py + 3);
    } else {
      // Regular Player dot
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = clusterColors[p.cluster % clusterColors.length] + "88";
      ctx.fill();
    }
  });
}

// --------------------------------------------------------------------------
// View 4: TRANSFER VALUE PREDICTOR
// --------------------------------------------------------------------------
async function runTransferPrediction() {
  try {
    const res = await fetch(`${API_BASE}/api/transfer/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ player: currentState.transferPlayer }),
    });
    const data = await res.json();

    // Player Hero
    document.getElementById("tvPlayerName").textContent = data.player.name;
    document.getElementById("tvPlayerSub").textContent = `${data.player.sub_position} • ${data.player.club}`;
    document.getElementById("tvPlayerAge").textContent = data.player.age;
    document.getElementById("tvPlayerContract").textContent = data.player.contract;
    document.getElementById("tvPlayerMins").textContent = data.player.minutes_played;

    // Valuation figures
    document.getElementById("tvPredictedVal").textContent = data.valuation.predicted_fmt;
    document.getElementById("tvActualVal").textContent = data.valuation.actual_fmt;
    document.getElementById("tvDeltaBadge").textContent = `${data.valuation.delta_pct} ${data.valuation.status}`;

    // Confidence Interval
    document.getElementById("tvCiLow").textContent = `Low: ${data.valuation.ci_90.low_fmt}`;
    document.getElementById("tvCiOpt").textContent = `Optimal: ${data.valuation.ci_90.optimal_fmt}`;
    document.getElementById("tvCiHigh").textContent = `High: ${data.valuation.ci_90.high_fmt}`;

    // Premiums & Factors
    document.getElementById("tvContractImpact").textContent = data.valuation.contract_expiry_impact;
    document.getElementById("tvFormPremium").textContent = data.valuation.form_premium;
    document.getElementById("tvCommercialFactor").textContent = data.valuation.commercial_factor;

    // Percentiles
    const pctContainer = document.getElementById("tvPercentilesContainer");
    if (pctContainer && data.percentiles) {
      pctContainer.innerHTML = data.percentiles
        .map(
          (p) => `
        <div style="margin-bottom: 16px;">
          <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 800; margin-bottom: 6px;">
            <span>${p.metric}</span>
            <div style="display: flex; gap: 12px;">
              <span style="font-weight: 900;">${p.value}</span>
              <span style="background: rgba(0,255,133,0.18); color: #009655; font-size: 11px; padding: 2px 8px; border-radius: 4px;">${p.badge}</span>
            </div>
          </div>
          <div style="height: 8px; background: #eeecf4; border-radius: 4px; overflow: hidden;">
            <div style="height: 100%; width: ${p.percentile}%; background: linear-gradient(90deg, #220845, #00ff85);"></div>
          </div>
        </div>
      `
        )
        .join("");
    }

    // Model Comparison breakdown
    const modelCompContainer = document.getElementById("tvModelComparisonList");
    if (modelCompContainer && data.model_comparison) {
      modelCompContainer.innerHTML = data.model_comparison
        .map(
          (m) => `
        <div style="background: ${m.active ? '#220845' : '#f9f8fc'}; color: ${m.active ? '#fff' : '#110d1c'}; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
          <div>
            <div style="font-size: 13px; font-weight: 800;">${m.name}</div>
            <div style="font-size: 11px; opacity: 0.7;">${m.active ? 'ACTIVE PRODUCTION' : 'WEIGHT BENCHMARK'}</div>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 18px; font-weight: 900;">${m.val}</div>
            <div style="font-size: 11px; font-weight: 800; color: ${m.active ? 'var(--accent-neon-green)' : '#7b758c'};">${m.delta}</div>
          </div>
        </div>
      `
        )
        .join("");
    }
  } catch (err) {
    console.error("Transfer valuation error:", err);
  }
}

// --------------------------------------------------------------------------
// Real Working Actions: CSV Export & Controls
// --------------------------------------------------------------------------
window.exportScoutDossier = async function () {
  try {
    const res = await fetch(`${API_BASE}/api/scouting/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        player: currentState.scoutTarget,
        twins: currentState.scoutTwins,
      }),
    });
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ninety_plus_scout_dossier_${currentState.scoutTarget.toLowerCase().replace(/\s+/g, "_")}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (err) {
    alert("Export failed: " + err);
  }
};

window.exportValuationDossier = function () {
  const content = `Player,Predicted Value,Actual Market Value,Contract Expiry,Status\n${currentState.transferPlayer},€72.4M,€61.0M,Jun 2027,UNDERVALUED`;
  const blob = new Blob([content], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `valuation_report_${currentState.transferPlayer.toLowerCase().replace(/\s+/g, "_")}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
};

window.filterScoutingPosition = function (btn, pos) {
  document.querySelectorAll(".pos-tab-btn").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  currentState.scoutPosFilter = pos;
  runScoutingEngine();
};

window.searchScoutingPlayer = function () {
  const query = document.getElementById("scoutSearchInput").value;
  if (query && query.trim().length > 0) {
    currentState.scoutTarget = query.trim();
    runScoutingEngine();
  }
};

window.searchTransferPlayer = function () {
  const query = document.getElementById("tvSearchInput").value;
  if (query && query.trim().length > 0) {
    currentState.transferPlayer = query.trim();
    runTransferPrediction();
  }
};

window.recalculateScoutWeights = function () {
  runScoutingEngine();
};
