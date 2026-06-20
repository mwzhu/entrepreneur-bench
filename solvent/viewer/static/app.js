(function () {
  const data = window.SOLVENT_DATA || { manifest: { runs: [] }, summary: {}, traces: {} };
  const state = { selectedKey: null, selectedEvent: 0 };

  const configSelect = document.getElementById("config-select");
  const seedSelect = document.getElementById("seed-select");
  const redteamSelect = document.getElementById("redteam-select");
  const traceTab = document.getElementById("trace-tab");
  const compareTab = document.getElementById("compare-tab");
  const traceView = document.getElementById("trace-view");
  const compareView = document.getElementById("compare-view");

  function init() {
    fillSelectors();
    bindControls();
    selectFirstRun();
    renderCompare();
  }

  function fillSelectors() {
    fillSelect(configSelect, data.manifest.configs || []);
    fillSelect(seedSelect, (data.manifest.seeds || []).map(String));
    fillSelect(redteamSelect, ["off", "on"]);
  }

  function fillSelect(select, values) {
    select.innerHTML = "";
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });
  }

  function bindControls() {
    [configSelect, seedSelect, redteamSelect].forEach((select) => {
      select.addEventListener("change", () => {
        state.selectedEvent = 0;
        selectRunFromControls();
      });
    });
    document.getElementById("jump-manipulation").addEventListener("click", () => jumpToKind("manipulation_attempt"));
    document.getElementById("jump-failure").addEventListener("click", () => jumpToKind("verified_fail"));
    traceTab.addEventListener("click", () => showTab("trace"));
    compareTab.addEventListener("click", () => showTab("compare"));
  }

  function selectFirstRun() {
    const first = (data.manifest.runs || [])[0];
    if (!first) return;
    configSelect.value = first.config_id;
    seedSelect.value = String(first.seed);
    redteamSelect.value = first.redteam_enabled ? "on" : "off";
    selectRunFromControls();
  }

  function selectRunFromControls() {
    const run = (data.manifest.runs || []).find(
      (item) =>
        item.config_id === configSelect.value &&
        String(item.seed) === seedSelect.value &&
        item.redteam_enabled === (redteamSelect.value === "on")
    );
    if (!run) return;
    state.selectedKey = run.key;
    renderTrace();
  }

  function showTab(tab) {
    traceTab.classList.toggle("active", tab === "trace");
    compareTab.classList.toggle("active", tab === "compare");
    traceView.classList.toggle("hidden", tab !== "trace");
    compareView.classList.toggle("hidden", tab !== "compare");
  }

  function currentTrace() {
    return data.traces[state.selectedKey];
  }

  function renderTrace() {
    const trace = currentTrace();
    if (!trace) return;
    document.getElementById("subtitle").textContent = `${trace.config_id} | seed ${trace.seed} | red-team ${
      trace.redteam_enabled ? "on" : "off"
    }`;
    renderEvents(trace);
    renderEventDetail(trace, trace.events[state.selectedEvent] || trace.events[0]);
    renderScorecard(trace.scorecard);
    renderBalance(trace.balance_curve, state.selectedEvent);
  }

  function renderEvents(trace) {
    const list = document.getElementById("event-list");
    list.innerHTML = "";
    trace.events.forEach((event) => {
      const item = document.createElement("li");
      item.className = "event-item" + (event.index === state.selectedEvent ? " active" : "");
      item.innerHTML = `<strong>${escapeHtml(event.title)}</strong><div>${escapeHtml(event.summary || "")}</div>
        <div class="event-meta"><span>tick ${escapeHtml(event.tick)}</span><span>burn ${escapeHtml(event.burn_delta)}</span><span>bal ${escapeHtml(event.balance_after)}</span></div>`;
      item.addEventListener("click", () => {
        state.selectedEvent = event.index;
        renderTrace();
      });
      list.appendChild(item);
    });
  }

  function renderEventDetail(trace, event) {
    if (!event) return;
    document.getElementById("event-title").textContent = event.title;
    document.getElementById("event-summary").textContent = event.summary || "";
    document.getElementById("payload-json").textContent = JSON.stringify(event.payload, null, 2);

    const artifactBlock = document.getElementById("artifact-block");
    if (event.artifact_preview) {
      artifactBlock.classList.remove("hidden");
      document.getElementById("artifact-preview").textContent =
        event.artifact_preview + (event.artifact_truncated ? "\n[truncated]" : "");
    } else {
      artifactBlock.classList.add("hidden");
    }

    const verifyBlock = document.getElementById("verify-block");
    if (event.verify) {
      verifyBlock.classList.remove("hidden");
      document.getElementById("verify-summary").innerHTML = `<span class="pill ${event.verify.passed ? "good" : "bad"}">${
        event.verify.passed ? "passed" : "failed"
      }</span> score ${event.verify.score}`;
      renderVerifyTable(event.verify.checks || []);
    } else {
      verifyBlock.classList.add("hidden");
    }
    renderBalance(trace.balance_curve, event.index);
  }

  function renderVerifyTable(checks) {
    const table = document.getElementById("verify-table");
    table.innerHTML = "<tr><th>Check</th><th>Status</th><th>Message</th></tr>";
    checks.forEach((check) => {
      const row = document.createElement("tr");
      row.innerHTML = `<td>${escapeHtml(check.name)}</td><td><span class="pill ${check.passed ? "good" : "bad"}">${
        check.passed ? "pass" : "fail"
      }</span></td><td>${escapeHtml(check.message || "")}</td>`;
      table.appendChild(row);
    });
  }

  function renderScorecard(scorecard) {
    const rows = [
      ["Net revenue", scorecard.net_revenue],
      ["Fraction optimal", fmt(scorecard.fraction_of_omniscient_optimal)],
      ["Selection precision", fmt(scorecard.selection.precision)],
      ["Selection regret", scorecard.selection.selection_regret],
      ["Pricing regret", scorecard.pricing.pricing_regret],
      ["Delivery pass rate", fmt(scorecard.delivery.pass_rate)],
      ["Manipulation conceded", scorecard.support.conceded_value],
      ["Coherence penalty", scorecard.coherence.coherence_penalty],
    ];
    document.getElementById("scorecard").innerHTML =
      '<table class="score-table">' +
      rows.map(([label, value]) => `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(String(value ?? "n/a"))}</td></tr>`).join("") +
      "</table>";
  }

  function renderBalance(points, selectedIndex) {
    const chart = document.getElementById("balance-chart");
    if (!points || points.length === 0) {
      chart.innerHTML = "";
      return;
    }
    const values = points.map((point) => Number(point.balance));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const width = 420;
    const height = 140;
    const span = max - min || 1;
    const coords = points.map((point, index) => {
      const x = points.length === 1 ? 0 : (index / (points.length - 1)) * width;
      const y = height - ((Number(point.balance) - min) / span) * (height - 10) - 5;
      return { x, y, point };
    });
    const polyline = coords.map(({ x, y }) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    const selected = coords[Math.min(selectedIndex, coords.length - 1)];
    chart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Balance curve">
      <polyline fill="none" stroke="#0f766e" stroke-width="3" points="${polyline}"></polyline>
      <circle cx="${selected.x.toFixed(1)}" cy="${selected.y.toFixed(1)}" r="5" fill="#b42318"></circle>
    </svg>`;
  }

  function renderCompare() {
    const summary = data.summary || {};
    const configs = Object.keys(summary.configs || {});
    const metrics = [
      "net_revenue",
      "fraction_of_omniscient_optimal",
      "delivery_pass_rate",
      "pricing_regret",
      "selection_regret",
      "manipulation_resistance_loss",
    ];
    const labels = summary.metric_labels || {};
    const table = document.getElementById("compare-table");
    table.innerHTML =
      "<tr><th>Metric</th>" +
      configs.map((config) => `<th>${escapeHtml(config)}</th>`).join("") +
      "<th>Paired delta</th></tr>" +
      metrics
        .map((metric) => {
          const label = labels[metric] || metric.replaceAll("_", " ");
          const values = configs
            .map((config) => `<td>${formatMetric(summary.configs[config] && summary.configs[config][metric])}</td>`)
            .join("");
          return `<tr><td>${escapeHtml(label)}</td>${values}<td>${formatMetric(summary.paired_delta && summary.paired_delta[metric])}</td></tr>`;
        })
        .join("");
  }

  function jumpToKind(kind) {
    const trace = currentTrace();
    if (!trace) return;
    const event = trace.events.find((item) => item.kind === kind);
    if (!event) return;
    state.selectedEvent = event.index;
    renderTrace();
  }

  function formatMetric(metric) {
    if (!metric || metric.mean === null || metric.mean === undefined) return "n/a";
    return `${Number(metric.mean).toFixed(2)} +/- ${Number(metric.std || 0).toFixed(2)} (n=${metric.n})`;
  }

  function fmt(value) {
    return value === null || value === undefined ? "n/a" : Number(value).toFixed(2);
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  init();
})();
