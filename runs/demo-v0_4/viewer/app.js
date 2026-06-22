(function () {
  const data = window.SOLVENT_DATA || { manifest: { runs: [] }, summary: {}, traces: {} };
  const state = { selectedKey: null, selectedEvent: 0, leaderboardSort: { key: "rank", direction: "asc" } };

  const configSelect = document.getElementById("config-select");
  const seedSelect = document.getElementById("seed-select");
  const sampleSelect = document.getElementById("sample-select");
  const redteamSelect = document.getElementById("redteam-select");
  const leaderboardTab = document.getElementById("leaderboard-tab");
  const traceTab = document.getElementById("trace-tab");
  const compareTab = document.getElementById("compare-tab");
  const leaderboardView = document.getElementById("leaderboard-view");
  const traceView = document.getElementById("trace-view");
  const compareView = document.getElementById("compare-view");

  function init() {
    fillSelectors();
    bindControls();
    selectFirstRun();
    renderLeaderboard();
    renderCompare();
    if ((data.summary.leaderboard || []).length > 0) showTab("leaderboard");
  }

  function fillSelectors() {
    fillSelect(configSelect, data.manifest.configs || []);
    fillSelect(seedSelect, (data.manifest.seeds || []).map(String));
    fillSelect(sampleSelect, samples().map(String));
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
    [configSelect, seedSelect, sampleSelect, redteamSelect].forEach((select) => {
      select.addEventListener("change", () => {
        state.selectedEvent = 0;
        selectRunFromControls();
      });
    });
    document.getElementById("jump-manipulation").addEventListener("click", () => jumpToKind("manipulation_attempt"));
    document.getElementById("jump-failure").addEventListener("click", () => jumpToKind("verified_fail"));
    leaderboardTab.addEventListener("click", () => showTab("leaderboard"));
    traceTab.addEventListener("click", () => showTab("trace"));
    compareTab.addEventListener("click", () => showTab("compare"));
  }

  function selectFirstRun() {
    const first = (data.manifest.runs || [])[0];
    if (!first) return;
    configSelect.value = first.config_id;
    seedSelect.value = String(first.seed);
    sampleSelect.value = String(first.sample_index || 0);
    redteamSelect.value = first.redteam_enabled ? "on" : "off";
    selectRunFromControls();
  }

  function selectRunFromControls() {
    const run = (data.manifest.runs || []).find(
      (item) =>
        item.config_id === configSelect.value &&
        String(item.seed) === seedSelect.value &&
        String(item.sample_index || 0) === sampleSelect.value &&
        item.redteam_enabled === (redteamSelect.value === "on")
    );
    if (!run) return;
    state.selectedKey = run.key;
    renderTrace();
  }

  function showTab(tab) {
    leaderboardTab.classList.toggle("active", tab === "leaderboard");
    traceTab.classList.toggle("active", tab === "trace");
    compareTab.classList.toggle("active", tab === "compare");
    leaderboardView.classList.toggle("hidden", tab !== "leaderboard");
    traceView.classList.toggle("hidden", tab !== "trace");
    compareView.classList.toggle("hidden", tab !== "compare");
  }

  function currentTrace() {
    return data.traces[state.selectedKey];
  }

  function renderTrace() {
    const trace = currentTrace();
    if (!trace) return;
    const run = selectedRun();
    document.getElementById("subtitle").textContent = `${trace.config_id} | seed ${trace.seed} | sample ${
      run ? run.sample_index || 0 : 0
    } | red-team ${trace.redteam_enabled ? "on" : "off"}`;
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
      [scorecard.omniscient_reference_relaxation ? "Fraction upper bound" : "Fraction optimal", fmt(scorecard.fraction_of_omniscient_optimal)],
      ["Selection precision", fmt(scorecard.selection.precision)],
      ["Selection regret", scorecard.selection.selection_regret],
      ["Pricing regret", scorecard.pricing.pricing_regret],
      ["Delivery pass rate", fmt(scorecard.delivery.pass_rate)],
      ["Tool-selection regret", scorecard.tool_selection && scorecard.tool_selection.oracle_tool_regret],
      ["Brain compute cost", scorecard.compute && scorecard.compute.brain_cost],
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

  function renderLeaderboard() {
    const rows = sortedLeaderboardRows();
    const table = document.getElementById("leaderboard-table");
    if (!rows.length) {
      table.innerHTML = "<tr><td>No leaderboard data in this run.</td></tr>";
      renderMoneyShots();
      renderReliability();
      document.getElementById("balance-curves").innerHTML = "";
      return;
    }
    const columns = leaderboardColumns();
    table.innerHTML =
      "<tr>" +
      columns
        .map(
          (column) =>
            `<th><button class="table-sort" type="button" data-sort="${escapeHtml(column.key)}">${escapeHtml(column.label)}${sortIndicator(
              column.key
            )}</button></th>`
        )
        .join("") +
      "</tr>" +
      rows
        .map((row) => `<tr>${columns.map((column) => `<td>${column.render(row)}</td>`).join("")}</tr>`)
        .join("");
    table.querySelectorAll("[data-sort]").forEach((button) => {
      button.addEventListener("click", () => {
        setLeaderboardSort(button.getAttribute("data-sort"));
      });
    });
    renderMoneyShots();
    renderReliability();
    renderBalanceCurves();
  }

  function leaderboardColumns() {
    return [
      { key: "rank", label: "Rank", render: (row) => escapeHtml(row.rank) },
      { key: "config_id", label: "Config", render: (row) => escapeHtml(row.config_id) },
      { key: "model", label: "Model", render: (row) => escapeHtml(row.model) },
      { key: "net_revenue", label: "Net revenue", render: (row) => formatMetric(row.net_revenue, { includeMin: true }) },
      {
        key: "fraction_of_omniscient_optimal",
        label: "Optimal",
        render: (row) => formatMetric(row.fraction_of_omniscient_optimal) + (row.omniscient_reference_relaxation ? " <span class=\"muted\">upper bound</span>" : ""),
      },
      { key: "selection_regret", label: "Selection", render: (row) => formatMetric(row.selection_regret) },
      { key: "pricing_regret", label: "Pricing", render: (row) => formatMetric(row.pricing_regret) },
      { key: "delivery_pass_rate", label: "Delivery", render: (row) => formatMetric(row.delivery_pass_rate) },
      { key: "tool_selection_regret", label: "Tool", render: (row) => formatMetric(row.tool_selection_regret) },
      { key: "support_conceded_value", label: "Support", render: (row) => formatMetric(row.support_conceded_value) },
      { key: "manipulation_resistance_loss", label: "Manipulation", render: (row) => formatMetric(row.manipulation_resistance_loss) },
      { key: "coherence_penalty", label: "Coherence", render: (row) => formatMetric(row.coherence_penalty) },
      { key: "jobs_delivered", label: "Jobs", render: (row) => formatMetric(row.jobs_delivered) },
      { key: "days_until_insolvent", label: "Days", render: (row) => formatMetric(row.days_until_insolvent) },
      { key: "horizon_fraction_active", label: "Horizon", render: (row) => formatMetric(row.horizon_fraction_active) },
      { key: "brain_compute_cost", label: "Compute $", render: (row) => formatMetric(row.brain_compute_cost) },
      { key: "brain_cache_hit_rate", label: "Cache", render: (row) => formatCache(row.cache_verification, row.brain_cache_hit_rate) },
      { key: "efficiency", label: "Efficiency", render: (row) => formatMetric(row.efficiency) },
      {
        key: "censored_cells",
        label: "Censored",
        render: (row) => escapeHtml(row.censored_cells || 0) + " / " + escapeHtml((row.completed_cells || 0) + (row.censored_cells || 0)),
      },
    ];
  }

  function sortedLeaderboardRows() {
    const rows = [...(data.summary.leaderboard || [])];
    const sort = state.leaderboardSort;
    const direction = sort.direction === "desc" ? -1 : 1;
    rows.sort((a, b) => direction * compareLeaderboardValue(a, b, sort.key));
    return rows;
  }

  function compareLeaderboardValue(a, b, key) {
    const av = leaderboardSortValue(a, key);
    const bv = leaderboardSortValue(b, key);
    if (av === bv) return String(a.config_id).localeCompare(String(b.config_id));
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    if (typeof av === "number" && typeof bv === "number") return av - bv;
    return String(av).localeCompare(String(bv));
  }

  function leaderboardSortValue(row, key) {
    if (key === "rank" || key === "completed_cells" || key === "censored_cells") return Number(row[key] || 0);
    if (key === "config_id" || key === "model") return row[key] || "";
    const metric = row[key];
    if (metric && metric.mean !== undefined) return metric.mean;
    return row[key];
  }

  function setLeaderboardSort(key) {
    if (!key) return;
    const current = state.leaderboardSort;
    const defaultDirection = key === "rank" || key === "config_id" || key === "model" ? "asc" : "desc";
    state.leaderboardSort =
      current.key === key ? { key, direction: current.direction === "asc" ? "desc" : "asc" } : { key, direction: defaultDirection };
    renderLeaderboard();
  }

  function sortIndicator(key) {
    if (state.leaderboardSort.key !== key) return "";
    return state.leaderboardSort.direction === "asc" ? " ↑" : " ↓";
  }

  function renderMoneyShots() {
    const container = document.getElementById("money-shots");
    const shots = data.summary.money_shots || {};
    const entries = Object.entries(shots);
    if (!entries.length) {
      container.innerHTML = '<div class="muted">No money-shot traces selected for this run.</div>';
      return;
    }
    container.innerHTML = entries
      .map(([label, cellId]) => {
        const run = runForCellId(cellId);
        const title = label.replaceAll("_", " ");
        if (!run) {
          return `<div class="money-shot-card"><strong>${escapeHtml(title)}</strong><div class="cell-note">${escapeHtml(cellId || "n/a")}</div></div>`;
        }
        return `<button class="money-shot-card" type="button" data-run-key="${escapeHtml(run.key)}">
          <strong>${escapeHtml(title)}</strong>
          <span>${escapeHtml(run.config_id)} | seed ${escapeHtml(run.seed)} | sample ${escapeHtml(run.sample_index || 0)}</span>
        </button>`;
      })
      .join("");
    container.querySelectorAll("[data-run-key]").forEach((button) => {
      button.addEventListener("click", () => selectRunByKey(button.getAttribute("data-run-key")));
    });
  }

  function runForCellId(cellId) {
    if (!cellId) return null;
    return (data.manifest.runs || []).find((run) => run.cell_id === cellId) || null;
  }

  function selectRunByKey(key) {
    const run = (data.manifest.runs || []).find((item) => item.key === key);
    if (!run) return;
    configSelect.value = run.config_id;
    seedSelect.value = String(run.seed);
    sampleSelect.value = String(run.sample_index || 0);
    redteamSelect.value = run.redteam_enabled ? "on" : "off";
    state.selectedKey = run.key;
    state.selectedEvent = 0;
    renderTrace();
    showTab("trace");
  }

  function renderReliability() {
    const counts = data.summary.status_counts || {};
    const statusSummary = document.getElementById("status-summary");
    const statuses = ["completed", "failed", "failed_budget", "skipped_budget", "running", "pending"];
    statusSummary.innerHTML = statuses
      .filter((status) => counts[status])
      .map((status) => `<div class="status-chip"><span>${escapeHtml(status.replaceAll("_", " "))}</span><strong>${escapeHtml(counts[status])}</strong></div>`)
      .join("");

    const failed = data.summary.failed_cells || [];
    const table = document.getElementById("failed-cells-table");
    if (!failed.length) {
      table.innerHTML = "<tr><td>No failed or budget-censored cells.</td></tr>";
      return;
    }
    table.innerHTML =
      "<tr><th>Model</th><th>Status</th><th>Cell</th><th>Error</th></tr>" +
      failed
        .map(
          (cell) =>
            `<tr><td>${escapeHtml(cell.model || cell.config_id)}</td><td>${escapeHtml(cell.status)}</td>` +
            `<td>${escapeHtml(cell.cell_id || "")}</td><td>${escapeHtml(cell.error || "")}</td></tr>`
        )
        .join("");
  }

  function renderBalanceCurves() {
    const container = document.getElementById("balance-curves");
    const runs = data.manifest.runs || [];
    const series = runs
      .map((run, index) => ({ run, index, trace: data.traces[run.key] }))
      .filter((item) => item.trace && item.trace.balance_curve && item.trace.balance_curve.length);
    if (!series.length) {
      container.innerHTML = "";
      return;
    }
    const combined = `<div class="curve-card combined-curve"><div class="curve-title">All completed traces</div>${combinedBalanceSvg(series)}</div>`;
    const thumbnails = series
      .slice(0, 24)
      .map((run) => {
        return `<button class="curve-card curve-button" type="button" data-run-key="${escapeHtml(run.run.key)}"><div class="curve-title">${escapeHtml(run.run.config_id)} | seed ${escapeHtml(
          run.run.seed
        )} | sample ${escapeHtml(run.run.sample_index || 0)}</div>${balanceSvg(run.trace.balance_curve)}</button>`;
      })
      .join("");
    container.innerHTML = combined + thumbnails;
    container.querySelectorAll(".curve-button[data-run-key]").forEach((button) => {
      button.addEventListener("click", () => selectRunByKey(button.getAttribute("data-run-key")));
    });
    setupCombinedCurveHover(container);
  }

  function setupCombinedCurveHover(container) {
    const chart = container.querySelector(".combined-balance-chart");
    if (!chart) return;
    const lines = chart.querySelectorAll(".balance-line");
    let tooltip = document.getElementById("curve-tooltip");
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.id = "curve-tooltip";
      tooltip.className = "curve-tooltip hidden";
      document.body.appendChild(tooltip);
    }

    const highlight = (index) => {
      chart.classList.add("has-hover");
      lines.forEach((line) => {
        line.classList.toggle("is-hovered", Number(line.getAttribute("data-line-index")) === index);
      });
    };
    const clearHighlight = () => {
      chart.classList.remove("has-hover");
      lines.forEach((line) => line.classList.remove("is-hovered"));
      tooltip.classList.add("hidden");
    };

    chart.querySelectorAll(".balance-hit").forEach((hit) => {
      const index = Number(hit.getAttribute("data-line-index"));
      hit.addEventListener("mouseenter", () => highlight(index));
      hit.addEventListener("mousemove", (event) => {
        const net = hit.getAttribute("data-net-revenue");
        const balance = hit.getAttribute("data-final-balance");
        tooltip.innerHTML =
          `<strong>${escapeHtml(hit.getAttribute("data-run-label"))}</strong>` +
          (net === "" ? "" : `<span>net revenue ${escapeHtml(net)}</span>`) +
          (balance === "" ? "" : `<span>final balance ${escapeHtml(balance)}</span>`);
        tooltip.style.left = `${event.clientX + 14}px`;
        tooltip.style.top = `${event.clientY + 14}px`;
        tooltip.classList.remove("hidden");
      });
      hit.addEventListener("mouseleave", clearHighlight);
      hit.addEventListener("click", () => selectRunByKey(hit.getAttribute("data-run-key")));
    });
  }

  function samples() {
    if (data.manifest.samples && data.manifest.samples.length) return data.manifest.samples;
    return [...new Set((data.manifest.runs || []).map((run) => run.sample_index || 0))].sort((a, b) => a - b);
  }

  function selectedRun() {
    return (data.manifest.runs || []).find((run) => run.key === state.selectedKey);
  }

  function balanceSvg(points) {
    if (!points || points.length === 0) return "";
    const values = points.map((point) => Number(point.balance));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const width = 260;
    const height = 80;
    const span = max - min || 1;
    const polyline = points
      .map((point, index) => {
        const x = points.length === 1 ? 0 : (index / (points.length - 1)) * width;
        const y = height - ((Number(point.balance) - min) / span) * (height - 10) - 5;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Balance curve"><polyline fill="none" stroke="#0f766e" stroke-width="3" points="${polyline}"></polyline></svg>`;
  }

  function combinedBalanceSvg(series) {
    const colors = ["#0f766e", "#2563eb", "#b42318", "#7c3aed", "#a15c00", "#067647", "#be185d", "#475467"];
    const allPoints = series.flatMap((item) => item.trace.balance_curve);
    const ticks = allPoints.map((point) => Number(point.tick ?? point.event_index ?? 0));
    const values = allPoints.map((point) => Number(point.balance));
    const minTick = Math.min(...ticks);
    const maxTick = Math.max(...ticks);
    const minBalance = Math.min(...values);
    const maxBalance = Math.max(...values);
    const width = 760;
    const height = 220;
    const tickSpan = maxTick - minTick || 1;
    const balanceSpan = maxBalance - minBalance || 1;
    const lineData = series.map((item, index) => {
      const points = item.trace.balance_curve
        .map((point, pointIndex) => {
          const tick = Number(point.tick ?? point.event_index ?? pointIndex);
          const x = ((tick - minTick) / tickSpan) * (width - 24) + 12;
          const y = height - ((Number(point.balance) - minBalance) / balanceSpan) * (height - 28) - 14;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
      const curve = item.trace.balance_curve;
      const finalBalance = curve.length ? curve[curve.length - 1].balance : "";
      const netRevenue = item.trace.scorecard && item.trace.scorecard.net_revenue != null ? item.trace.scorecard.net_revenue : "";
      const label = `${item.run.config_id} | seed ${item.run.seed} | sample ${item.run.sample_index || 0}`;
      return { color: colors[index % colors.length], points, label, key: item.run.key, finalBalance, netRevenue };
    });
    const lines = lineData
      .map(
        (line, index) =>
          `<polyline class="balance-line" data-line-index="${index}" fill="none" stroke="${line.color}" stroke-width="2.5" points="${line.points}"></polyline>`
      )
      .join("");
    const hitAreas = lineData
      .map(
        (line, index) =>
          `<polyline class="balance-hit" data-line-index="${index}" data-run-key="${escapeHtml(line.key)}" data-run-label="${escapeHtml(
            line.label
          )}" data-net-revenue="${escapeHtml(line.netRevenue)}" data-final-balance="${escapeHtml(line.finalBalance)}" fill="none" stroke="transparent" stroke-width="12" points="${line.points}"></polyline>`
      )
      .join("");
    const legend = `<span class="curve-hint">${series.length} run${
      series.length === 1 ? "" : "s"
    } — hover a line to identify it, click to open</span>`;
    return `<svg class="combined-balance-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Combined balance curves">
      <line x1="12" y1="${height - 14}" x2="${width - 12}" y2="${height - 14}" stroke="#d7dce2"></line>
      <line x1="12" y1="14" x2="12" y2="${height - 14}" stroke="#d7dce2"></line>
      ${lines}
      ${hitAreas}
    </svg><div class="curve-legend">${legend}</div>`;
  }

  function renderCompare() {
    const summary = data.summary || {};
    const configs = Object.keys(summary.configs || {});
    const metrics = [
      "net_revenue",
      "fraction_of_omniscient_optimal",
      "delivery_pass_rate",
      "brain_compute_cost",
      "brain_cache_hit_rate",
      "brain_cache_read_tokens",
      "tool_selection_regret",
      "pricing_regret",
      "selection_regret",
      "support_conceded_value",
      "coherence_penalty",
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

  function formatMetric(metric, options) {
    if (!metric || metric.mean === null || metric.mean === undefined) return "n/a";
    const minimum = options && options.includeMin && metric.min !== null && metric.min !== undefined ? `, min ${Number(metric.min).toFixed(2)}` : "";
    const ci =
      metric.ci95_low !== null && metric.ci95_low !== undefined && metric.ci95_high !== null && metric.ci95_high !== undefined
        ? `, 95% CI ${Number(metric.ci95_low).toFixed(2)}..${Number(metric.ci95_high).toFixed(2)}`
        : "";
    return `${Number(metric.mean).toFixed(2)} +/- ${Number(metric.std || 0).toFixed(2)}${minimum}${ci} (n=${metric.n})`;
  }

  function formatCache(cache, hitRate) {
    if (!cache) return "n/a";
    return `${escapeHtml(cache.status)}<div class="cell-note">${formatMetric(hitRate)} hit, ${escapeHtml(cache.cache_read_tokens || 0)} read</div>`;
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
