(() => {
  "use strict";

  const API = {
    summary: "/api/summary",
    demo: "/api/demo",
    logs: "/api/logs",
  };

  const SVG_NS = "http://www.w3.org/2000/svg";
  const HEALTH_STALE_AFTER_S = 30;
  const ALLOWED_INTERVALS = [2000, 5000, 10000];

  const TONES = ["ok", "warn", "crit", "info", "neutral"];
  const TASK_STATES = {
    in_progress: ["В работе", "ok"],
    pending: ["Ожидает", "warn"],
    waiting: ["Ожидает", "warn"],
    done: ["Завершена", "neutral"],
    completed: ["Завершена", "neutral"],
    failed: ["Ошибка", "crit"],
    error: ["Ошибка", "crit"],
    stopped: ["Остановлена", "warn"],
  };
  const KIND_LABELS = {
    decision: "Решение",
    action: "Действие",
    verification: "Проверка",
    chat: "Чат",
    error: "Ошибка",
    observation: "Наблюдение",
    other: "Событие",
  };
  const LEVEL_LABELS = {
    critical: "Критично",
    warning: "Внимание",
    info: "Инфо",
  };
  const ROLE_LABELS = {
    operator: "оператор",
    agent: "агент",
    system: "система",
  };

  const $ = (id) => document.getElementById(id);

  const refs = {
    statusPill: $("status-pill"),
    statusLabel: $("status-label"),
    stalePill: $("stale-pill"),
    connPill: $("conn-pill"),
    demoBadge: $("demo-badge"),
    runId: $("run-id"),
    roleTeam: $("role-team"),
    runMode: $("run-mode"),
    lastUpdate: $("last-update"),
    interval: $("refresh-interval"),
    logButton: $("log-button"),
    demoButton: $("demo-button"),
    demoButtonLabel: $("demo-button-label"),
    refreshButton: $("refresh-button"),

    errorBanner: $("error-banner"),
    errorDetail: $("error-detail"),
    errorRetry: $("error-retry"),
    connectBanner: $("connect-banner"),

    taskCard: document.querySelector(".task-card"),
    taskTitle: $("task-title"),
    taskStatus: $("task-status"),
    taskRunMeta: $("task-run-meta"),
    taskText: $("task-text"),
    runStarted: $("run-started"),
    runAgent: $("run-agent"),
    runModels: $("run-models"),
    kpiGrid: document.querySelector(".kpi-grid"),

    noLogState: $("no-log-state"),
    noLogDemo: $("no-log-demo"),

    decisionPanel: $("decision-panel"),
    decisionMeta: $("decision-meta"),
    decisionSource: $("decision-source"),
    decisionBody: $("decision-body"),

    alertsPanel: $("alerts-panel"),
    alertsCount: $("alerts-count"),
    alertsBody: $("alerts-body"),

    timelinePanel: $("timeline-panel"),
    timelineCount: $("timeline-count"),
    timelineFilters: $("timeline-filters"),
    timelineList: $("timeline-list"),

    messagesPanel: $("messages-panel"),
    messagesCount: $("messages-count"),
    messagesList: $("messages-list"),

    actionsPanel: $("actions-panel"),
    actionsCount: $("actions-count"),
    actionsBody: $("actions-body"),

    statePanel: $("state-panel"),
    stateMeta: $("state-meta"),
    stateBody: $("state-body"),

    footerSource: $("footer-source"),

    logDialog: $("log-dialog"),
    logMeta: $("log-meta"),
    logClose: $("log-close"),
    logKind: $("log-kind"),
    logReload: $("log-reload"),
    logStatus: $("log-status"),
    logPre: $("log-pre"),
  };

  const app = {
    data: null,
    hasData: false,
    demo: false,
    demoBusy: false,
    intervalMs: readStoredInterval(),
    timer: null,
    seq: 0,
    logSeq: 0,
    fetchAbort: null,
    logAbort: null,
    online: true,
    busy: false,
    activeFilter: "all",
    lastGoodAt: null,
  };

  const numberFmt = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 });
  const numberFmt1 = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 });
  const timeFmt = new Intl.DateTimeFormat("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const dateTimeFmt = new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
  const fullFmt = new Intl.DateTimeFormat("ru-RU", {
    dateStyle: "short",
    timeStyle: "medium",
  });

  /* ---------- Small utilities ---------- */

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function icon(name, className = "icon") {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", className);
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", "#i-" + name);
    svg.appendChild(use);
    return svg;
  }

  function setText(node, value) {
    if (!node) return;
    const next = value === undefined || value === null ? "" : String(value);
    if (node.textContent !== next) node.textContent = next;
  }

  function setHidden(node, hidden) {
    if (!node) return;
    if (node.hidden !== Boolean(hidden)) node.hidden = Boolean(hidden);
  }

  function clear(node) {
    if (node) node.replaceChildren();
  }

  function textValue(value, fallback = "—") {
    if (value === null || value === undefined || value === "") return fallback;
    return String(value);
  }

  function displayValue(value, fallback = "—") {
    if (value === null || value === undefined || value === "") return fallback;
    if (typeof value === "object") {
      try {
        return JSON.stringify(value);
      } catch {
        return fallback;
      }
    }
    return String(value);
  }

  function numeric(value) {
    if (value === null || value === undefined || value === "") return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function fmtNumber(value, digits = 0) {
    const n = numeric(value);
    if (n === null) return "—";
    return (digits > 0 ? numberFmt1 : numberFmt).format(n);
  }

  function parseDate(value) {
    if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value;
    if (value === null || value === undefined || value === "") return null;
    if (typeof value === "number") {
      const ms = Math.abs(value) < 1e12 ? value * 1000 : value;
      const fromNumber = new Date(ms);
      return Number.isNaN(fromNumber.getTime()) ? null : fromNumber;
    }
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function formatTime(value) {
    const date = parseDate(value);
    if (!date) return "—";
    return timeFmt.format(date);
  }

  function formatWhen(value) {
    const date = parseDate(value);
    if (!date) return "—";
    const now = new Date();
    const sameDay =
      date.getFullYear() === now.getFullYear() &&
      date.getMonth() === now.getMonth() &&
      date.getDate() === now.getDate();
    return sameDay ? timeFmt.format(date) : dateTimeFmt.format(date);
  }

  function fullTime(value) {
    const date = parseDate(value);
    return date ? fullFmt.format(date) : "Время неизвестно";
  }

  function formatAgeSeconds(seconds) {
    const n = Number(seconds);
    if (!Number.isFinite(n)) return "—";
    if (n < 1) return "только что";
    if (n < 60) return `${Math.floor(n)} с назад`;
    if (n < 3600) return `${Math.floor(n / 60)} мин назад`;
    if (n < 86400) return `${Math.floor(n / 3600)} ч назад`;
    if (n < 2592000) return `${Math.floor(n / 86400)} дн назад`;
    return `${Math.floor(n / 2592000)} мес назад`;
  }

  function relativeAge(value) {
    const date = parseDate(value);
    if (!date) return "—";
    const diff = (Date.now() - date.getTime()) / 1000;
    if (diff < -2) {
      const ahead = Math.abs(diff);
      if (ahead < 60) return `через ${Math.floor(ahead)} с`;
      if (ahead < 3600) return `через ${Math.floor(ahead / 60)} мин`;
      return `через ${Math.floor(ahead / 3600)} ч`;
    }
    return formatAgeSeconds(Math.max(0, diff));
  }

  function arrayOf(value) {
    return Array.isArray(value) ? value : [];
  }

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function compareTimeAsc(a, b) {
    const at = parseDate(a && a.ts);
    const bt = parseDate(b && b.ts);
    const av = at ? at.getTime() : 0;
    const bv = bt ? bt.getTime() : 0;
    if (av !== bv) return av - bv;
    return (Number(a && a.seq) || 0) - (Number(b && b.seq) || 0);
  }

  function compareTimeDesc(a, b) {
    return -compareTimeAsc(a, b);
  }

  function fingerprint(value) {
    let source;
    try {
      source = JSON.stringify(value);
    } catch {
      source = String(value);
    }
    if (source === undefined) source = "undefined";
    let hash = 5381;
    for (let i = 0; i < source.length; i += 1) {
      hash = ((hash << 5) + hash + source.charCodeAt(i)) | 0;
    }
    return String(hash);
  }

  function flash(node) {
    if (!node) return;
    node.classList.remove("is-updated");
    void node.offsetWidth;
    node.classList.add("is-updated");
  }

  function shouldRender(node, key, animate = true) {
    if (!node) return false;
    if (node.dataset.fp === key) return false;
    const hadPrevious = node.dataset.fp !== undefined;
    node.dataset.fp = key;
    if (hadPrevious && animate) flash(node);
    return true;
  }

  function tagNode(text, tone = "neutral") {
    const tag = el("span", `tag tag--${tone}`, text);
    return tag;
  }

  function statusLine(tone, text, iconName) {
    const line = el("span", "status-line");
    line.dataset.tone = tone;
    line.appendChild(icon(iconName));
    line.appendChild(el("span", null, text));
    return line;
  }

  function emptyState(options) {
    const {
      iconName = "info",
      title,
      text,
      tone = "neutral",
      actionLabel = null,
      onAction = null,
    } = options || {};
    const wrap = el("div", "empty");
    const iconBox = el(
      "span",
      tone === "ok" ? "empty__icon empty__icon--ok" : "empty__icon",
    );
    iconBox.appendChild(icon(iconName));
    wrap.appendChild(iconBox);
    if (title) wrap.appendChild(el("h3", null, title));
    if (text) wrap.appendChild(el("p", null, text));
    if (actionLabel && typeof onAction === "function") {
      const button = el("button", "btn btn--small btn--ghost");
      button.type = "button";
      button.appendChild(icon("refresh"));
      button.appendChild(el("span", null, actionLabel));
      button.addEventListener("click", onAction);
      wrap.appendChild(button);
    }
    return wrap;
  }

  function readStoredInterval() {
    try {
      const stored = Number(window.localStorage.getItem("dome.ops.interval"));
      if (ALLOWED_INTERVALS.includes(stored)) return stored;
    } catch {
      /* Storage can be unavailable in private/offline contexts. */
    }
    return 2000;
  }

  function storeInterval(value) {
    try {
      window.localStorage.setItem("dome.ops.interval", String(value));
    } catch {
      /* Non-critical preference. */
    }
  }

  function setBusy(busy) {
    app.busy = Boolean(busy);
    refs.refreshButton.classList.toggle("is-loading", app.busy);
    refs.refreshButton.setAttribute("aria-busy", String(app.busy));
  }

  function setOnline(online) {
    app.online = Boolean(online);
    setHidden(refs.connPill, app.online);
    renderFooter();
  }

  function describeError(error) {
    if (!error) return "Неизвестная ошибка.";
    if (error.name === "AbortError") return "Запрос отменён.";
    if (error instanceof TypeError) {
      return "Сетевой запрос не выполнен. Проверьте, запущен ли backend.";
    }
    return error.message ? String(error.message) : "Неизвестная ошибка.";
  }

  /* ---------- Header / status ---------- */

  function resolveHealth(data) {
    const health = asObject(data && data.health);
    const run = asObject(data && data.run);
    const status = String(health.status || "").toLowerCase();
    const age = numeric(health.last_record_age_s);

    if (run.stopped === true || status === "stopped") {
      return { label: "STOPPED", tone: "neutral", stale: false, hint: "Смена остановлена" };
    }
    if (status === "error") {
      return { label: "ERROR", tone: "crit", stale: false, hint: "Агент сообщает об ошибке" };
    }
    if (status === "stale") {
      return { label: "STALE", tone: "warn", stale: true, hint: "Данные устарели" };
    }
    if (status === "waiting" || health.has_log === false) {
      return { label: "WAITING", tone: "warn", stale: false, hint: "Ожидание журнала смены" };
    }
    if (status === "ok") {
      if (age !== null && age > HEALTH_STALE_AFTER_S) {
        return { label: "STALE", tone: "warn", stale: true, hint: "Данные устарели" };
      }
      return { label: "LIVE", tone: "ok", stale: false, hint: "Данные поступают" };
    }
    if (age !== null && age > HEALTH_STALE_AFTER_S) {
      return { label: "STALE", tone: "warn", stale: true, hint: "Данные устарели" };
    }
    return { label: "WAITING", tone: "warn", stale: false, hint: "Ожидание данных" };
  }

  function setStatus(label, tone, hint) {
    setText(refs.statusLabel, label);
    const classes = ["pill"].concat(TONES.map((item) => `pill--${item}`));
    refs.statusPill.classList.remove(...classes);
    refs.statusPill.classList.add(`pill--${tone}`);
    refs.statusPill.title = hint || "";
    refs.statusPill.setAttribute(
      "aria-label",
      `Статус: ${label}${hint ? ". " + hint : ""}`,
    );
  }

  function renderDemoUi() {
    document.body.dataset.demo = String(app.demo);
    setHidden(refs.demoBadge, !app.demo);
    refs.demoButton.setAttribute("aria-pressed", String(app.demo));
    setText(refs.demoButtonLabel, app.demo ? "Выйти из демо" : "Показать демо");
    const use = refs.demoButton.querySelector("use");
    if (use) use.setAttribute("href", app.demo ? "#i-x" : "#i-play");
    document.title = app.demo
      ? "Dome Ops — демо-данные"
      : "Dome Ops — наблюдение за сменой";
    renderFooter();
  }

  function renderHeader(data) {
    const health = asObject(data && data.health);
    const run = asObject(data && data.run);
    const resolved = resolveHealth(data);

    setStatus(resolved.label, resolved.tone, resolved.hint);
    setHidden(refs.stalePill, !resolved.stale);
    setHidden(refs.connPill, app.online);
    setHidden(refs.demoBadge, !app.demo);

    const runId = textValue(run.run_id);
    setText(refs.runId, runId);
    refs.runId.title = run.run_id ? String(run.run_id) : "Идентификатор запуска неизвестен";

    const role = run.role ? String(run.role) : "";
    const team = run.team !== null && run.team !== undefined ? String(run.team) : "";
    setText(refs.roleTeam, role ? `${role}${team ? " / " + team : ""}` : "—");

    const config = asObject(run.config);
    setText(refs.runMode, textValue(config.mode));

    if (data && data.generated_at) {
      setText(refs.lastUpdate, `обновлено ${relativeAge(data.generated_at)}`);
      refs.lastUpdate.title = `Снимок: ${fullTime(data.generated_at)}`;
    } else {
      setText(refs.lastUpdate, "время неизвестно");
      refs.lastUpdate.title = "";
    }

    if (!app.online && app.lastGoodAt) {
      setText(refs.lastUpdate, `нет связи · данные ${relativeAge(app.lastGoodAt)}`);
      refs.lastUpdate.title = `Последние данные: ${fullTime(app.lastGoodAt)}`;
    }
  }

  /* ---------- Task ---------- */

  function taskState(status) {
    const key = String(status || "").toLowerCase();
    if (TASK_STATES[key]) return TASK_STATES[key];
    if (status) return [String(status), "neutral"];
    return ["Нет задачи", "neutral"];
  }

  function renderTask(data) {
    const state = asObject(data && data.state);
    const task = asObject(state.task);
    const run = asObject(data && data.run);
    const health = asObject(data && data.health);
    const observation = asObject(data && data.observation);

    const [statusLabel, statusTone] = task ? taskState(task.status) : ["Нет задачи", "neutral"];
    setText(refs.taskStatus, statusLabel);
    refs.taskStatus.className = `tag tag--${statusTone}`;

    const domeTime = observation.dome_time ? `Dome ${observation.dome_time}` : "";
    setText(
      refs.taskRunMeta,
      domeTime || (health.status === "ok" ? "телеметрия поступает" : "ожидание телеметрии"),
    );

    setText(refs.taskText, task.text || "Задача не задана. Агент ожидает постановки от оператора.");

    const agent = asObject(run.agent);
    const models = arrayOf(agent.models).filter(Boolean);
    const agentName = textValue(agent.name);
    const agentVersion = agent.version ? `v${agent.version}` : "";
    setText(refs.runStarted, formatWhen(run.started_at));
    refs.runStarted.title = run.started_at ? fullTime(run.started_at) : "";
    setText(refs.runAgent, agentVersion ? `${agentName} · ${agentVersion}` : agentName);
    setText(refs.runModels, models.length ? models.join(", ") : "—");
    refs.runModels.title = models.join(", ");

    if (
      shouldRender(
        refs.taskCard,
        fingerprint([state.task, run, observation.dome_time, health.status]),
      )
    ) {
      /* Text nodes above are already updated; the animation marks the refresh. */
    }
  }

  /* ---------- KPI ---------- */

  function machineTone(value) {
    const raw = String(value || "").toUpperCase();
    if (!raw) return "neutral";
    if (/(ERROR|FAULT|FAIL|OFFLINE|ALARM|BLOCK)/.test(raw)) return "crit";
    if (/(PRINT|RUN|ACTIVE|BUSY|ON|WORK)/.test(raw)) return "ok";
    if (/(IDLE|STANDBY|PAUSE|OFF|READY)/.test(raw)) return "neutral";
    return "info";
  }

  function machineState(value, kind) {
    const raw = String(value || "").toUpperCase();
    if (!raw) return "нет данных";
    if (/(ERROR|FAULT|FAIL|ALARM|BLOCK)/.test(raw)) return "ошибка";
    if (/(OFFLINE|LOST)/.test(raw)) return "нет связи";
    if (kind === "printer") {
      if (/PRINT/.test(raw)) return "печатает";
      if (/IDLE|READY/.test(raw)) return "простой";
      if (/PAUSE/.test(raw)) return "пауза";
    }
    if (kind === "cnc") {
      if (/RUN|WORK|ACTIVE/.test(raw)) return "работает";
      if (/IDLE|READY/.test(raw)) return "простой";
      if (/PAUSE/.test(raw)) return "пауза";
    }
    return String(value);
  }

  function metric(data, key) {
    const metrics = asObject(data && data.metrics);
    const observation = asObject(data && data.observation);
    const observationMetrics = asObject(observation.metrics);
    if (metrics[key] !== undefined && metrics[key] !== null) return metrics[key];
    if (observationMetrics[key] !== undefined && observationMetrics[key] !== null) {
      return observationMetrics[key];
    }
    return null;
  }

  function setKpi(name, options) {
    const card = $(`kpi-${name}`);
    if (!card) return;
    const valueNode = $(`kpi-${name}-value`);
    const stateNode = $(`kpi-${name}-state`);
    card.dataset.tone = options.tone || "neutral";
    setText(valueNode, options.value);
    setText(stateNode, options.state);
    if (options.title) card.title = options.title;
  }

  function renderKpis(data) {
    const printer = metric(data, "printer");
    const cnc = metric(data, "cnc");
    const co2 = numeric(metric(data, "co2_ppm"));
    const soc = numeric(metric(data, "soc_pct"));
    const fume = metric(data, "fume");
    const power = numeric(metric(data, "power_w"));

    setKpi("printer", {
      value: textValue(printer, "—"),
      tone: machineTone(printer),
      state: machineState(printer, "printer"),
      title: "Текущее состояние 3D-принтера",
    });

    setKpi("cnc", {
      value: textValue(cnc, "—"),
      tone: machineTone(cnc),
      state: machineState(cnc, "cnc"),
      title: "Текущее состояние CNC-станка",
    });

    let co2Tone = "neutral";
    let co2State = "нет данных";
    if (co2 !== null) {
      if (co2 >= 1600) {
        co2Tone = "crit";
        co2State = "критично";
      } else if (co2 >= 1200) {
        co2Tone = "warn";
        co2State = "выше нормы";
      } else {
        co2Tone = "ok";
        co2State = "в норме";
      }
    }
    setKpi("co2", {
      value: fmtNumber(co2),
      tone: co2Tone,
      state: co2State,
      title: "Порог внимания — 1200 ppm, критический — 1600 ppm",
    });

    let socTone = "neutral";
    let socState = "нет данных";
    if (soc !== null) {
      if (soc <= 20) {
        socTone = "crit";
        socState = "критический заряд";
      } else if (soc <= 40) {
        socTone = "warn";
        socState = "низкий заряд";
      } else {
        socTone = "ok";
        socState = "заряд в норме";
      }
    }
    setKpi("battery", {
      value: soc !== null ? fmtNumber(soc) : "—",
      tone: socTone,
      state: socState,
      title: "Порог внимания — 40%, критический — 20%",
    });
    const batteryFill = $("kpi-battery-fill");
    if (batteryFill) {
      const width = soc === null ? 0 : Math.min(100, Math.max(0, soc));
      batteryFill.style.width = `${width}%`;
    }

    const fumeOn = /^(ON|1|TRUE|ACTIVE|RUNNING)$/i.test(String(fume || ""));
    const printerActive = /PRINT|RUN|ACTIVE|BUSY/i.test(String(printer || ""));
    let fumeTone = "neutral";
    let fumeState = "нет данных";
    if (fume !== null && fume !== undefined && fume !== "") {
      if (fumeOn) {
        fumeTone = "ok";
        fumeState = "включена";
      } else if (printerActive) {
        fumeTone = "crit";
        fumeState = "выключена при печати";
      } else {
        fumeTone = "warn";
        fumeState = "выключена";
      }
    }
    setKpi("fume", {
      value: textValue(fume, "—"),
      tone: fumeTone,
      state: fumeState,
      title: "Состояние вытяжки; выключенная вытяжка при печати — критично",
    });

    setKpi("power", {
      value: fmtNumber(power),
      tone: power === null ? "neutral" : "info",
      state: power === null ? "нет данных" : "текущая нагрузка",
      title: "Текущее потребление мощности",
    });

    shouldRender(
      refs.kpiGrid,
      fingerprint({
        printer,
        cnc,
        co2,
        soc,
        fume,
        power,
      }),
    );
  }

  /* ---------- Decision ---------- */

  function renderDecision(data) {
    const decision = data && data.decision;
    const key = fingerprint(decision);
    if (!shouldRender(refs.decisionPanel, key)) return;

    clear(refs.decisionBody);
    setHidden(refs.decisionSource, true);
    setText(refs.decisionMeta, "—");

    if (!decision || typeof decision !== "object") {
      refs.decisionBody.appendChild(
        emptyState({
          iconName: "activity",
          title: "Решений пока нет",
          text: "Агент наблюдает за телеметрией и ещё не выбрал действие для этой смены.",
        }),
      );
      return;
    }

    setText(refs.decisionMeta, `${formatWhen(decision.ts)}${decision.ref ? " · ref " + decision.ref : ""}`);
    refs.decisionMeta.title = decision.ts ? fullTime(decision.ts) : "";

    if (decision.source) {
      setText(refs.decisionSource, `источник: ${decision.source}`);
      refs.decisionSource.className = "tag tag--info";
      setHidden(refs.decisionSource, false);
    }

    refs.decisionBody.appendChild(
      el(
        "p",
        "decision__situation",
        textValue(decision.situation, "Ситуация не описана."),
      ),
    );

    if (decision.cause) {
      refs.decisionBody.appendChild(el("p", "decision__label", "Причина"));
      refs.decisionBody.appendChild(el("p", "decision__cause", String(decision.cause)));
    }

    const confidence = numeric(decision.confidence);
    if (confidence !== null) {
      const normalized = confidence > 1 ? confidence / 100 : confidence;
      const clamped = Math.min(1, Math.max(0, normalized));
      const tone =
        clamped >= 0.75 ? "var(--mint)" : clamped >= 0.5 ? "var(--amber)" : "var(--coral)";
      const block = el("div", "confidence");
      block.style.setProperty("--confidence-tone", tone);
      const head = el("div", "confidence__head");
      head.appendChild(el("span", null, "Уверенность"));
      head.appendChild(
        el("span", "confidence__value", `${Math.round(clamped * 100)}%`),
      );
      const track = el("div", "confidence__track");
      const fill = el("span", "confidence__fill");
      fill.style.width = `${clamped * 100}%`;
      track.appendChild(fill);
      block.setAttribute("role", "progressbar");
      block.setAttribute("aria-label", "Уверенность агента");
      block.setAttribute("aria-valuemin", "0");
      block.setAttribute("aria-valuemax", "100");
      block.setAttribute("aria-valuenow", String(Math.round(clamped * 100)));
      block.appendChild(head);
      block.appendChild(track);
      refs.decisionBody.appendChild(block);
    }

    const plan = arrayOf(decision.plan).filter(
      (step) => step !== null && step !== undefined && step !== "",
    );
    if (plan.length) {
      const block = el("div", "plan-block");
      block.appendChild(el("p", "decision__label", "План"));
      const list = el("ol", "plan");
      for (const step of plan) {
        list.appendChild(el("li", null, displayValue(step)));
      }
      block.appendChild(list);
      refs.decisionBody.appendChild(block);
    }

    const evidence = arrayOf(decision.evidence);
    if (evidence.length) {
      const block = el("div", "plan-block");
      block.appendChild(el("p", "decision__label", "Доказательства"));
      const list = el("ul", "evidence");
      for (const item of evidence) {
        const entry = asObject(item);
        const pill = el("li");
        if (entry.node) pill.appendChild(el("b", null, String(entry.node)));
        if (entry.param) pill.appendChild(el("span", null, String(entry.param)));
        pill.appendChild(el("span", null, `= ${displayValue(entry.value)}`));
        list.appendChild(pill);
      }
      block.appendChild(list);
      refs.decisionBody.appendChild(block);
    }
  }

  /* ---------- Alerts ---------- */

  function alertSort(a, b) {
    const rank = { critical: 0, warning: 1, info: 2 };
    const ar = rank[String(a && a.level).toLowerCase()] ?? 3;
    const br = rank[String(b && b.level).toLowerCase()] ?? 3;
    if (ar !== br) return ar - br;
    return compareTimeDesc(a, b);
  }

  function renderAlerts(data) {
    const alerts = arrayOf(data && data.alerts).slice().sort(alertSort);
    const key = fingerprint(alerts);
    setText(refs.alertsCount, alerts.length ? String(alerts.length) : "0");
    if (!shouldRender(refs.alertsPanel, key)) return;

    clear(refs.alertsBody);
    if (!alerts.length) {
      refs.alertsBody.appendChild(
        emptyState({
          iconName: "check-circle",
          tone: "ok",
          title: "Тревог нет",
          text: "Агент не зафиксировал активных предупреждений по этой смене.",
        }),
      );
      return;
    }

    const list = el("ul", "alert-list");
    for (const item of alerts) {
      const alert = asObject(item);
      const level = ["critical", "warning", "info"].includes(
        String(alert.level || "").toLowerCase(),
      )
        ? String(alert.level).toLowerCase()
        : "info";
      const row = el("li", "alert-item");
      row.dataset.level = level;
      const iconBox = el("span", "alert-item__icon");
      iconBox.appendChild(icon(level === "info" ? "info" : "alert-triangle"));
      row.appendChild(iconBox);

      const main = el("div");
      const head = el("div", "alert-item__head");
      head.appendChild(el("p", "alert-item__title", textValue(alert.title, "Тревога")));
      head.appendChild(el("span", "alert-item__level", LEVEL_LABELS[level] || level));
      main.appendChild(head);
      if (alert.detail) {
        main.appendChild(el("p", "alert-item__detail", String(alert.detail)));
      }
      const time = el("time", "alert-item__time", formatWhen(alert.ts));
      if (alert.ts) time.dateTime = String(alert.ts);
      time.title = fullTime(alert.ts);
      main.appendChild(time);
      row.appendChild(main);
      list.appendChild(row);
    }

    const scroll = el("div", "scroll-area alerts__scroll");
    scroll.appendChild(list);
    refs.alertsBody.appendChild(scroll);
  }

  /* ---------- Timeline ---------- */

  function canonicalKind(kind) {
    const raw = String(kind || "").toLowerCase();
    if (raw.includes("decision")) return "decision";
    if (raw.includes("verif")) return "verification";
    if (raw.includes("action") || raw.includes("control") || raw.includes("command")) {
      return "action";
    }
    if (raw.includes("message") || raw.includes("chat")) return "chat";
    if (raw.includes("error") || raw.includes("fail")) return "error";
    if (raw.includes("observ")) return "observation";
    return "other";
  }

  function normalizeStatus(status) {
    const raw = String(status || "info").toLowerCase();
    if (["success", "ok", "done", "completed"].includes(raw)) return "success";
    if (["warning", "warn", "pending", "stale"].includes(raw)) return "warning";
    if (["error", "fail", "failed", "critical"].includes(raw)) return "error";
    return "info";
  }

  function timelineMatches(record, filter) {
    if (filter === "all") return true;
    if (filter === "error") {
      return canonicalKind(record.kind) === "error" || normalizeStatus(record.status) === "error";
    }
    return canonicalKind(record.kind) === filter;
  }

  function updateTimelineCounts(records) {
    const counts = {
      all: records.length,
      decision: 0,
      action: 0,
      verification: 0,
      chat: 0,
      error: 0,
    };
    for (const record of records) {
      const kind = canonicalKind(record.kind);
      if (Object.prototype.hasOwnProperty.call(counts, kind)) counts[kind] += 1;
      if (normalizeStatus(record.status) === "error") counts.error += 1;
    }
    const nodes = refs.timelineFilters.querySelectorAll("[data-count]");
    for (const node of nodes) {
      setText(node, String(counts[node.dataset.count] || 0));
    }
  }

  function renderTimelineRecords() {
    const list = refs.timelineList;
    const all = arrayOf(app.data && app.data.timeline);
    const signature = `${app.activeFilter}|${fingerprint(all)}`;
    if (list.dataset.sig === signature) return;

    const filterChanged = list.dataset.filter !== app.activeFilter;
    const wasNearTop = list.scrollTop < 24;
    const oldHeight = list.scrollHeight;
    const oldTop = list.scrollTop;

    list.dataset.sig = signature;
    list.dataset.filter = app.activeFilter;

    const filtered = all
      .filter((record) => timelineMatches(asObject(record), app.activeFilter))
      .slice()
      .sort(compareTimeDesc);

    clear(list);
    if (!filtered.length) {
      list.appendChild(
        emptyState({
          iconName: "list",
          title: app.activeFilter === "all" ? "Событий пока нет" : "Нет записей для фильтра",
          text:
            app.activeFilter === "all"
              ? "Лента заполнится, когда агент начнёт смену."
              : "Выберите другой тип событий в фильтре выше.",
        }),
      );
    } else {
      const ol = el("ol", "timeline-list");
      for (const item of filtered) {
        const record = asObject(item);
        const kind = canonicalKind(record.kind);
        const status = normalizeStatus(record.status);

        const row = el("li", "tl-item");
        const time = el("time", "tl-time", formatWhen(record.ts));
        if (record.ts) time.dateTime = String(record.ts);
        time.title = fullTime(record.ts);
        row.appendChild(time);

        const dot = el("span", "tl-dot");
        dot.dataset.status = status;
        dot.title =
          status === "success"
            ? "Успешно"
            : status === "warning"
              ? "Предупреждение"
              : status === "error"
                ? "Ошибка"
                : "Информация";
        row.appendChild(dot);

        const main = el("div", "tl-main");
        const top = el("div", "tl-top");
        const badge = el("span", "kind-badge", KIND_LABELS[kind] || KIND_LABELS.other);
        badge.dataset.kind = kind;
        top.appendChild(badge);
        if (status !== "info") {
          const stateText =
            status === "success" ? "ок" : status === "warning" ? "внимание" : "ошибка";
          const state = el("span", "tl-state", stateText);
          state.dataset.status = status;
          top.appendChild(state);
        }
        if (record.ref !== null && record.ref !== undefined) {
          top.appendChild(el("span", "tl-ref", `ref ${record.ref}`));
        }
        main.appendChild(top);

        main.appendChild(el("p", "tl-title", textValue(record.title, "Событие")));
        if (record.detail) {
          main.appendChild(el("p", "tl-detail", String(record.detail)));
        }
        row.appendChild(main);
        ol.appendChild(row);
      }
      list.appendChild(ol);
    }

    if (wasNearTop || filterChanged) {
      list.scrollTop = 0;
    } else {
      const delta = list.scrollHeight - oldHeight;
      if (delta > 0) list.scrollTop = oldTop + delta;
    }
  }

  function renderTimeline(data) {
    const records = arrayOf(data && data.timeline).slice().sort(compareTimeDesc);
    setText(refs.timelineCount, `записей: ${records.length}`);
    updateTimelineCounts(records);
    renderTimelineRecords();
    shouldRender(refs.timelinePanel, fingerprint(records));
  }

  /* ---------- Messages ---------- */

  function renderMessages(data) {
    const messages = arrayOf(data && data.messages).slice().sort(compareTimeAsc);
    const key = fingerprint(messages);
    setText(refs.messagesCount, String(messages.length));

    const list = refs.messagesList;
    const firstRender = refs.messagesPanel.dataset.fp === undefined;
    if (!shouldRender(refs.messagesPanel, key)) return;

    const wasNearBottom =
      firstRender ||
      list.scrollHeight - list.scrollTop - list.clientHeight < 56;

    clear(list);
    if (!messages.length) {
      list.appendChild(
        emptyState({
          iconName: "message",
          title: "Сообщений пока нет",
          text: "Оператор ещё не писал агенту в этой смене.",
        }),
      );
      return;
    }

    for (const item of messages) {
      const message = asObject(item);
      const sender = asObject(message.sender);
      const role = String(sender.role || "").toLowerCase();
      const direction = String(message.direction || "").toLowerCase();
      const operator = role === "operator" || (direction === "in" && role !== "system");
      const system = role === "system";
      const row = el(
        "article",
        "msg" + (operator ? " msg--operator" : system ? " msg--system" : ""),
      );

      const head = el("div", "msg__head");
      const name = sender.name ? String(sender.name) : operator ? "Оператор" : "Агент";
      head.appendChild(el("span", "msg__who", name));
      head.appendChild(
        el("span", "msg__role", ROLE_LABELS[role] || role || "участник"),
      );

      const dir = el("span", "msg__dir");
      dir.appendChild(icon(direction === "out" ? "out" : "in"));
      dir.appendChild(el("span", null, direction === "out" ? "исходящее" : "входящее"));
      if (message.to) dir.title = `Кому: ${message.to}`;
      head.appendChild(dir);

      const time = el("time", "msg__time", formatWhen(message.ts));
      if (message.ts) time.dateTime = String(message.ts);
      time.title = fullTime(message.ts);
      head.appendChild(time);
      row.appendChild(head);

      row.appendChild(el("p", "msg__text", textValue(message.text, "—")));
      list.appendChild(row);
    }

    if (wasNearBottom) {
      requestAnimationFrame(() => {
        list.scrollTop = list.scrollHeight;
      });
    }
  }

  /* ---------- Actions and verifications ---------- */

  function verificationModel(value) {
    if (!value || typeof value !== "object") return null;
    return {
      ok: value.ok === true,
      known: value.ok === true || value.ok === false,
      expected: displayValue(value.expected),
      observed: displayValue(value.observed),
      ts: value.ts,
    };
  }

  function renderActions(data) {
    const actions = arrayOf(data && data.actions).slice().sort(compareTimeDesc);
    const verifications = arrayOf(data && data.verifications);
    const key = fingerprint({ actions, verifications });

    setText(
      refs.actionsCount,
      `действий: ${actions.length} · проверок: ${verifications.length}`,
    );
    if (!shouldRender(refs.actionsPanel, key)) return;

    clear(refs.actionsBody);
    if (!actions.length && !verifications.length) {
      refs.actionsBody.appendChild(
        emptyState({
          iconName: "shield",
          title: "Действий пока нет",
          text: "Агент не отправлял управляющих команд в этой смене.",
        }),
      );
      return;
    }

    const used = new Array(verifications.length).fill(false);
    const rows = [];

    for (const item of actions) {
      const action = asObject(item);
      const request = asObject(action.request);
      let ver = verificationModel(action.verification);

      if (action.seq !== null && action.seq !== undefined) {
        const matchIndex = verifications.findIndex(
          (candidate, index) =>
            !used[index] &&
            candidate &&
            String(candidate.ref) === String(action.seq),
        );
        if (matchIndex >= 0) {
          if (!ver) ver = verificationModel(verifications[matchIndex]);
          used[matchIndex] = true;
        }
      }

      rows.push({
        ts: action.ts,
        request,
        response: asObject(action.response),
        hasResponse: action.response !== null && action.response !== undefined,
        verification: ver,
        orphan: false,
      });
    }

    for (let i = 0; i < verifications.length; i += 1) {
      if (!used[i]) {
        rows.push({
          ts: verifications[i] && verifications[i].ts,
          request: {},
          response: {},
          hasResponse: false,
          verification: verificationModel(verifications[i]),
          orphan: true,
        });
      }
    }

    rows.sort(compareTimeDesc);

    const wrap = el("div", "table-wrap");
    const table = el("table", "data-table");
    const caption = el("caption", "sr-only", "Действия агента и результаты проверки");
    table.appendChild(caption);

    const thead = el("thead");
    const headRow = el("tr");
    for (const label of ["Время", "Действие", "Узел", "Отправка", "Проверка"]) {
      const th = el("th", null, label);
      th.scope = "col";
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = el("tbody");
    for (const row of rows) {
      const tr = el("tr");
      const request = row.request;
      const response = row.response;

      const timeCell = el("td");
      timeCell.dataset.label = "Время";
      const time = el("span", "cell-time", formatWhen(row.ts));
      time.title = fullTime(row.ts);
      timeCell.appendChild(time);
      tr.appendChild(timeCell);

      const actionCell = el("td");
      actionCell.dataset.label = "Действие";
      actionCell.appendChild(
        el(
          "span",
          "cell-strong",
          row.orphan
            ? "Проверка"
            : textValue(request.action || request.type, "—"),
        ),
      );
      const requestBits = [];
      if (request.type && request.type !== request.action) requestBits.push(String(request.type));
      if (request.value !== null && request.value !== undefined) {
        requestBits.push(`= ${displayValue(request.value)}`);
      }
      if (requestBits.length) {
        actionCell.appendChild(el("span", "cell-sub", requestBits.join(" ")));
      }
      tr.appendChild(actionCell);

      const nodeCell = el("td");
      nodeCell.dataset.label = "Узел";
      nodeCell.appendChild(
        el("span", "cell-strong", textValue(request.node_id, row.orphan ? "—" : "—")),
      );
      tr.appendChild(nodeCell);

      const sendCell = el("td");
      sendCell.dataset.label = "Отправка";
      if (row.orphan || !row.hasResponse) {
        sendCell.appendChild(statusLine("neutral", "—", "clock"));
      } else if (response.success === true) {
        sendCell.appendChild(statusLine("ok", "успешно", "check-circle"));
      } else if (response.success === false) {
        sendCell.appendChild(statusLine("crit", "отказ", "alert-circle"));
      } else {
        sendCell.appendChild(statusLine("neutral", "нет ответа", "clock"));
      }
      if (response.error) {
        sendCell.appendChild(el("span", "cell-sub", String(response.error)));
      }
      tr.appendChild(sendCell);

      const verifyCell = el("td");
      verifyCell.dataset.label = "Проверка";
      const verifyBox = el("div", "verify-cell");
      const ver = row.verification;
      if (!ver) {
        verifyBox.appendChild(statusLine("neutral", "не проверено", "clock"));
      } else if (ver.known && ver.ok) {
        verifyBox.appendChild(statusLine("ok", "подтверждено", "check-circle"));
      } else if (ver.known && !ver.ok) {
        verifyBox.appendChild(statusLine("crit", "не совпало", "alert-circle"));
      } else {
        verifyBox.appendChild(statusLine("neutral", "результат неизвестен", "info"));
      }
      if (ver && ver.expected !== "—") {
        verifyBox.appendChild(el("b", null, `ож: ${ver.expected}`));
      }
      if (ver && ver.observed !== "—") {
        verifyBox.appendChild(el("span", null, `факт: ${ver.observed}`));
      }
      verifyCell.appendChild(verifyBox);
      tr.appendChild(verifyCell);

      tbody.appendChild(tr);
    }

    table.appendChild(tbody);
    wrap.appendChild(table);
    refs.actionsBody.appendChild(wrap);
  }

  /* ---------- State / system ---------- */

  function appendDef(grid, label, value, options = {}) {
    const item = el("div", "def");
    item.appendChild(el("dt", null, label));
    const dd = el("dd", options.mono ? "mono" : options.muted ? "muted" : null);
    if (value instanceof Node) {
      dd.appendChild(value);
    } else {
      dd.textContent = textValue(value);
      if (options.muted) dd.classList.add("muted");
    }
    item.appendChild(dd);
    grid.appendChild(item);
  }

  function renderState(data) {
    const state = asObject(data && data.state);
    const run = asObject(data && data.run);
    const config = asObject(run.config);
    const health = asObject(data && data.health);
    const counts = asObject(data && data.counts);
    const resolved = resolveHealth(data);

    const key = fingerprint({
      state,
      run,
      config,
      health: {
        status: health.status,
        has_log: health.has_log,
        has_state: health.has_state,
        log_file: health.log_file,
        state_file: health.state_file,
        records_loaded: health.records_loaded,
        last_record_age_s: health.last_record_age_s,
      },
      counts,
    });

    if (!shouldRender(refs.statePanel, key, false)) return;

    clear(refs.stateBody);
    setText(
      refs.stateMeta,
      state.updated_at ? `обновлено ${relativeAge(state.updated_at)}` : "—",
    );
    refs.stateMeta.title = state.updated_at ? fullTime(state.updated_at) : "";

    const healthRow = el("div", "state-health");
    healthRow.appendChild(tagNode(resolved.label, resolved.tone));
    healthRow.appendChild(el("span", "state-health__text", resolved.hint));
    const recordAge =
      health.last_record_age_s !== null && health.last_record_age_s !== undefined
        ? `запись ${formatAgeSeconds(health.last_record_age_s)}`
        : "возраст записи неизвестен";
    healthRow.appendChild(el("span", "state-health__age", recordAge));
    refs.stateBody.appendChild(healthRow);

    const defs = el("dl", "def-grid");
    appendDef(defs, "Режим", textValue(config.mode));
    appendDef(
      defs,
      "Управление",
      tagNode(
        config.allow_control === true ? "разрешено" : "запрещено",
        config.allow_control === true ? "warn" : "neutral",
      ),
    );
    appendDef(
      defs,
      "Dry-run",
      tagNode(config.dry_run === true ? "да" : "нет", config.dry_run === true ? "warn" : "neutral"),
    );
    const [taskLabel] = taskState(asObject(state.task).status);
    appendDef(defs, "Задача", state.task ? taskLabel : "не задана");
    appendDef(
      defs,
      "Последнее сообщение",
      state.last_message_id !== null && state.last_message_id !== undefined
        ? `#${state.last_message_id}`
        : "—",
      { mono: true },
    );
    appendDef(defs, "Состояние обновлено", formatWhen(state.updated_at), { mono: true });
    appendDef(
      defs,
      "Смена",
      run.stopped === true ? "остановлена" : "активна",
      { muted: run.stopped !== true },
    );
    appendDef(defs, "Dome URL", textValue(config.dome_url), { mono: true, muted: !config.dome_url });

    if (state.co2_override !== null && state.co2_override !== undefined) {
      appendDef(defs, "CO₂ override", displayValue(state.co2_override), { mono: true });
    }
    if (run.stop_reason) {
      appendDef(defs, "Причина остановки", String(run.stop_reason));
    }
    refs.stateBody.appendChild(defs);

    const countGrid = el("div", "counts");
    const countItems = [
      ["records", "записей", "neutral"],
      ["decisions", "решений", "neutral"],
      ["actions", "действий", "neutral"],
      ["verifications", "проверок", "neutral"],
      ["messages", "сообщений", "neutral"],
      ["errors", "ошибок", "crit"],
    ];
    for (const [field, label, tone] of countItems) {
      const value = numeric(counts[field]);
      const box = el("div", "count");
      box.dataset.tone = tone === "crit" && value !== null && value > 0 ? "crit" : "neutral";
      box.appendChild(el("span", "count__value", value === null ? "—" : fmtNumber(value)));
      box.appendChild(el("span", "count__label", label));
      countGrid.appendChild(box);
    }
    refs.stateBody.appendChild(countGrid);

    const files = el("dl", "file-list");
    appendFile(files, "Лог", health.log_file || run.file);
    appendFile(files, "Состояние", health.state_file);
    refs.stateBody.appendChild(files);
  }

  function appendFile(list, label, value) {
    const row = el("div", "file-row");
    row.appendChild(el("dt", null, label));
    row.appendChild(el("dd", null, textValue(value, "файл не указан")));
    list.appendChild(row);
  }

  /* ---------- Empty / no-log / no-data ---------- */

  function renderNoLog(data) {
    const health = asObject(data && data.health);
    const show = !app.demo && health.has_log === false;
    setHidden(refs.noLogState, !show);
  }

  function renderNoData(error) {
    app.hasData = false;
    document.body.dataset.ready = "true";
    setHidden(refs.connectBanner, true);
    setOnline(false);
    setStatus("WAITING", "warn", "Backend не отвечает");
    setHidden(refs.stalePill, true);
    setHidden(refs.demoBadge, !app.demo);
    setText(refs.runId, "—");
    setText(refs.roleTeam, "—");
    setText(refs.runMode, "—");
    setText(refs.lastUpdate, "нет связи");
    refs.lastUpdate.title = error ? describeError(error) : "";

    setText(refs.taskText, "Данные агента недоступны — backend не отвечает.");
    setText(refs.taskStatus, "Нет связи");
    refs.taskStatus.className = "tag tag--crit";
    setText(refs.taskRunMeta, "API недоступен");
    setText(refs.runStarted, "—");
    setText(refs.runAgent, "—");
    setText(refs.runModels, "—");

    for (const name of ["printer", "cnc", "co2", "battery", "fume", "power"]) {
      setKpi(name, {
        value: "—",
        tone: "neutral",
        state: name === "power" ? "нет данных" : "нет связи",
      });
    }
    const batteryFill = $("kpi-battery-fill");
    if (batteryFill) batteryFill.style.width = "0%";

    setHidden(refs.decisionSource, true);
    setText(refs.decisionMeta, "—");
    setText(refs.alertsCount, "—");
    setText(refs.timelineCount, "—");
    setText(refs.messagesCount, "—");
    setText(refs.actionsCount, "—");
    setText(refs.stateMeta, "—");

    for (const node of refs.timelineFilters.querySelectorAll("[data-count]")) {
      setText(node, "0");
    }

    const unavailable = (title, iconName) =>
      emptyState({
        iconName,
        title,
        text: "Не удалось получить данные от backend. Проверьте соединение и повторите запрос.",
      });

    clear(refs.decisionBody);
    refs.decisionBody.appendChild(unavailable("Решение недоступно", "activity"));
    clear(refs.alertsBody);
    refs.alertsBody.appendChild(unavailable("Тревоги недоступны", "alert-triangle"));
    clear(refs.timelineList);
    refs.timelineList.appendChild(unavailable("Лента недоступна", "list"));
    clear(refs.messagesList);
    refs.messagesList.appendChild(unavailable("Чат недоступен", "message"));
    clear(refs.actionsBody);
    refs.actionsBody.appendChild(unavailable("Действия недоступны", "shield"));
    clear(refs.stateBody);
    refs.stateBody.appendChild(unavailable("Состояние недоступно", "database"));
    setHidden(refs.noLogState, true);
    renderFooter();
  }

  function renderFooter() {
    if (!app.online) {
      setText(refs.footerSource, "нет связи с API");
      return;
    }
    if (app.demo) {
      setText(refs.footerSource, "GET /api/demo · демо-данные");
      return;
    }
    setText(refs.footerSource, `GET /api/summary · ${Math.round(app.intervalMs / 1000)} с`);
  }

  function showError(error, hasData) {
    setOnline(false);
    const detail = describeError(error);
    if (hasData) {
      refs.errorDetail.textContent = `Последние данные: ${relativeAge(app.lastGoodAt)}. Показанное состояние сохранено.`;
    } else {
      refs.errorDetail.textContent = detail;
    }
    refs.errorDetail.title = detail;
    setHidden(refs.errorBanner, false);
    if (hasData && app.data) {
      setText(refs.lastUpdate, `нет связи · данные ${relativeAge(app.lastGoodAt)}`);
      refs.lastUpdate.title = `Последние данные: ${fullTime(app.lastGoodAt)}`;
    }
  }

  function hideError() {
    setHidden(refs.errorBanner, true);
    refs.errorDetail.title = "";
  }

  /* ---------- Apply data ---------- */

  function applySummary(data, options = {}) {
    if (!data || typeof data !== "object") {
      throw new Error("API вернул пустой ответ.");
    }

    app.data = data;
    app.hasData = true;
    app.demo = options.fromDemo === true || data.demo === true;
    app.lastGoodAt = Date.now();

    setOnline(true);
    hideError();
    document.body.dataset.ready = "true";

    renderDemoUi();
    renderHeader(data);
    renderTask(data);
    renderKpis(data);
    renderDecision(data);
    renderAlerts(data);
    renderTimeline(data);
    renderMessages(data);
    renderActions(data);
    renderState(data);
    renderNoLog(data);
    renderFooter();

    setHidden(refs.connectBanner, true);
  }

  /* ---------- Fetching ---------- */

  async function fetchJson(url, signal) {
    const response = await fetch(url, {
      signal,
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      const status = [response.status, response.statusText].filter(Boolean).join(" ");
      throw new Error(`HTTP ${status || "error"}`);
    }
    return response.json();
  }

  async function refresh(options = {}) {
    const manual = options.manual === true;
    const seq = (app.seq += 1);
    if (app.fetchAbort) app.fetchAbort.abort();
    const controller = new AbortController();
    app.fetchAbort = controller;

    if (manual) setBusy(true);

    try {
      const data = await fetchJson(API.summary, controller.signal);
      if (seq !== app.seq) return;
      applySummary(data, { fromDemo: false });
    } catch (error) {
      if (error && error.name === "AbortError") return;
      if (seq !== app.seq) return;
      setOnline(false);
      showError(error, app.hasData);
      if (!app.hasData) renderNoData(error);
    } finally {
      if (seq === app.seq && manual) setBusy(false);
    }
  }

  async function loadDemo() {
    const seq = (app.seq += 1);
    if (app.fetchAbort) app.fetchAbort.abort();
    const controller = new AbortController();
    app.fetchAbort = controller;

    try {
      const data = await fetchJson(API.demo, controller.signal);
      if (seq !== app.seq) return false;
      applySummary(data, { fromDemo: true });
      return true;
    } catch (error) {
      if (error && error.name === "AbortError") return false;
      if (seq !== app.seq) return false;
      setOnline(false);
      showError(error, app.hasData);
      if (!app.hasData) renderNoData(error);
      return false;
    }
  }

  async function enterDemo() {
    if (app.demoBusy || app.demo) return;
    app.demoBusy = true;
    stopTimer();
    refs.demoButton.setAttribute("aria-busy", "true");
    const ok = await loadDemo();
    app.demoBusy = false;
    refs.demoButton.removeAttribute("aria-busy");
    if (!ok && !app.demo) scheduleNext();
  }

  async function exitDemo() {
    if (app.demoBusy) return;
    app.demoBusy = true;
    app.demo = false;
    renderDemoUi();
    stopTimer();
    await refresh({ manual: true });
    app.demoBusy = false;
    scheduleNext();
  }

  function toggleDemo() {
    if (app.demo) {
      exitDemo();
    } else {
      enterDemo();
    }
  }

  async function manualRefresh() {
    if (app.demo) {
      await loadDemo();
      return;
    }
    stopTimer();
    await refresh({ manual: true });
    scheduleNext();
  }

  function stopTimer() {
    if (app.timer !== null) {
      window.clearTimeout(app.timer);
      app.timer = null;
    }
  }

  function scheduleNext() {
    stopTimer();
    if (document.hidden || app.demo || app.demoBusy) return;
    app.timer = window.setTimeout(async () => {
      app.timer = null;
      await refresh({ reason: "poll" });
      scheduleNext();
    }, app.intervalMs);
  }

  /* ---------- Raw log dialog ---------- */

  function openLogDialog() {
    if (!refs.logDialog.open) {
      if (typeof refs.logDialog.showModal === "function") {
        refs.logDialog.showModal();
      } else {
        refs.logDialog.setAttribute("open", "");
      }
    }
    loadLog();
  }

  function closeLogDialog() {
    if (typeof refs.logDialog.close === "function" && refs.logDialog.open) {
      refs.logDialog.close();
    } else {
      refs.logDialog.removeAttribute("open");
    }
    app.logSeq += 1;
    if (app.logAbort) app.logAbort.abort();
  }

  async function loadLog() {
    const seq = (app.logSeq += 1);
    if (app.logAbort) app.logAbort.abort();
    const controller = new AbortController();
    app.logAbort = controller;

    const runId = app.data && app.data.run ? app.data.run.run_id : null;
    const params = new URLSearchParams();
    params.set("limit", "200");
    if (refs.logKind.value) params.set("kind", refs.logKind.value);
    if (runId) params.set("run_id", String(runId));

    refs.logMeta.textContent = runId
      ? `run ${runId} · последние 200 записей`
      : "Последние 200 записей";
    refs.logStatus.dataset.tone = "";
    refs.logStatus.textContent = "Загрузка…";
    refs.logPre.textContent = "Загрузка…";

    try {
      const payload = await fetchJson(`${API.logs}?${params.toString()}`, controller.signal);
      if (seq !== app.logSeq) return;
      const records = Array.isArray(payload.records) ? payload.records : [];
      const count = payload.count !== undefined && payload.count !== null
        ? payload.count
        : records.length;

      const metaBits = [
        payload.file || "файл не указан",
        `записей: ${count}`,
        payload.truncated ? "список обрезан до лимита" : "",
      ].filter(Boolean);
      refs.logMeta.textContent = metaBits.join(" · ");
      refs.logStatus.dataset.tone = "";
      refs.logStatus.textContent = records.length ? `${records.length} записей` : "Записей нет";
      refs.logPre.textContent = JSON.stringify(records, null, 2);
      refs.logPre.scrollTop = 0;
      refs.logPre.scrollLeft = 0;
    } catch (error) {
      if (error && error.name === "AbortError") return;
      if (seq !== app.logSeq) return;
      refs.logStatus.dataset.tone = "crit";
      refs.logStatus.textContent = "Ошибка";
      refs.logPre.textContent = `Не удалось загрузить журнал.\n\n${describeError(error)}`;
    }
  }

  /* ---------- Events ---------- */

  function wireEvents() {
    refs.refreshButton.addEventListener("click", manualRefresh);
    refs.errorRetry.addEventListener("click", manualRefresh);
    refs.demoButton.addEventListener("click", toggleDemo);
    refs.noLogDemo.addEventListener("click", enterDemo);

    refs.interval.addEventListener("change", () => {
      const value = Number(refs.interval.value);
      if (!ALLOWED_INTERVALS.includes(value)) return;
      app.intervalMs = value;
      storeInterval(value);
      renderFooter();
      if (!app.demo) {
        stopTimer();
        scheduleNext();
      }
    });

    refs.timelineFilters.addEventListener("click", (event) => {
      const target = event.target instanceof Element ? event.target : null;
      const button = target ? target.closest("button[data-filter]") : null;
      if (!button) return;
      app.activeFilter = button.dataset.filter || "all";
      for (const chip of refs.timelineFilters.querySelectorAll("button[data-filter]")) {
        const active = chip === button;
        chip.classList.toggle("is-active", active);
        chip.setAttribute("aria-pressed", String(active));
      }
      renderTimelineRecords();
    });

    refs.logButton.addEventListener("click", openLogDialog);
    refs.logClose.addEventListener("click", closeLogDialog);
    refs.logReload.addEventListener("click", loadLog);
    refs.logKind.addEventListener("change", loadLog);
    refs.logDialog.addEventListener("close", () => {
      app.logSeq += 1;
      if (app.logAbort) app.logAbort.abort();
    });
    refs.logDialog.addEventListener("click", (event) => {
      if (event.target === refs.logDialog) closeLogDialog();
    });

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        stopTimer();
        return;
      }
      if (!app.demo && !app.demoBusy) {
        refresh({ reason: "visible" }).finally(scheduleNext);
      }
    });

    window.addEventListener("offline", () => {
      setOnline(false);
      stopTimer();
      if (app.hasData && app.lastGoodAt) {
        setText(refs.lastUpdate, `нет связи · данные ${relativeAge(app.lastGoodAt)}`);
      }
    });

    window.addEventListener("online", () => {
      if (!app.demo) manualRefresh();
    });
  }

  function init() {
    refs.interval.value = String(app.intervalMs);
    renderDemoUi();
    renderFooter();
    wireEvents();
    refresh({ reason: "init" }).finally(() => {
      if (!app.demo) scheduleNext();
    });
  }

  init();
})();
