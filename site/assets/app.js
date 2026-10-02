// Newtrinos verifications: faceted browsing of entries (from index.json), copy buttons, code highlighting.
(function () {
  const holder = document.querySelector("[data-root]") || document.body;
  const root = holder.dataset.root || "./";
  document.body.dataset.index = holder.dataset.index || "";

  if (window.hljs) hljs.highlightAll();

  document.querySelectorAll("button.copy").forEach((b) => {
    b.addEventListener("click", async () => {
      const el = document.getElementById(b.dataset.copy);
      try {
        await navigator.clipboard.writeText(el.innerText);
        b.textContent = "Copied";
      } catch (e) {
        b.textContent = "Select & copy";
      }
      setTimeout(() => (b.textContent = "Copy"), 1500);
    });
  });

  const list = document.getElementById("list");
  const facetsEl = document.getElementById("facets");
  if (!list || !facetsEl) return;

  const FACETS = [
    { key: "experiment", label: "Experiment", get: (e) => e.experiments },
    { key: "year", label: "Year", get: (e) => [String(e.year)] },
    { key: "parameter", label: "Parameter", get: (e) => e.parameters },
    { key: "status", label: "Status", get: (e) => [e.status] },
  ];
  const STATUS = { reproduced: "Reproduced", partial: "Partially reproduced", "in-progress": "In progress" };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const params = new URLSearchParams(location.search);
  const state = {};
  FACETS.forEach((f) => (state[f.key] = new Set(params.getAll(f.key))));
  const search = document.getElementById("search");
  const sort = document.getElementById("sort");
  search.value = params.get("q") || "";
  sort.value = params.get("sort") || "newest";

  let entries = [];

  function matches(e, skip) {
    for (const f of FACETS) {
      if (f.key === skip || state[f.key].size === 0) continue;
      if (!f.get(e).some((v) => state[f.key].has(v))) return false;
    }
    const q = search.value.trim().toLowerCase();
    if (q && ![e.title, e.short, e.doi, e.arxiv, e.journal, e.experiments.join(" ")].join(" ").toLowerCase().includes(q)) return false;
    return true;
  }

  function card(e) {
    const tags = e.experiments.map((x) => `<span class="tag exp">${esc(x)}</span>`).join(" ") +
      ` <span class="tag">${e.year}</span> <span class="tag status-${e.status}">${esc(STATUS[e.status] || e.status)}</span>`;
    return `<article class="card">
      <a class="card-link" href="${root}entries/${esc(e.slug)}/${document.body.dataset.index || ""}"><h3>${esc(e.short)}</h3><p class="muted">${esc(e.title)}</p></a>
      <div class="tags">${tags}</div>
      <div class="meta muted">${esc(e.journal)} (${e.year}) · ${e.figures} figure(s) · added ${esc(e.added)}</div>
    </article>`;
  }

  function renderFacets() {
    facetsEl.innerHTML = FACETS.map((f) => {
      const counts = {};
      entries.filter((e) => matches(e, f.key)).forEach((e) => f.get(e).forEach((v) => (counts[v] = (counts[v] || 0) + 1)));
      const all = new Set(entries.flatMap(f.get));
      const values = [...all].sort((a, b) => (f.key === "year" ? b.localeCompare(a) : (counts[b] || 0) - (counts[a] || 0) || a.localeCompare(b)));
      const items = values.map((v) => `<label><span><input type="checkbox" data-facet="${f.key}" value="${esc(v)}" ${state[f.key].has(v) ? "checked" : ""}>${esc(f.key === "status" ? STATUS[v] || v : v)}</span><span class="n">${counts[v] || 0}</span></label>`).join("");
      return `<div class="facet"><h4>${f.label}</h4>${items}</div>`;
    }).join("") + `<a href="#" class="clear" id="clear">Clear all</a>`;
    facetsEl.querySelectorAll("input").forEach((i) => i.addEventListener("change", () => {
      const s = state[i.dataset.facet];
      i.checked ? s.add(i.value) : s.delete(i.value);
      update();
    }));
    document.getElementById("clear").addEventListener("click", (ev) => {
      ev.preventDefault();
      FACETS.forEach((f) => state[f.key].clear());
      search.value = "";
      update();
    });
  }

  function update() {
    const shown = entries.filter((e) => matches(e));
    const by = {
      newest: (a, b) => b.year - a.year || b.added.localeCompare(a.added),
      oldest: (a, b) => a.year - b.year,
      added: (a, b) => b.added.localeCompare(a.added),
      title: (a, b) => a.short.localeCompare(b.short),
    }[sort.value];
    shown.sort(by);
    list.innerHTML = shown.map(card).join("") || `<p class="muted">No entries match.</p>`;
    document.getElementById("count").textContent = `${shown.length} / ${entries.length} entries`;
    const p = new URLSearchParams();
    FACETS.forEach((f) => state[f.key].forEach((v) => p.append(f.key, v)));
    if (search.value) p.set("q", search.value);
    if (sort.value !== "newest") p.set("sort", sort.value);
    try { history.replaceState(null, "", p.toString() ? `?${p}` : location.pathname); } catch (e) { /* sandboxed preview */ }
    renderFacets();
  }

  search.addEventListener("input", update);
  sort.addEventListener("change", update);
  fetch(`${root}index.json`).then((r) => r.json()).then((d) => { entries = d; update(); })
    .catch(() => { facetsEl.innerHTML = `<p class="muted">Filters unavailable (index.json could not be loaded).</p>`; });
})();
