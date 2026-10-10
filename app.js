let data;
let assignments = {};
let names = [];
let personRenames = {};

const groupsElement = document.querySelector("#groups");
const escapeHtml = (value) => String(value).replace(
  /[&<>"']/g,
  (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  })[character],
);

fetch("/api/data")
  .then((response) => response.json())
  .then((result) => {
    data = result;
    data.faces.forEach((face) => { assignments[face.id] = face.group; });
    render();
  });

function render() {
  names = [...new Set([...Object.values(assignments), "DROP"])]
    .sort((left, right) => left === "DROP" ? 1 : right === "DROP" ? -1 : left.localeCompare(right));
  document.querySelector("#stats").textContent =
    `${data.faces.length} faces · ${data.images} images · ${names.length} groups` +
    (data.knownPeople ? ` · ${data.knownPeople} saved people` : "") +
    (data.errors ? ` · ${data.errors} unreadable` : "");
  groupsElement.innerHTML = names.map(groupMarkup).join("");
  bindGroupEvents();
}

function groupMarkup(name) {
  const faces = data.faces.filter((face) => assignments[face.id] === name);
  const isDrop = name === "DROP";
  const knownIds = [...new Set(
    faces.map((face) => face.knownPersonId).filter((id) => id !== null)
  )];
  const knownPersonId =
    knownIds.length === 1 &&
    faces.every((face) => face.knownPersonId === knownIds[0])
      ? knownIds[0]
      : null;
  const heading = isDrop
    ? '<div class="group-name">DROP</div>'
    : `<input class="group-name" value="${escapeHtml(name)}" data-old="${escapeHtml(name)}" ${knownPersonId === null ? "" : `data-person-id="${knownPersonId}"`} aria-label="Group name">`;
  const cards = faces.map((face) => `
    <figure class="face ${face.lowQuality ? "low" : ""}" draggable="true" data-id="${face.id}" title="${escapeHtml(face.file)}">
      <img src="${face.crop}" loading="lazy">
      <figcaption>${escapeHtml(face.file)}</figcaption>
    </figure>`).join("");
  return `
    <section class="group ${isDrop ? "drop-group" : ""}" data-name="${escapeHtml(name)}">
      <div class="group-head">
        ${heading}
        <span class="count">${faces.length} face${faces.length === 1 ? "" : "s"}${isDrop ? " · excluded from filenames" : ""}</span>
      </div>
      <div class="face-grid">${cards}</div>
    </section>`;
}

function bindGroupEvents() {
  document.querySelectorAll("input.group-name").forEach((input) => {
    input.addEventListener("change", () => {
      const oldName = input.dataset.old;
      const value = input.value.trim();
      if (!value) { input.value = oldName; return; }
      if (input.dataset.personId) personRenames[input.dataset.personId] = value;
      Object.keys(assignments).forEach((id) => {
        if (assignments[id] === oldName) assignments[id] = value;
      });
      render();
    });
  });
  document.querySelectorAll(".face").forEach((face) => {
    face.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", face.dataset.id);
      face.classList.add("dragging");
    });
    face.addEventListener("dragend", () => face.classList.remove("dragging"));
    face.addEventListener("contextmenu", (event) => showMenu(event, face.dataset.id));
  });
  document.querySelectorAll(".group").forEach((group) => {
    group.addEventListener("dragover", (event) => {
      event.preventDefault();
      group.classList.add("dragover");
    });
    group.addEventListener("dragleave", () => group.classList.remove("dragover"));
    group.addEventListener("drop", (event) => {
      event.preventDefault();
      assignments[event.dataTransfer.getData("text/plain")] = group.dataset.name;
      render();
    });
  });
}

function showMenu(event, id) {
  event.preventDefault();
  document.querySelector(".menu")?.remove();
  const menu = document.createElement("div");
  menu.className = "menu";
  menu.style.left = `${Math.min(event.clientX, innerWidth - 230)}px`;
  menu.style.top = `${Math.min(event.clientY, innerHeight - 320)}px`;
  menu.innerHTML = '<div style="padding:5px 9px;color:#99a3ae">Move face to…</div>' +
    names.map((name) => `<button data-name="${escapeHtml(name)}">${escapeHtml(name)}</button>`).join("") +
    '<button data-name="__new">＋ New person…</button>';
  document.body.append(menu);
  menu.querySelectorAll("button").forEach((button) => {
    button.onclick = () => {
      let name = button.dataset.name;
      if (name === "__new") name = prompt("New person name:")?.trim();
      if (name) { assignments[id] = name; render(); }
      menu.remove();
    };
  });
  setTimeout(() => addEventListener("click", () => menu.remove(), { once: true }), 0);
}

document.querySelector("#confirm").onclick = async () => {
  const button = document.querySelector("#confirm");
  if (!confirm("Write face metadata and rename the image files now? This changes the files on disk.")) return;
  button.disabled = true;
  button.textContent = "Saving…";
  try {
    const response = await fetch("/api/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        assignments,
        personRenames,
        maxNames: Number(document.querySelector("#maxNames").value),
      }),
    });
    const result = await response.json();
    if (!response.ok) throw Error(result.error || "Save failed");
    button.textContent = "Completed";
    let extra = ` Metadata written to ${result.metadataWritten} images.`;
    if (result.metadataSkipped.length) extra += ` ${result.metadataSkipped.length} unsupported format(s) skipped: ${result.metadataSkipped.join(", ")}.`;
    if (result.metadataErrors.length) extra += ` Metadata errors: ${result.metadataErrors.join(" | ")}`;
    extra += ` Remembered ${result.databaseSaved} reviewed face${result.databaseSaved === 1 ? "" : "s"}.`;
    if (result.databaseError) extra += ` Database error: ${result.databaseError}`;
    toast(
      `Renamed ${result.renamed} files; ${result.unchanged} unchanged.${extra}`,
      result.metadataErrors.length > 0 || Boolean(result.databaseError),
    );
  } catch (error) {
    button.disabled = false;
    button.textContent = "Confirm & rename";
    toast(error.message, true);
  }
};

function toast(text, error = false) {
  const element = document.createElement("div");
  element.className = `toast${error ? " error" : ""}`;
  element.textContent = text;
  document.body.append(element);
  setTimeout(() => element.remove(), 6000);
}
