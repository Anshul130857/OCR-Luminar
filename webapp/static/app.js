const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const scanDoc = document.getElementById("scan-doc");
const queueList = document.getElementById("queue-list");
const queueEmpty = document.getElementById("queue-empty");
const queueCount = document.getElementById("queue-count");
const engineDot = document.getElementById("engine-dot");
const engineText = document.getElementById("engine-text");
const formatSelect = document.getElementById("format-select");
const dpiSelect = document.getElementById("dpi-select");

const jobs = new Map(); // job_id -> job state
const pollers = new Map();

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.ok) {
      engineDot.className = "dot ok";
      engineText.textContent = `${data.model} ready`;
    } else {
      engineDot.className = "dot bad";
      engineText.textContent = "model unreachable";
    }
  } catch {
    engineDot.className = "dot bad";
    engineText.textContent = "backend unreachable";
  }
}
checkHealth();
setInterval(checkHealth, 15000);

["dragover", "dragenter"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
  })
);
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  fileInput.value = "";
});

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);

  const format = formatSelect.value;
  const dpi = dpiSelect.value;

  let jobId;
  try {
    const res = await fetch(`/api/jobs?format=${format}&dpi=${dpi}`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "upload failed" }));
      alert(err.detail || "Upload failed");
      return;
    }
    const data = await res.json();
    jobId = data.job_id;
  } catch {
    alert("Could not reach the backend.");
    return;
  }

  jobs.set(jobId, { id: jobId, filename: file.name, status: "queued", done_pages: 0, total_pages: 0 });
  renderQueue();
  pollJob(jobId);
}

function pollJob(jobId) {
  if (pollers.has(jobId)) return;
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) return;
      const data = await res.json();
      jobs.set(jobId, data);
      renderQueue();
      if (data.status === "done" || data.status === "error") {
        clearInterval(interval);
        pollers.delete(jobId);
      }
    } catch {
      /* transient network hiccup, keep polling */
    }
  }, 1200);
  pollers.set(jobId, interval);
}

function renderQueue() {
  const list = Array.from(jobs.values());
  queueCount.textContent = list.length;
  queueEmpty.style.display = list.length ? "none" : "block";

  const anyActive = list.some((j) => j.status === "processing" || j.status === "reading" || j.status === "queued");
  scanDoc.classList.toggle("active", anyActive);

  queueList.querySelectorAll(".job").forEach((el) => el.remove());

  for (const job of list) {
    const el = document.createElement("div");
    el.className = "job";

    const pct = job.total_pages ? Math.round((job.done_pages / job.total_pages) * 100) : 0;
    const statusLabel =
      job.status === "processing" && job.total_pages
        ? `${job.status} &middot; ${job.done_pages}/${job.total_pages}`
        : job.status;

    el.innerHTML = `
      <div class="job-row">
        <span class="job-name">${escapeHtml(job.filename)}</span>
        <span class="job-status status-${job.status}">${statusLabel}</span>
      </div>
      ${
        job.status === "processing" || job.status === "reading"
          ? `<div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>`
          : ""
      }
      ${job.status === "error" ? `<div class="job-error">${escapeHtml(job.error || "unknown error")}</div>` : ""}
      ${
        job.status === "done"
          ? `<div class="job-actions">
              ${job.available_formats?.includes("md") ? `<a href="/api/jobs/${job.id}/download/md" download>markdown</a>` : ""}
              ${job.available_formats?.includes("json") ? `<a href="/api/jobs/${job.id}/download/json" download>json</a>` : ""}
            </div>`
          : ""
      }
    `;
    queueList.appendChild(el);
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}