"use strict";

// CONFIG 키 → input id 매핑 (id가 키와 동일하므로 그대로 사용)
const NUMS = ["base_width", "video_length_frames", "video_fps", "video_steps",
  "guidance_scale", "interpolate_to_fps", "upscale_factor",
  "contrast", "saturation", "brightness"];
const STRS = ["image_prompt", "image_model", "image_seed", "aspect",
  "model_id", "motion_prompt", "gguf_quant"];
const BOOLS = ["use_rife", "upscale_video"];

let pollTimer = null;

const $ = (id) => document.getElementById(id);

// ---- 폼 초기화: 선택지 + 기본값 채우기 ----
async function initForm() {
  const res = await fetch("/api/options");
  const { image_models, aspects, wan_models, gguf_quants, defaults } = await res.json();

  fillSelect("image_model", image_models);
  fillSelect("aspect", aspects);
  fillSelect("model_id", wan_models, shortWan);
  fillSelect("gguf_quant", gguf_quants || [""], (q) => q ? q : "사용 안 함 (bf16)");

  for (const k of [...STRS, ...NUMS]) {
    if (k in defaults && defaults[k] !== null) $(k).value = defaults[k];
  }
  for (const k of BOOLS) $(k).checked = !!defaults[k];
  $("upscale_factor").value = String(defaults.upscale_factor || 2);
  if (defaults.image_seed !== null) $("image_seed").value = defaults.image_seed;
}

function fillSelect(id, items, labelFn) {
  const sel = $(id);
  sel.innerHTML = "";
  for (const it of items) {
    const o = document.createElement("option");
    o.value = it;
    o.textContent = labelFn ? labelFn(it) : it;
    sel.appendChild(o);
  }
}

// "Wan-AI/Wan2.2-TI2V-5B-Diffusers" → "Wan2.2-TI2V-5B (경량)"
function shortWan(id) {
  const tail = id.split("/").pop().replace("-Diffusers", "");
  const note = tail.includes("TI2V-5B") ? " (경량·12GB)" : " (고품질)";
  return tail + note;
}

// ---- 폼 → CONFIG dict ----
function buildConfig() {
  const cfg = {};
  for (const k of STRS) cfg[k] = $(k).value;
  for (const k of NUMS) cfg[k] = Number($(k).value);
  for (const k of BOOLS) cfg[k] = $(k).checked;
  return cfg;
}

// ---- 제출 ----
$("job-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("submit-btn").disabled = true;

  const fd = new FormData();
  fd.append("config", JSON.stringify(buildConfig()));
  const bgm = $("bgm").files[0];
  if (bgm) fd.append("bgm", bgm);

  try {
    const res = await fetch("/api/jobs", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const { job_id } = await res.json();
    watchJob(job_id);
    refreshList();
  } catch (err) {
    alert("잡 생성 실패: " + err.message);
  } finally {
    $("submit-btn").disabled = false;
  }
});

// ---- 잡 폴링 ----
function watchJob(jobId) {
  $("status-card").hidden = false;
  $("job-id").textContent = "#" + jobId;
  $("result").hidden = true;
  $("error").hidden = true;
  $("status-card").scrollIntoView({ behavior: "smooth" });

  if (pollTimer) clearInterval(pollTimer);
  const tick = async () => {
    const res = await fetch("/api/jobs/" + jobId);
    if (!res.ok) return;
    const job = await res.json();
    renderJob(job);
    if (job.status === "done" || job.status === "error") {
      clearInterval(pollTimer);
      pollTimer = null;
      refreshList();
    }
  };
  tick();
  pollTimer = setInterval(tick, 2000);
}

function renderJob(job) {
  // 단계 진행바
  const ol = $("stages");
  ol.innerHTML = "";
  for (const s of job.stages) {
    const li = document.createElement("li");
    const dot = document.createElement("span");
    dot.className = "dot " + (s.status === "pending" ? "" : s.status);
    li.appendChild(dot);
    li.appendChild(document.createTextNode(s.label));
    ol.appendChild(li);
  }

  // 로그
  $("log").textContent = (job.log || []).join("\n");

  // 결과
  if (job.status === "done" && job.final_url) {
    $("result").hidden = false;
    $("result-video").src = job.final_url;
    $("download-link").href = job.final_url;
  }

  // 에러
  if (job.status === "error") {
    $("error").hidden = false;
    $("error").textContent = job.error || "알 수 없는 오류";
  }
}

// ---- 최근 잡 목록 ----
async function refreshList() {
  const res = await fetch("/api/jobs");
  const jobs = await res.json();
  const ul = $("job-list");
  ul.innerHTML = "";
  if (!jobs.length) {
    ul.innerHTML = '<li class="muted">아직 잡이 없습니다.</li>';
    return;
  }
  for (const j of jobs) {
    const li = document.createElement("li");
    li.onclick = () => watchJob(j.id);
    const left = document.createElement("span");
    left.textContent = (j.prompt || j.id);
    const badge = document.createElement("span");
    badge.className = "badge " + j.status;
    badge.textContent = j.status;
    li.append(left, badge);
    ul.appendChild(li);
  }
}

initForm();
refreshList();
