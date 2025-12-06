// frontend logic: upload PDF -> /initialize ; ask question -> /ask
const uploadBox = document.getElementById("uploadBox");
const pdfInput = document.getElementById("pdfInput");
const uploadText = document.getElementById("uploadText");
const initBtn = document.getElementById("initBtn");
const askBtn = document.getElementById("askBtn");
const questionEl = document.getElementById("question");
const responseBox = document.getElementById("responseBox");
const responseText = document.getElementById("responseText");

let selectedFile = null;
let sessionId = null;

// Click or drop to select
uploadBox.addEventListener("click", () => pdfInput.click());

pdfInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    selectedFile = e.target.files[0];
    uploadText.innerText = `📄 Selected: ${selectedFile.name}`;
  }
});

// support drag & drop
uploadBox.addEventListener("dragover", (e) => { e.preventDefault(); uploadBox.style.borderColor = "#2563eb"; });
uploadBox.addEventListener("dragleave", () => { uploadBox.style.borderColor = ""; });
uploadBox.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && f.type === "application/pdf") {
    selectedFile = f;
    uploadText.innerText = `📄 Selected: ${f.name}`;
    pdfInput.files = e.dataTransfer.files; // keep input updated
  } else {
    alert("Please drop a PDF file.");
  }
});

// Initialize (upload PDF to backend)
initBtn.addEventListener("click", async () => {
  if (!selectedFile) { alert("Please upload a PDF first."); return; }

  const form = new FormData();
  form.append("model", "groq");               // or "ollama"/"perplexity" if available
  form.append("pdf_file", selectedFile);

  try {
    const res = await fetch("http://localhost:8000/initialize", { method: "POST", body: form });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`Init failed: ${res.status} ${txt}`);
    }
    const data = await res.json();
    sessionId = data.session_id;
    localStorage.setItem("session_id", sessionId);
    alert("Initialized. Session id saved.");
  } catch (err) {
    console.error(err);
    alert("Initialization error: see console.");
  }
});

// Ask
askBtn.addEventListener("click", async () => {
  const q = questionEl.value.trim();
  if (!q) { alert("Type a question"); return; }
  if (!sessionId) {
    sessionId = localStorage.getItem("session_id");
    if (!sessionId) { alert("Initialize first (upload PDF)."); return; }
  }

  try {
    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question: q }),
    });
    if (!res.ok) { const t = await res.text(); throw new Error(t || res.status); }
    const data = await res.json();
    responseText.innerText = data.answer || "[no answer]";
    responseBox.classList.remove("hidden");
  } catch (err) {
    console.error("Ask error:", err);
    alert("Ask failed — see console.");
  }
});
