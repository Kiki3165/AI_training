// UX upload + spinner + lightbox images
document.addEventListener("DOMContentLoaded", () => {
  // ---- Upload UX ----
  const dz = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const form = document.getElementById("upload-form");
  const btn = document.getElementById("analyze-btn");
  const spinner = document.getElementById("spinner");

  if (dz && fileInput) {
    dz.addEventListener("click", () => fileInput.click());
    ["dragenter", "dragover"].forEach(ev =>
      dz.addEventListener(ev, e => { e.preventDefault(); dz.style.borderColor = "#5b9cff"; })
    );
    ["dragleave", "drop"].forEach(ev =>
      dz.addEventListener(ev, e => { e.preventDefault(); dz.style.borderColor = ""; })
    );
    dz.addEventListener("drop", e => {
      const files = e.dataTransfer.files;
      if (files && files.length) fileInput.files = files;
    });
  }

  if (form && btn && spinner) {
    form.addEventListener("submit", () => {
      btn.setAttribute("disabled", "disabled");
      spinner.classList.remove("hidden");
    });
  }

  // ---- Lightbox pour les images des résultats ----
  const lb = document.getElementById("lightbox");
  const lbImg = document.getElementById("lightbox-img");
  document.querySelectorAll(".zoomable img").forEach(img => {
    img.addEventListener("click", () => {
      if (!lb || !lbImg) return;
      lbImg.src = img.src;
      lb.classList.remove("hidden");
    });
  });
  if (lb) lb.addEventListener("click", () => lb.classList.add("hidden"));
});
