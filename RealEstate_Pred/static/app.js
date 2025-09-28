// Upload AJAX avec barre de progression, état "Traitement…", et lightbox images
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-input");
  const dz = document.getElementById("dropzone");
  const btn = document.getElementById("analyze-btn");
  const spinner = document.getElementById("spinner");
  const errBox = document.getElementById("error-box");

  const progressWrap = document.getElementById("progress-wrap");
  const progressBar = document.getElementById("progress-bar");
  const progressText = document.getElementById("progress-text");

  // drag & drop
  if (dz && fileInput) {
    dz.addEventListener("click", () => fileInput.click());
    ["dragenter","dragover"].forEach(ev =>
      dz.addEventListener(ev, e => { e.preventDefault(); dz.style.borderColor="#5b9cff"; })
    );
    ["dragleave","drop"].forEach(ev =>
      dz.addEventListener(ev, e => { e.preventDefault(); dz.style.borderColor=""; })
    );
    dz.addEventListener("drop", e => {
      const files = e.dataTransfer.files;
      if (files && files.length) fileInput.files = files;
    });
  }

  // soumission AJAX avec progress
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      errBox && errBox.classList.add("hidden");

      const file = fileInput?.files?.[0];
      if (!file) {
        if (errBox) { errBox.textContent = "Aucun fichier CSV sélectionné."; errBox.classList.remove("hidden"); }
        return;
      }

      // UI: démarrage upload
      btn?.setAttribute("disabled","disabled");
      spinner?.classList.remove("hidden");

      progressWrap?.classList.remove("hidden");
      progressBar.style.width = "0%";
      progressText.textContent = "0%";
      progressText.style.color = "#0b0f14";

      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/analyze");

      // progression d'upload
      xhr.upload.onprogress = (ev) => {
        if (!ev.lengthComputable) return;
        const pct = Math.round((ev.loaded / ev.total) * 100);
        progressBar.style.width = pct + "%";
        progressText.textContent = pct + "%";
      };

      xhr.onreadystatechange = () => {
        if (xhr.readyState === XMLHttpRequest.DONE) {
          // l'upload est fini depuis longtemps; ici la requête se termine après le traitement serveur
          spinner?.classList.add("hidden");

          try {
            const res = JSON.parse(xhr.responseText || "{}");
            if (xhr.status >= 200 && xhr.status < 300 && res.redirect) {
              window.location.href = res.redirect; // go résultats
            } else {
              throw new Error(res.error || "Erreur serveur.");
            }
          } catch (err) {
            btn?.removeAttribute("disabled");
            if (errBox) { errBox.textContent = err.message || "Erreur inconnue."; errBox.classList.remove("hidden"); }
          }
        }
      };

      xhr.onerror = () => {
        btn?.removeAttribute("disabled");
        spinner?.classList.add("hidden");
        if (errBox) { errBox.textContent = "Erreur réseau."; errBox.classList.remove("hidden"); }
      };

      // on envoie le fichier
      const data = new FormData();
      data.append("file", file);
      xhr.send(data);

      // Une fois l'upload terminé (progress 100%), on affiche "Traitement…"
      xhr.upload.onload = () => {
        progressText.textContent = "Traitement…";
        progressText.style.color = "#eaf1ff";
      };
    });
  }

  // Lightbox résultats
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
