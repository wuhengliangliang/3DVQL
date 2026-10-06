const topButton = document.querySelector(".scroll-to-top");

window.addEventListener("scroll", () => {
  topButton.classList.toggle("visible", window.scrollY > 700);
});

topButton.addEventListener("click", () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
});

const copyButton = document.querySelector(".copy-bibtex-btn");
copyButton.addEventListener("click", async () => {
  const text = document.querySelector("#bibtex-code").innerText;
  await navigator.clipboard.writeText(text);
  copyButton.textContent = "Copied";
  setTimeout(() => {
    copyButton.textContent = "Copy";
  }, 1200);
});
