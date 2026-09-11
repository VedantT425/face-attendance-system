/**
 * main.js — Shared JS utilities for FaceAttend
 */

/**
 * Show a toast notification at the bottom-right.
 * @param {string} message
 * @param {"success"|"error"|"info"} type
 */
function showToast(message, type = "success") {
  const toast = document.getElementById("toast");
  if (!toast) return;

  const colors = {
    success: { bg: "rgba(16,185,129,0.15)", border: "#10b981", color: "#10b981" },
    error:   { bg: "rgba(239,68,68,0.15)",  border: "#ef4444", color: "#ef4444" },
    info:    { bg: "rgba(59,130,246,0.15)", border: "#3b82f6", color: "#3b82f6" },
  };
  const c = colors[type] || colors.info;
  toast.style.background = c.bg;
  toast.style.borderColor = c.border;
  toast.style.color = c.color;
  toast.textContent = message;
  toast.classList.remove("hidden");

  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.add("hidden"), 3500);
}
