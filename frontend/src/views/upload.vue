<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="logo-section">
        <img src="/LogoShoeSense.png" alt="ShoeSense AI Logo" class="logo" />
        <h1 class="brand">ShoeSense AI</h1>
      </div>
    </header>

    <!-- Upload Area -->
    <main class="main-content">
      <div class="upload-box" @dragover.prevent @drop.prevent="onDrop">
        <!-- Mode Kamera -->
        <div v-if="useCamera" class="camera-wrapper">
          <video ref="videoRef" autoplay playsinline></video>
          <button class="capture-btn" @click="capturePhoto">Ambil Foto</button>
          <button class="cancel-btn" @click="cancelCamera">✖</button>
        </div>

        <!-- Mode Loading -->
        <div v-else-if="loading" class="loading-overlay">
          <div class="scan-full">
            <img
              v-if="previewUrl"
              :src="previewUrl"
              class="scan-bg"
              alt="scanning"
            />

            <div class="scan-line"></div>
            <div class="scan-glow"></div>

            <p class="scan-text">{{ loadingText }}</p>
          </div>
        </div>

        <!-- Mode Upload -->
        <template v-else>
          <input
            type="file"
            id="fileInput"
            accept="image/*"
            hidden
            @change="onFileChange"
          />
          <label v-if="!previewUrl" for="fileInput" class="upload-placeholder">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="70"
              height="70"
              fill="#93c5fd"
              viewBox="0 0 24 24"
            >
              <path d="M19 15v4H5v-4H3v4a2 2 0 002 2h14a2 2 0 002-2v-4z" />
              <path
                d="M11 16h2V8l3.5 3.5 1.42-1.42L12 4.66 6.08 10.08 7.5 11.5 11 8z"
              />
            </svg>
            <h2>Unggah atau ambil foto sepatu kamu untuk dianalisis</h2>
            <p>Klik untuk memilih atau seret gambar ke sini</p>
          </label>

          <div v-if="previewUrl" class="preview-container">
            <img :src="previewUrl" alt="Preview" class="preview-img" />
            <button class="cancel-btn" @click="cancelUpload">✖</button>
          </div>
        </template>
      </div>

      <!-- Buttons -->
      <div class="action-buttons">
        <button
          class="analyze-btn"
          :disabled="loading || !selectedFile"
          @click="uploadImage"
        >
          {{ loading ? "Menganalisis..." : "Analisis Sekarang" }}
        </button>
        <button class="camera-btn" @click="toggleCamera">Foto Sekarang</button>
      </div>

      <!-- Hasil -->
      <transition name="fade-scale">
        <div v-if="result" ref="resultRef" class="result-card">
          <h2>Hasil Analisis</h2>
          <div class="result-field"><span>Bahan:</span> {{ result.bahan }}</div>
          <div class="result-field">
            <span>Tingkat Kebersihan:</span> {{ displayTingkat }}
          </div>
          <div class="result-field">
            <span>Rekomendasi:</span> {{ result.rekomendasi }}
          </div>
        </div>
      </transition>

      <p v-if="error" class="error-msg">{{ error }}</p>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onBeforeUnmount, nextTick } from "vue";

const API_BASE = "http://127.0.0.1:5000";
const selectedFile = ref(null);
const previewUrl = ref(null);
const useCamera = ref(false);
const videoRef = ref(null);
let stream = null;
const result = ref(null);
const error = ref(null);
const loading = ref(false);
const showHistory = ref(false);
const history = ref([]);
const resultRef = ref(null);

// teks dinamis saat loading
const loadingText = ref("AI sedang menganalisis gambar...");
const loadingMessages = [
  "Menganalisis bahan sepatu...",
  "Mendeteksi tingkat kebersihan...",
  "Mengenali tekstur permukaan...",
  "Menghasilkan rekomendasi perawatan...",
];
let loadingInterval;

const fallbackThumb =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    "<svg xmlns='http://www.w3.org/2000/svg' width='60' height='60'><rect width='100%' height='100%' rx='8' fill='#f3f4f6'/><text x='50%' y='55%' text-anchor='middle' fill='#9ca3af' font-size='10'>no image</text></svg>"
  );

const displayTingkat = computed(() => {
  if (!result.value) return "-";
  const t = (result.value.tingkat_kotor || "").toLowerCase();
  if (t.includes("banget") || t.includes("sangat")) return "Sangat Kotor";
  if (t.includes("kotor")) return "Kotor";
  return "Bersih";
});

function onDrop(e) {
  handleFile(e.dataTransfer.files[0]);
}
function onFileChange(e) {
  handleFile(e.target.files[0]);
}
function handleFile(file) {
  if (!file) return;
  selectedFile.value = file;
  previewUrl.value = URL.createObjectURL(file);
  result.value = null;
  error.value = null;
}

function cancelUpload() {
  previewUrl.value = null;
  selectedFile.value = null;
  result.value = null;
}

async function uploadImage() {
  if (!selectedFile.value)
    return (error.value = "Pilih gambar terlebih dahulu.");
  loading.value = true;
  error.value = null;
  result.value = null;

  let index = 0;
  loadingText.value = loadingMessages[0];
  loadingInterval = setInterval(() => {
    index = (index + 1) % loadingMessages.length;
    loadingText.value = loadingMessages[index];
  }, 1800);

  const formData = new FormData();
  formData.append("file", selectedFile.value);

  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Gagal menganalisis gambar");

    result.value = data.prediction;

    // simpan lokal
    const entry = {
      image: previewUrl.value,
      result: data.prediction,
      timestamp: new Date().toLocaleString(),
    };
    const local = JSON.parse(localStorage.getItem("history") || "[]");
    local.unshift(entry);
    localStorage.setItem("history", JSON.stringify(local));

    // tunggu render hasil lalu scroll
    await nextTick();
    resultRef.value?.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    error.value = err.message;
  } finally {
    clearInterval(loadingInterval);
    loading.value = false;
  }
}

function toggleCamera() {
  useCamera.value = !useCamera.value;
  if (useCamera.value) startCamera();
  else stopCamera();
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.value.srcObject = stream;
  } catch {
    error.value = "Tidak dapat mengakses kamera";
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

function cancelCamera() {
  stopCamera();
  useCamera.value = false;
}

function capturePhoto() {
  const video = videoRef.value;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    selectedFile.value = new File([blob], "photo.jpg", { type: "image/jpeg" });
    previewUrl.value = URL.createObjectURL(blob);
    stopCamera();
    useCamera.value = false;
  }, "image/jpeg");
}

onBeforeUnmount(stopCamera);

function toggleHistory() {
  const local = JSON.parse(localStorage.getItem("history") || "[]");
  history.value = local;
  showHistory.value = !showHistory.value;
}
function clearAllHistory() {
  localStorage.removeItem("history");
  history.value = [];
}

function deleteHistory(index) {
  history.value.splice(index, 1);
  localStorage.setItem("history", JSON.stringify(history.value));
}

function viewHistory(item) {
  result.value = item.result;
  previewUrl.value = item.image;
  showHistory.value = false;
  nextTick(() => resultRef.value?.scrollIntoView({ behavior: "smooth" }));
}
</script>

<style scoped>
/* ============================
   FULL FRAME SCANNER
   ============================ */

.scan-full {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  border-radius: 18px;
  overflow: hidden;
  background: rgba(240, 244, 255, 0.3);
  backdrop-filter: blur(4px);
  border: 2px solid rgba(96, 165, 250, 0.5);
  box-shadow: 0 0 25px rgba(96, 165, 250, 0.25);
  display: flex;
  align-items: center;
  justify-content: center;
}

/* foto sebagai background */
.scan-bg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  opacity: 0.45;
  filter: blur(2px);
}

/* garis scan */
.scan-line {
  position: absolute;
  top: -60px;
  left: 0;
  width: 100%;
  height: 80px;

  background: linear-gradient(
    to bottom,
    rgba(96, 165, 250, 0) 0%,
    rgba(96, 165, 250, 0.35) 50%,
    rgba(96, 165, 250, 0) 100%
  );
  animation: scanMove 2.3s ease-in-out infinite;
}

@keyframes scanMove {
  0% {
    top: -80px;
  }
  50% {
    top: 100%;
  }
  100% {
    top: -80px;
  }
}

/* glow bingkai */
.scan-glow {
  position: absolute;
  inset: 0;
  border-radius: 18px;
  box-shadow: 0 0 40px rgba(59, 130, 246, 0.35) inset;
  animation: glowPulse 2s ease-in-out infinite;
}

@keyframes glowPulse {
  0%,
  100% {
    opacity: 0.5;
  }
  50% {
    opacity: 1;
  }
}

/* teks loading */
.scan-text {
  z-index: 5;
  font-size: 1.1rem;
  font-weight: 600;
  color: #1e40af;
  text-shadow: 0 0 8px rgba(96, 165, 250, 0.5);
  animation: textPulse 1.5s ease-in-out infinite;
}

@keyframes textPulse {
  0%,
  100% {
    opacity: 0.45;
  }
  50% {
    opacity: 1;
  }
}

.preview-img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 16px;
}

.preview-container {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

/* GENERAL APP CONTAINER */
.app-container {
  background: #f9fafb;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  font-family: "Avenir", Helvetica, Arial, sans-serif;
  color: #1f2937;
}

/* =========================
   HEADER
   ========================= */
.app-header {
  background: #ffffff;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.9rem 2rem;
  border-bottom: 1px solid #e2e8f0;
  box-shadow: 0 1px 6px rgba(0, 0, 0, 0.05);
}

.logo-section {
  display: flex;
  align-items: center;
  gap: 0.7rem;
}

.logo {
  width: 34px;
  height: 34px;
  border-radius: 8px;
}

.brand {
  font-size: 1.35rem;
  font-weight: 700;
  font-family: "Poppins", sans-serif;
  letter-spacing: 0.3px;
  color: #1e40af; /* lebih elegan */
}

.history-btn:hover {
  background: #1d4ed8;
  box-shadow: 0 0 8px rgba(37, 99, 235, 0.35);
}

/* =========================
   UPLOAD MAIN AREA
   ========================= */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2.5rem 1rem;
}

.upload-box {
  background: #ffffff;
  border: 2px dashed #bfdbfe;
  border-radius: 18px;
  width: 580px;
  max-width: 90%;
  height: 330px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s ease, transform 0.2s ease;
}

.upload-box:hover {
  border-color: #60a5fa;
  transform: translateY(-2px);
}

.upload-placeholder h2 {
  margin-top: 1rem;
  font-size: 1.15rem;
  color: #1d4ed8;
  font-weight: 600;
}

.upload-placeholder p {
  color: #64748b;
  font-size: 0.9rem;
  margin-top: 0.35rem;
}

/* PREVIEW IMAGE */
.preview-container {
  position: absolute;
  inset: 0;
  background: #f1f5f9;
}

.preview-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 14px;
}

/* CANCEL BUTTON ON PREVIEW */
.cancel-btn {
  position: absolute;
  top: 12px;
  right: 12px;
  border: none;
  padding: 5px 10px;
  border-radius: 6px;
  font-weight: 600;
  cursor: pointer;
  background: rgba(0, 0, 0, 0.48);
  color: #fff;
  transition: 0.2s;
}

.cancel-btn:hover {
  background: rgba(239, 68, 68, 0.9);
}

/* =========================
   CAMERA MODE
   ========================= */
.camera-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
}

.camera-wrapper video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 14px;
}

.capture-btn {
  position: absolute;
  bottom: 14px;
  right: 14px;
  background: #2563eb;
  color: white;
  padding: 9px 14px;
  border-radius: 8px;
  border: none;
  font-weight: 600;
  cursor: pointer;
}

.capture-btn:hover {
  background: #1e40af;
}

/* =========================
   LOADING OVERLAY
   ========================= */
.loading-overlay {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.spinner {
  width: 55px;
  height: 55px;
  border: 5px solid #dbeafe;
  border-top-color: #2563eb;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.loading-text {
  margin-top: 1rem;
  color: #1d4ed8;
  font-size: 1rem;
  font-weight: 500;
  animation: pulse 1.4s infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@keyframes pulse {
  50% {
    opacity: 0.45;
  }
}

/* =========================
   BUTTONS
   ========================= */
.action-buttons {
  display: flex;
  gap: 1rem;
  margin-top: 1.8rem;
}

.analyze-btn,
.camera-btn {
  padding: 13px 22px;
  border-radius: 10px;
  font-size: 1rem;
  border: none;
  cursor: pointer;
  transition: 0.25s ease;
}

.analyze-btn {
  background: #2563eb;
  color: white;
}

.analyze-btn:hover:not(:disabled) {
  background: #1d4ed8;
  transform: translateY(-2px);
}

.analyze-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.camera-btn {
  background: #e2e8f0;
}

.camera-btn:hover {
  background: #cbd5e1;
  transform: translateY(-2px);
}

/* =========================
   RESULT CARD
   ========================= */
.result-card {
  background: #ffffff;
  padding: 1.6rem 2rem;
  border-radius: 16px;
  margin-top: 2rem;
  width: 480px;
  max-width: 95%;
  box-shadow: 0 3px 18px rgba(0, 0, 0, 0.08);
  animation: fadeIn 0.35s ease;
}

.result-card h2 {
  color: #1d4ed8;
  margin-bottom: 1rem;
  font-size: 1.3rem;
  font-weight: 600;
}

.result-field {
  margin-bottom: 0.5rem;
  font-size: 1rem;
}

.result-field span {
  font-weight: 600;
  color: #0f172a;
}

.error-msg {
  margin-top: 10px;
  color: #dc2626;
  font-weight: 500;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* =========================
   HISTORY SIDEBAR
   ========================= */
.history-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: 330px;
  height: 100%;
  background: #ffffff;
  border-left: 1px solid #e5e7eb;
  box-shadow: -3px 0 12px rgba(0, 0, 0, 0.08);
  padding: 1rem;
  overflow-y: auto;
  animation: slideIn 0.35s ease;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
  }
  to {
    transform: translateX(0);
  }
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.clear-all {
  background: #ef4444;
  color: #fff;
  padding: 6px 10px;
  border: none;
  border-radius: 8px;
  font-size: 0.8rem;
  cursor: pointer;
}

.clear-all:hover {
  background: #dc2626;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.15rem;
  cursor: pointer;
  color: #475569;
}

.history-item {
  display: flex;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid #e5e7eb;
  cursor: pointer;
  transition: 0.2s ease;
}

.history-item:hover {
  background: #f8fafc;
}

.thumb {
  width: 58px;
  height: 58px;
  object-fit: cover;
  border-radius: 8px;
}

.delete-btn {
  border: none;
  background: none;
  cursor: pointer;
  color: #ef4444;
  font-size: 1rem;
}

.delete-btn:hover {
  color: #dc2626;
}

/* No History */
.empty-history {
  color: #94a3b8;
  text-align: center;
  margin-top: 2rem;
}

/* TRANSITIONS */
.slide-enter-active,
.slide-leave-active {
  transition: transform 0.35s ease;
}
.slide-enter-from,
.slide-leave-to {
  transform: translateX(100%);
}
.fade-scale-enter-active {
  transition: all 0.3s ease;
}
.fade-scale-enter-from {
  opacity: 0;
  transform: scale(0.95);
}
.fade-scale-leave-active {
  transition: all 0.2s ease;
}
.fade-scale-leave-to {
  opacity: 0;
  transform: scale(0.95);
}
</style>
