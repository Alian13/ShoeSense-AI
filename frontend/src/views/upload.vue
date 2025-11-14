<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="logo-section">
        <img src="/LogoShoeSense.png" alt="ShoeSense AI Logo" class="logo" />
        <h1 class="brand">ShoeSense AI</h1>
      </div>
      <button class="history-btn" @click="toggleHistory">
        <span>Riwayat</span>
      </button>
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
          <div class="spinner"></div>
          <p class="loading-text">{{ loadingText }}</p>
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
          {{ loading ? "Menganalisis..." : "🔍 Analisis Sekarang" }}
        </button>
        <button class="camera-btn" @click="toggleCamera">
          📸 Foto Sekarang
        </button>
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
            <span>Keyakinan:</span>
            {{
              (
                ((result.confidence_bahan || 0) +
                  (result.confidence_kotor || 0)) /
                2
              ).toFixed(1)
            }}%
          </div>
          <div class="result-field">
            <span>Rekomendasi:</span> {{ result.rekomendasi }}
          </div>
        </div>
      </transition>

      <p v-if="error" class="error-msg">{{ error }}</p>
    </main>

    <!-- Riwayat -->
    <transition name="slide">
      <aside v-if="showHistory" class="history-panel">
        <div class="history-header">
          <h2>Riwayat Analisis</h2>
          <div class="history-actions">
            <button class="clear-all" @click="clearAllHistory">
              Hapus Semua
            </button>
            <button class="close-btn" @click="toggleHistory">✖</button>
          </div>
        </div>
        <div v-if="!history.length" class="empty-history">
          Belum ada riwayat.
        </div>
        <div
          v-for="(item, i) in history"
          :key="i"
          class="history-item"
          @click="viewHistory(item)"
        >
          <img :src="item.image || fallbackThumb" alt="thumb" class="thumb" />
          <div class="history-info">
            <strong>{{ item.result.bahan }}</strong>
            <p>{{ item.result.tingkat_kotor }}</p>
            <small>{{ item.timestamp }}</small>
          </div>
          <button class="delete-btn" @click.stop="deleteHistory(i)">🗑️</button>
        </div>
      </aside>
    </transition>
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
/* General */
.app-container {
  background: #f8fafc;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  font-family: "Poppins", sans-serif;
  color: #1e293b;
}

/* Header */
.app-header {
  background: #ffffff;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.8rem 2rem;
  border-bottom: 1px solid #e5e7eb;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.logo-section {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.logo {
  width: 32px;
  height: 32px;
}
.brand {
  font-size: 1.2rem;
  font-weight: 600;
  color: #1d4ed8;
}
.history-btn {
  background: #3b82f6;
  color: #fff;
  border: none;
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
}

/* Main */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}
.upload-box {
  background: #fff;
  border: 2px dashed #93c5fd;
  border-radius: 18px;
  width: 600px;
  max-width: 90%;
  height: 340px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}
.upload-placeholder h2 {
  font-size: 1.15rem;
  margin-top: 1rem;
  color: #1e3a8a;
}
.upload-placeholder p {
  color: #6b7280;
  margin-top: 0.4rem;
}

/* Loading overlay */
.loading-overlay {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 100%;
  animation: fadeIn 0.3s ease;
}
.spinner {
  width: 64px;
  height: 64px;
  border: 6px solid #dbeafe;
  border-top-color: #2563eb;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}
.loading-text {
  margin-top: 1rem;
  color: #1e3a8a;
  font-weight: 500;
  font-size: 1rem;
  animation: pulse 1.6s ease-in-out infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

/* Camera */
.camera-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
}
.camera-wrapper video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 16px;
}
.capture-btn,
.cancel-btn {
  position: absolute;
  border: none;
  border-radius: 8px;
  padding: 8px 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s ease;
}
.capture-btn {
  bottom: 14px;
  right: 14px;
  background: #2563eb;
  color: #fff;
}
.cancel-btn {
  top: 10px;
  right: 10px;
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
}
.cancel-btn:hover {
  background: rgba(220, 38, 38, 0.85);
}

/* Buttons */
.action-buttons {
  display: flex;
  gap: 1rem;
  margin-top: 1.5rem;
}
.analyze-btn,
.camera-btn {
  padding: 12px 20px;
  border: none;
  border-radius: 10px;
  font-size: 1rem;
  cursor: pointer;
  transition: transform 0.15s ease, background 0.2s ease;
}
.analyze-btn {
  background: #3b82f6;
  color: #fff;
}
.analyze-btn:hover {
  background: #2563eb;
  transform: translateY(-2px);
}
.camera-btn {
  background: #e2e8f0;
}
.camera-btn:hover {
  background: #cbd5e1;
  transform: translateY(-2px);
}

/* Result */
.result-card {
  background: #fff;
  border-radius: 16px;
  padding: 1.5rem 2rem;
  width: 500px;
  margin-top: 2rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  animation: fadeIn 0.6s ease;
}
.result-card h2 {
  color: #1e3a8a;
  margin-bottom: 1rem;
}
.result-field {
  margin-bottom: 0.5rem;
}
.result-field span {
  font-weight: 600;
  color: #0f172a;
}
.error-msg {
  color: #dc2626;
  margin-top: 10px;
}

/* Fade-in + scale animation */
.fade-scale-enter-active {
  transition: all 0.6s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.fade-scale-enter-from {
  opacity: 0;
  transform: scale(0.9);
}

/* History panel */
.history-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: 320px;
  height: 100%;
  background: #fff;
  border-left: 1px solid #e5e7eb;
  box-shadow: -4px 0 12px rgba(0, 0, 0, 0.06);
  padding: 1rem;
  overflow-y: auto;
}
.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.close-btn {
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
}
.history-item {
  display: flex;
  gap: 10px;
  align-items: center;
  border-bottom: 1px solid #e5e7eb;
  padding: 8px 0;
}
.thumb {
  width: 60px;
  height: 60px;
  object-fit: cover;
  border-radius: 8px;
}
.history-info p {
  margin: 0;
  font-size: 0.9rem;
  color: #475569;
}
.empty-history {
  color: #94a3b8;
  text-align: center;
  margin-top: 2rem;
}

/* Animations */
.slide-enter-active,
.slide-leave-active {
  transition: transform 0.4s ease;
}
.slide-enter-from,
.slide-leave-to {
  transform: translateX(100%);
}

/* Perbaikan tombol Riwayat di navbar */
.history-btn {
  background: #2563eb;
  color: #fff;
  border: none;
  padding: 8px 18px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: 0.3s ease;
}
.history-btn:hover {
  background: #1e40af;
  box-shadow: 0 0 10px rgba(37, 99, 235, 0.4);
}

/* Tombol dan layout riwayat */
.history-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
.clear-all {
  background: #ef4444;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 0.8rem;
  cursor: pointer;
}
.clear-all:hover {
  background: #dc2626;
}
.delete-btn {
  margin-left: auto;
  background: none;
  border: none;
  color: #ef4444;
  font-size: 1rem;
  cursor: pointer;
  transition: 0.2s;
}
.delete-btn:hover {
  color: #dc2626;
}
.history-item {
  cursor: pointer;
  transition: background 0.2s ease;
}
.history-item:hover {
  background: #f1f5f9;
}
</style>
