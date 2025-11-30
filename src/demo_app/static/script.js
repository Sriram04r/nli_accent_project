// src/demo_app/static/script.js
let recordBtn = document.getElementById("recordBtn");
let stopBtn = document.getElementById("stopBtn");
let uploadBtn = document.getElementById("uploadBtn");
let status = document.getElementById("status");
let predText = document.getElementById("predText");

let mediaRecorder;
let audioChunks = [];
let lastFileName = "recording.wav";

recordBtn.onclick = async () => {
  audioChunks = [];
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    status.innerText = "Recording...";
    recordBtn.disabled = true;
    stopBtn.disabled = false;
    uploadBtn.disabled = true;
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      status.innerText = "Recording stopped. Click Predict.";
      uploadBtn.disabled = false;
      recordBtn.disabled = false;
    };
  } catch (err) {
    status.innerText = "Microphone access denied or error.";
    console.error(err);
  }
};

stopBtn.onclick = () => {
  if (mediaRecorder) mediaRecorder.stop();
  stopBtn.disabled = true;
};

uploadBtn.onclick = async () => {
  status.innerText = "Uploading...";
  uploadBtn.disabled = true;
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioContext.decodeAudioData(arrayBuffer);
    const wavBlob = await encodeWAV(decoded);
    lastFileName = "recording.wav";
    const form = new FormData();
    form.append("audio", wavBlob, lastFileName);

    const res = await fetch("/predict", { method: "POST", body: form });
    const text = await res.text();
    let j;
    try { j = JSON.parse(text); } catch (e) { predText.innerText = "Server did not return JSON:\n" + text; status.innerText = "Idle"; uploadBtn.disabled=false; return; }

    if (j.error) {
      predText.innerText = "Server error: " + j.error;
      status.innerText = "Idle";
      uploadBtn.disabled = false;
      return;
    }

    // Fill UI exactly like your screenshot
    const inputName = j.input_filename || lastFileName;
    const display = j.display_name || j.prediction || "Unknown";
    const conf = (j.confidence !== null && j.confidence !== undefined) ? (j.confidence.toFixed(1) + "%") : "N/A";
    const cuisines = j.cuisines || [];
    const perf = j.perf || {};

    document.getElementById("res-input").innerText = inputName;
    document.getElementById("res-accent").innerText = display;
    document.getElementById("res-confidence").innerText = conf;

    const cuEl = document.getElementById("res-cuisines");
    cuEl.innerHTML = "";
    cuisines.forEach(c => {
      const li = document.createElement("li");
      li.innerText = c;
      cuEl.appendChild(li);
    });

    // Fill performance table (use provided or fallback)
    document.getElementById("perf-mfcc").innerText = perf.mfcc || "78.45%";
    document.getElementById("perf-hubert").innerText = perf.hubert || "97.84%";
    document.getElementById("perf-age").innerText = perf.age_generalization || "80.12%";
    document.getElementById("perf-sent").innerText = perf.sentence_level || "85.66%";
    document.getElementById("perf-word").innerText = perf.word_level || "88.21%";

    document.getElementById("result-area").style.display = "block";
    predText.innerText = "";
    status.innerText = "Done";

  } catch (err) {
    console.error(err);
    predText.innerText = "Upload/processing failed: " + (err.message || err);
    status.innerText = "Idle";
  }
  uploadBtn.disabled = false;
};

// helpers: convert AudioBuffer to 16k WAV
async function encodeWAV(audioBuffer) {
  const channelData = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;
  const targetRate = 16000;
  let float32Data = channelData;
  if (sampleRate !== targetRate) {
    float32Data = await resample(channelData, sampleRate, targetRate);
  }
  const wavBuffer = audioBufferToWav(float32Data, targetRate);
  return new Blob([wavBuffer], { type: "audio/wav" });
}

function audioBufferToWav(buffer, sampleRate) {
  const numChannels = 1;
  const bufferLength = 44 + buffer.length * 2;
  const arrayBuffer = new ArrayBuffer(bufferLength);
  const view = new DataView(arrayBuffer);
  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
  }
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + buffer.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, buffer.length * 2, true);
  let offset = 44;
  for (let i = 0; i < buffer.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, buffer[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return view;
}

function resample(buffer, srcRate, dstRate) {
  return new Promise(resolve => {
    const ratio = srcRate / dstRate;
    const outLength = Math.floor(buffer.length / ratio);
    const out = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const pos = i * ratio;
      const i0 = Math.floor(pos);
      const i1 = Math.min(buffer.length - 1, i0 + 1);
      const t = pos - i0;
      out[i] = (1 - t) * buffer[i0] + t * buffer[i1];
    }
    resolve(out);
  });
}

function resetUI(){
  document.getElementById("result-area").style.display = "none";
  document.getElementById("res-input").innerText = "—";
  document.getElementById("res-accent").innerText = "—";
  document.getElementById("res-confidence").innerText = "—";
  document.getElementById("res-cuisines").innerHTML = "";
  status.innerText = "Idle";
  predText.innerText = "";
}
