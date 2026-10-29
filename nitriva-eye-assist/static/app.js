/**
 * Nitriva Frontend Core Engine
 * Manages WebSocket telemetry, Eye-Gaze Pointer, Blink/Wink State Machine,
 * Calibration Protocol, Audio Speech Synthesis, and Smart Device Control.
 */

// Global State
const state = {
  currentView: "login", // 'login' | 'setup' | 'calibration' | 'dashboard' | 'communication' | 'devices'
  accessKey: "1234",
  isLoggedIn: false,
  selectedKeywords: [],
  activeDevices: { light: false, fan: false, ac: false, nurse: false },
  sosActive: false,
  gaze: { x: 0.5, y: 0.5 },
  lastFocusedElement: null,
  dwellTimer: null,
  dwellProgress: 0,
  calibrationIndex: 0,
  calibrationTargets: [],
  isCalibrating: false,
};

// WebSocket Connection
let socket = null;
let localMediaStream = null;
let isCloudWebcamActive = false;
let frameStreamCanvas = null;
let frameStreamInterval = null;
let frameInFlight = false;
let frameSentAt = 0;

async function initBrowserCamera() {
  if (localMediaStream) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.log("[Nitriva Camera] getUserMedia not supported in this environment.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });
    localMediaStream = stream;
    isCloudWebcamActive = true;

    // Attach stream to opticsVideo
    const opticsVideo = document.getElementById("opticsVideo");
    const opticsImage = document.getElementById("opticsImage");
    if (opticsVideo) {
      opticsVideo.srcObject = stream;
      opticsVideo.style.display = "block";
      if (opticsImage) {
        opticsImage.style.display = "none";
        opticsImage.removeAttribute("src");
      }
    }

    // Attach stream to setupVideo
    const setupVideo = document.getElementById("setupVideo");
    const setupImage = document.getElementById("setupImage");
    if (setupVideo) {
      setupVideo.srcObject = stream;
      setupVideo.style.display = "block";
      if (setupImage) {
        setupImage.style.display = "none";
        setupImage.removeAttribute("src");
      }
    }

    // Start streaming frames to backend over WebSocket at ~22 FPS
    startFrameStreaming(setupVideo || opticsVideo);
    console.log("[Nitriva Camera] Browser webcam stream active for cloud deployment.");
  } catch (err) {
    console.log("[Nitriva Camera] Browser camera initialization note:", err);
  }
}

function startFrameStreaming(videoElement) {
  if (frameStreamInterval) clearInterval(frameStreamInterval);
  if (!frameStreamCanvas) {
    frameStreamCanvas = document.createElement("canvas");
    frameStreamCanvas.width = 320;
    frameStreamCanvas.height = 240;
  }
  const ctx = frameStreamCanvas.getContext("2d");

  frameStreamInterval = setInterval(() => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    // Wait for inference before sending another frame. Slow shared CPUs must
    // drop frames rather than accumulate seconds of stale gaze input.
    if (frameInFlight) {
      if (performance.now() - frameSentAt > 10000) socket.close();
      return;
    }
    if (socket.bufferedAmount > 0) return;
    const v = (document.getElementById("setupVideo")?.videoWidth > 0)
      ? document.getElementById("setupVideo")
      : document.getElementById("opticsVideo");
    if (!v || v.videoWidth === 0) return;

    ctx.drawImage(v, 0, 0, 320, 240);
    const dataUrl = frameStreamCanvas.toDataURL("image/jpeg", 0.60);
    socket.send(JSON.stringify({ type: "frame", data: dataUrl }));
    frameInFlight = true;
    frameSentAt = performance.now();
  }, 45); // ~22 FPS streaming
}

function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    frameInFlight = false;
    console.log("[Nitriva WS] Connected to telemetry stream.");
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.frame_processed) frameInFlight = false;
      handleTelemetry(data);
    } catch (err) {
      console.error("[Nitriva WS] Message parse error:", err);
    }
  };

  socket.onclose = () => {
    frameInFlight = false;
    setTimeout(initWebSocket, 2000);
  };
}

let lastProcessedEvent = null;
let lastProcessedEventTime = 0;

// Telemetry Handling
function handleTelemetry(data) {
  // Update Optics Sensor UI
  const opticsPill = document.getElementById("opticsStatusPill");
  const opticsDot = document.getElementById("opticsDot");
  const opticsText = document.getElementById("opticsStatusText");

  if (opticsText && data.state_text) {
    opticsText.textContent = data.state_text;
    if (data.state_text.includes("BLINK")) {
      opticsPill.className = "optics-status-pill blink-active";
    } else if (data.state_text.includes("WINK")) {
      opticsPill.className = "optics-status-pill wink-active";
    } else {
      opticsPill.className = "optics-status-pill";
    }
  }

  // Update Gaze Coordinates (freeze card focus changes during blinks or winks so target doesn't slip)
  const isEyeBlinking = (data.blink_left > 0.35 || data.blink_right > 0.35);
  if (data.gaze_x !== undefined && data.gaze_y !== undefined) {
    state.gaze.x = data.gaze_x;
    state.gaze.y = data.gaze_y;
    if (!isEyeBlinking) {
      updateGazeCursor(data.gaze_x, data.gaze_y);
    }
  }

  // Check Discrete Events with 380ms client debounce
  if (data.event) {
    const now = performance.now();
    if (data.event !== lastProcessedEvent || (now - lastProcessedEventTime > 380)) {
      lastProcessedEvent = data.event;
      lastProcessedEventTime = now;
      handleGazeEvent(data.event);
    }
  }
}

// Stabilized Target Focus with Fixation Latching (Anti-Jitter)
let currentFocusedElement = null;
let candidateElement = null;
let candidateStreak = 0;
const SWITCH_THRESHOLD_FRAMES = 3; // ~90ms stable gaze in new zone before switching

function getClosestInteractiveTarget(normX, normY) {
  let activeContainer = null;
  if (state.currentView === "dashboard") {
    activeContainer = document.getElementById("dashboardContent");
  } else if (state.currentView === "communication") {
    activeContainer = document.getElementById("communicationContent");
  } else if (state.currentView === "devices") {
    activeContainer = document.getElementById("devicesContent");
  }

  if (!activeContainer) return null;

  const targets = Array.from(activeContainer.querySelectorAll(".interactive-target"));
  if (targets.length === 0) return null;

  const screenX = (normX !== undefined ? normX : 0.5) * window.innerWidth;
  const screenY = (normY !== undefined ? normY : 0.5) * window.innerHeight;

  let best = null;
  let minDist = Infinity;
  for (const t of targets) {
    const rect = t.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dist = Math.hypot(screenX - cx, screenY - cy);
    if (dist < minDist) {
      minDist = dist;
      best = t;
    }
  }
  return best;
}

function updateGazeCursor(normX, normY) {
  // During calibration or setup, skip normal module gaze focus
  if (state.currentView === "calibration" || state.currentView === "login" || state.currentView === "setup") {
    return;
  }

  const screenX = normX * window.innerWidth;
  const screenY = normY * window.innerHeight;

  // Get current active view container
  let activeContainer = null;
  if (state.currentView === "dashboard") {
    activeContainer = document.getElementById("dashboardContent");
  } else if (state.currentView === "communication") {
    activeContainer = document.getElementById("communicationContent");
  } else if (state.currentView === "devices") {
    activeContainer = document.getElementById("devicesContent");
  }

  if (!activeContainer) return;

  // Collect active interactive cards
  let targets = Array.from(activeContainer.querySelectorAll(".interactive-target"));

  // Include sidebar nav items if user's gaze is directed to the left sidebar (normX < 0.22)
  if (normX < 0.22) {
    const sidebarItems = Array.from(document.querySelectorAll(".sidebar .interactive-target"));
    targets.push(...sidebarItems);
  }

  if (targets.length === 0) return;

  // Proximity-based card focus with hysteresis
  let bestTarget = null;
  let minDistance = Infinity;

  for (const target of targets) {
    const rect = target.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;

    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;

    const dx = screenX - cx;
    const dy = screenY - cy;
    let dist = Math.hypot(dx, dy);

    // Hysteresis bias: currently focused card gets a 35% distance advantage to eliminate flicker
    if (target === currentFocusedElement) {
      dist *= 0.65;
    }

    if (dist < minDistance) {
      minDistance = dist;
      bestTarget = target;
    }
  }

  if (!bestTarget) return;

  // If candidate is already focused, keep it solid
  if (bestTarget === currentFocusedElement) {
    candidateElement = null;
    candidateStreak = 0;
    return;
  }

  // Accumulate frames before switching (eliminates transient eye tremors)
  if (bestTarget === candidateElement) {
    candidateStreak++;
    if (candidateStreak >= SWITCH_THRESHOLD_FRAMES) {
      if (currentFocusedElement) {
        currentFocusedElement.classList.remove("gaze-focused");
      }
      currentFocusedElement = bestTarget;
      currentFocusedElement.classList.add("gaze-focused");
      state.lastFocusedElement = currentFocusedElement;
      candidateElement = null;
      candidateStreak = 0;
    }
  } else {
    candidateElement = bestTarget;
    candidateStreak = 1;
  }
}

// Process eye events: Wink Left (Select), Double Blink (Back), Triple Blink (SOS)
function handleGazeEvent(eventCode) {
  console.log("[Nitriva Event]", eventCode);

  if (eventCode === "WINK_LEFT" || eventCode === "SELECT") {
    // Activate currently focused interactive card/button with visual confirmation
    let target = state.lastFocusedElement || currentFocusedElement || candidateElement;
    if (!target) {
      target = getClosestInteractiveTarget(state.gaze.x, state.gaze.y);
    }

    if (target) {
      target.classList.add("wink-pulse");
      setTimeout(() => {
        target.classList.remove("wink-pulse");
      }, 250);
      triggerElementAction(target);
    }
  } else if (eventCode === "WINK_RIGHT" || eventCode === "CLEAR") {
    if (state.currentView === "communication") {
      clearKeywords();
    } else {
      // On dashboard or devices, allow right wink to also select focused card
      let target = state.lastFocusedElement || currentFocusedElement || candidateElement;
      if (!target) {
        target = getClosestInteractiveTarget(state.gaze.x, state.gaze.y);
      }
      if (target) {
        target.classList.add("wink-pulse");
        setTimeout(() => {
          target.classList.remove("wink-pulse");
        }, 250);
        triggerElementAction(target);
      }
    }
  } else if (eventCode === "DOUBLE_BLINK") {
    // 2 Blinks = Back Navigation
    handleBackNavigation();
  } else if (eventCode === "TRIPLE_BLINK") {
    // 3 Blinks = Logout / Emergency Alert
    triggerEmergencySOS();
  }
}

// Trigger action on focused card
function triggerElementAction(el) {
  if (!el) return;
  const action = el.dataset.action;
  const keyword = el.dataset.keyword;
  const device = el.dataset.device;

  // Audio click feedback
  playSoftBeep(600, 0.08);

  if (action === "nav-dashboard") {
    switchView("dashboard");
  } else if (action === "nav-comm") {
    switchView("communication");
  } else if (action === "nav-devices") {
    switchView("devices");
  } else if (action === "trigger-sos") {
    triggerEmergencySOS();
  } else if (action === "system-prefs") {
    displayAndVocalize("Nitriva preferences: High-precision gaze and blink tracking active.");
  } else if (keyword) {
    handleKeywordSelect(keyword);
  } else if (device) {
    handleDeviceToggle(device);
  } else if (action === "login") {
    handleLogin();
  } else if (action === "start-calib") {
    startCalibrationFlow();
  } else if (action === "clear-comm") {
    clearKeywords();
  } else if (typeof el.click === "function") {
    el.click();
  }
}

// Back Navigation (2 Blinks)
function handleBackNavigation() {
  playSoftBeep(450, 0.1);
  if (state.sosActive) {
    dismissSOS();
    return;
  }
  if (state.currentView === "communication" || state.currentView === "devices") {
    switchView("dashboard");
  } else if (state.currentView === "dashboard") {
    switchView("login");
  }
}

// View Management
function switchView(viewName) {
  state.currentView = viewName;

  // Clear dark phase if leaving calibration
  if (viewName !== "calibration") {
    const calibScreen = document.getElementById("calibrationScreen");
    if (calibScreen) calibScreen.classList.remove("dark-phase");
  }

  // Screens
  document.getElementById("loginScreen").style.display = viewName === "login" ? "flex" : "none";
  document.getElementById("setupScreen").style.display = viewName === "setup" ? "flex" : "none";
  document.getElementById("calibrationScreen").style.display = viewName === "calibration" ? "flex" : "none";
  document.getElementById("dashboardContent").style.display = viewName === "dashboard" ? "flex" : "none";
  document.getElementById("communicationContent").style.display = viewName === "communication" ? "flex" : "none";
  document.getElementById("devicesContent").style.display = viewName === "devices" ? "flex" : "none";

  // Sidebar visibility
  const appContainer = document.getElementById("appContainer");
  const sidebar = document.getElementById("sidebar");
  if (viewName === "login" || viewName === "setup" || viewName === "calibration") {
    sidebar.style.display = "none";
  } else {
    sidebar.style.display = "flex";
  }

  // Update active sidebar nav pills
  document.querySelectorAll(".nav-item").forEach(item => {
    item.classList.remove("active");
    if (item.dataset.view === viewName) {
      item.classList.add("active");
    }
  });

  // Clear focused element & candidate latch
  if (currentFocusedElement) {
    currentFocusedElement.classList.remove("gaze-focused");
    currentFocusedElement = null;
  }
  candidateElement = null;
  candidateStreak = 0;
  if (state.lastFocusedElement) {
    state.lastFocusedElement.classList.remove("gaze-focused");
    state.lastFocusedElement = null;
  }
}

// Login Handler
function handleLogin() {
  const input = document.getElementById("patientAccessKey");
  const key = input ? input.value.trim() : "1234";

  state.isLoggedIn = true;
  playSoftBeep(650, 0.08);

  // Initialize client camera (seamless for cloud deployment)
  initBrowserCamera();

  // Background API call
  fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: key || "1234" })
  }).catch(() => {});

  // Immediate smooth transition to Patient Setup
  switchView("setup");
  initSetupScreen();
}

function skipCalibration() {
  state.isCalibrating = false;
  const calibScreen = document.getElementById("calibrationScreen");
  if (calibScreen) calibScreen.classList.remove("dark-phase");

  fetch("/api/calibrate/finish", { method: "POST" }).catch(() => {});
  playSoftBeep(880, 0.15);
  switchView("dashboard");
}

// Patient Setup Screen
function initSetupScreen() {
  const btn = document.getElementById("setupBtn");
  if (!btn) return;
  btn.textContent = "Booting AI...";
  btn.classList.remove("ready");

  setTimeout(() => {
    btn.textContent = "Booting AI...";
    // Automatically advance to calibration once AI models are primed (Screenshot 1)
    setTimeout(() => {
      if (state.currentView === "setup") {
        startCalibrationFlow();
      }
    }, 1200);
  }, 1600);
}

// Calibration Sequence (Dual-Phase: Light Mode + Dark/Black Mode)
async function startCalibrationFlow() {
  switchView("calibration");
  state.isCalibrating = true;

  const calibScreen = document.getElementById("calibrationScreen");
  if (calibScreen) calibScreen.classList.remove("dark-phase");

  try {
    const res = await fetch("/api/calibrate/targets");
    const data = await res.json();
    state.calibrationTargets = data.targets;
  } catch (e) {
    state.calibrationTargets = [
      // Phase 1: Light Mode (Screen White)
      { id: "light_center", phase: "light", name: "1. LOOK CENTER (Phase: Light Mode)", x: 0.50, y: 0.50 },
      { id: "light_top_left", phase: "light", name: "2. LOOK TOP LEFT (Phase: Light Mode)", x: 0.12, y: 0.18 },
      { id: "light_top_right", phase: "light", name: "3. LOOK TOP RIGHT (Phase: Light Mode)", x: 0.88, y: 0.18 },
      { id: "light_bottom_left", phase: "light", name: "4. LOOK BOTTOM LEFT (Phase: Light Mode)", x: 0.12, y: 0.82 },
      { id: "light_bottom_right", phase: "light", name: "5. LOOK BOTTOM RIGHT (Phase: Light Mode)", x: 0.88, y: 0.82 },

      // Phase 2: Dark / Black Mode (Screen Black)
      { id: "dark_center", phase: "dark", name: "1. LOOK CENTER (Phase: Dark Mode)", x: 0.50, y: 0.50 },
      { id: "dark_top_left", phase: "dark", name: "2. LOOK TOP LEFT (Phase: Dark Mode)", x: 0.12, y: 0.18 },
      { id: "dark_top_right", phase: "dark", name: "3. LOOK TOP RIGHT (Phase: Dark Mode)", x: 0.88, y: 0.18 },
      { id: "dark_bottom_left", phase: "dark", name: "4. LOOK BOTTOM LEFT (Phase: Dark Mode)", x: 0.12, y: 0.82 },
      { id: "dark_bottom_right", phase: "dark", name: "5. LOOK BOTTOM RIGHT (Phase: Dark Mode)", x: 0.88, y: 0.82 }
    ];
  }

  runCalibrationStep(0);
}

async function runCalibrationStep(idx) {
  const calibScreen = document.getElementById("calibrationScreen");

  if (idx >= state.calibrationTargets.length) {
    // Finished full dual-phase calibration
    if (calibScreen) calibScreen.classList.remove("dark-phase");
    await fetch("/api/calibrate/finish", { method: "POST" });
    state.isCalibrating = false;
    playSoftBeep(880, 0.2);
    switchView("dashboard");
    return;
  }

  const target = state.calibrationTargets[idx];
  const dot = document.getElementById("calibDot");
  const footer = document.getElementById("calibFooter");

  // Toggle Dark Mode / Black Mode vs Light Mode phase
  if (calibScreen) {
    if (target.phase === "dark") {
      calibScreen.classList.add("dark-phase");
    } else {
      calibScreen.classList.remove("dark-phase");
    }
  }

  dot.style.left = `${target.x * 100}%`;
  dot.style.top = `${target.y * 100}%`;
  footer.textContent = target.name;

  // Allow user to dwell gaze on dot for 1.8 seconds while gathering samples
  setTimeout(async () => {
    try {
      await fetch("/api/calibrate/point", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_id: target.id, samples_count: 15 })
      });
    } catch (e) { }

    playSoftBeep(700, 0.08);
    runCalibrationStep(idx + 1);
  }, 1800);
}

// Communication Board Keyword Selection
async function handleKeywordSelect(keyword) {
  if (keyword === "CLEAR / BACK") {
    clearKeywords();
    return;
  }

  const index = state.selectedKeywords.indexOf(keyword);
  if (index > -1) {
    // Deselect if already present
    state.selectedKeywords.splice(index, 1);
  } else {
    // Add (limit to 2 keywords maximum)
    if (state.selectedKeywords.length >= 2) {
      state.selectedKeywords.shift();
    }
    state.selectedKeywords.push(keyword);
  }

  updateCommUI();

  if (state.selectedKeywords.length > 0) {
    synthesizeAndSpeak(state.selectedKeywords);
  }
}

function clearKeywords() {
  state.selectedKeywords = [];
  updateCommUI();
  hideSpeechAlert();
}

function updateCommUI() {
  const chipsContainer = document.getElementById("commChips");
  chipsContainer.innerHTML = "";

  state.selectedKeywords.forEach(kw => {
    const chip = document.createElement("div");
    chip.className = "comm-chip";
    chip.innerHTML = `<span>${kw}</span>`;
    chipsContainer.appendChild(chip);
  });

  // Update card selected visual states
  document.querySelectorAll(".comm-card").forEach(card => {
    const kw = card.dataset.keyword;
    if (state.selectedKeywords.includes(kw)) {
      card.classList.add("selected");
    } else {
      card.classList.remove("selected");
    }
  });
}

// AI Sentence Generation and Voice Playback
async function synthesizeAndSpeak(keywords) {
  try {
    const res = await fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keywords })
    });
    const data = await res.json();
    displayAndVocalize(data.sentence);
  } catch (err) {
    const fallback = `I need assistance regarding ${keywords.join(" and ")}.`;
    displayAndVocalize(fallback);
  }
}

function displayAndVocalize(sentence) {
  const alertBox = document.getElementById("speechAlert");
  const alertText = document.getElementById("speechAlertText");
  if (alertBox && alertText) {
    alertText.textContent = sentence;
    alertBox.style.display = "flex";
  }

  // Vocalize aloud using Web Speech API
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(sentence);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.lang = "en-US";
    window.speechSynthesis.speak(utterance);
  }
}

function hideSpeechAlert() {
  const alertBox = document.getElementById("speechAlert");
  if (alertBox) {
    alertBox.style.display = "none";
  }
}

// Smart Devices Control
async function handleDeviceToggle(deviceId) {
  try {
    const res = await fetch("/api/devices/toggle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: deviceId })
    });
    const data = await res.json();
    state.activeDevices[deviceId] = data.state;
    updateDeviceCards();
    playSoftBeep(520, 0.12);
  } catch (e) { }
}

function updateDeviceCards() {
  document.querySelectorAll("[data-device]").forEach(card => {
    const dev = card.dataset.device;
    const isActive = state.activeDevices[dev];
    const statusText = card.querySelector(".device-status-text");

    if (isActive) {
      card.classList.add("selected");
      if (statusText) statusText.textContent = dev === "nurse" ? "CALLED" : "ON";
    } else {
      card.classList.remove("selected");
      if (statusText) statusText.textContent = dev === "nurse" ? "STANDBY" : "OFF";
    }
  });
}

// Emergency SOS System
let sirenInterval = null;

function triggerEmergencySOS() {
  state.sosActive = true;
  const overlay = document.getElementById("sosOverlay");
  overlay.style.display = "flex";

  // Loud emergency siren
  startEmergencySiren();

  // Vocal alert
  if ("speechSynthesis" in window) {
    const utterance = new SpeechSynthesisUtterance("Emergency! Care alert triggered! Patient requires immediate assistance!");
    utterance.rate = 1.1;
    utterance.pitch = 1.2;
    window.speechSynthesis.speak(utterance);
  }

  fetch("/api/sos", { method: "POST" }).catch(() => { });
}

function dismissSOS() {
  state.sosActive = false;
  const overlay = document.getElementById("sosOverlay");
  overlay.style.display = "none";
  stopEmergencySiren();
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

function startEmergencySiren() {
  stopEmergencySiren();
  let high = true;
  sirenInterval = setInterval(() => {
    playSoftBeep(high ? 900 : 600, 0.25, "sawtooth");
    high = !high;
  }, 350);
}

function stopEmergencySiren() {
  if (sirenInterval) {
    clearInterval(sirenInterval);
    sirenInterval = null;
  }
}

// Web Audio Helper for Audible Feedback
function playSoftBeep(freq = 600, duration = 0.1, type = "sine") {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) { }
}

// Keyboard / Simulation Shortcuts for Rapid Testing
window.addEventListener("keydown", (e) => {
  const key = e.key.toUpperCase();

  if (e.key === " " || key === "W") {
    // Wink Left (Select)
    handleGazeEvent("WINK_LEFT");
  } else if (key === "B") {
    // Single Blink
    handleGazeEvent("BLINK");
  } else if (e.key === "2") {
    // Double Blink (Back)
    handleGazeEvent("DOUBLE_BLINK");
  } else if (e.key === "3") {
    // Triple Blink (SOS)
    handleGazeEvent("TRIPLE_BLINK");
  } else if (e.key === "Escape") {
    handleBackNavigation();
  }
});

// Allow mouse to guide gaze cursor for instant testing
window.addEventListener("mousemove", (e) => {
  const normX = e.clientX / window.innerWidth;
  const normY = e.clientY / window.innerHeight;
  updateGazeCursor(normX, normY);
});

// Immediate card focus on hover
document.addEventListener("mouseover", (e) => {
  const target = e.target.closest(".interactive-target");
  if (target) {
    if (currentFocusedElement && currentFocusedElement !== target) {
      currentFocusedElement.classList.remove("gaze-focused");
    }
    currentFocusedElement = target;
    currentFocusedElement.classList.add("gaze-focused");
    state.lastFocusedElement = target;
  }
});

// Click fallback
window.addEventListener("click", (e) => {
  const interactive = e.target.closest(".interactive-target");
  if (interactive) {
    triggerElementAction(interactive);
  }
});

// Initialization
window.addEventListener("DOMContentLoaded", () => {
  initWebSocket();
  switchView("login");

  // Fetch initial devices state
  fetch("/api/devices")
    .then(r => r.json())
    .then(data => {
      state.activeDevices = data;
      updateDeviceCards();
    })
    .catch(() => { });
});
