/**
 * PRNN Self-Learning Car Game — Main Loop & Learning Controller
 * Stuart-Landau dynamics, Continuous CHL, Annealed Noise
 */

// Core
const baseSpeedDefault = 3.5;
const prnn = new PRNN();
const car = new Car(400, 100, 0);
car.baseSpeed = baseSpeedDefault;
let track = new Track('oval');

// Canvases
const gameCanvas = document.getElementById('gameCanvas');
const gameCtx = gameCanvas.getContext('2d');
const networkCanvas = document.getElementById('networkCanvas');
const networkCtx = networkCanvas.getContext('2d');

// State
let isRunning = false;
let manualDrive = false;
let trainingEnabled = true;
let trainingMode = 'contrastive';
let crashCount = 0;
let survivalFrames = 0;
let maxSurvivalTime = 0;
let totalDistance = 0.0;
let totalFrames = 0;
const dangerThreshold = 0.40;
let noiseLevel = 0.05;
const initialNoise = 0.08;
let noiseScheduleActive = true;

const keys = { ArrowLeft: false, ArrowRight: false };

// Network visualisation
const nodePositions = [
    { name: 'L Sensor', x: 60, y: 110, color: '#10b981' },
    { name: 'C Sensor', x: 60, y: 240, color: '#10b981' },
    { name: 'R Sensor', x: 60, y: 370, color: '#10b981' },
    { name: 'Ref Bias', x: 180, y: 240, color: '#60a5fa' },
    { name: 'Hidden 1', x: 300, y: 110, color: '#8b5cf6' },
    { name: 'Hidden 2', x: 300, y: 240, color: '#8b5cf6' },
    { name: 'Hidden 3', x: 300, y: 370, color: '#8b5cf6' },
    { name: 'Steering', x: 430, y: 240, color: '#ec4899' }
];

// ─── Event Listeners ───
document.getElementById('playPauseBtn').addEventListener('click', toggleRun);
document.getElementById('resetBtn').addEventListener('click', resetSimulation);
document.getElementById('trackSelect').addEventListener('change', changeTrack);
document.getElementById('trainingModeSelect').addEventListener('change', changeTrainingMode);
document.getElementById('manualDriveBtn').addEventListener('click', toggleManualDrive);
document.getElementById('clearWeightsBtn').addEventListener('click', clearWeights);

const trainToggle = document.getElementById('trainToggle');
trainToggle.addEventListener('change', (e) => {
    trainingEnabled = e.target.checked;
    document.getElementById('trainBadge').innerText = trainingEnabled ? 'ON' : 'OFF';
    document.getElementById('trainBadge').style.background = trainingEnabled
        ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255,255,255,0.08)';
});

const lrSlider = document.getElementById('lrSlider');
lrSlider.addEventListener('input', (e) => {
    prnn.learningRate = parseFloat(e.target.value);
    document.getElementById('lrVal').innerText = prnn.learningRate.toFixed(3);
});

const noiseSlider = document.getElementById('noiseSlider');
noiseSlider.addEventListener('input', (e) => {
    noiseLevel = parseFloat(e.target.value);
    document.getElementById('noiseVal').innerText = noiseLevel.toFixed(3);
    noiseScheduleActive = false; // manual override
});

const speedSlider = document.getElementById('speedSlider');
speedSlider.addEventListener('input', (e) => {
    car.baseSpeed = parseFloat(e.target.value);
    document.getElementById('speedVal').innerText = car.baseSpeed.toFixed(1);
});

window.addEventListener('keydown', (e) => {
    if (e.key in keys) { keys[e.key] = true; updateKeyUI(); }
});
window.addEventListener('keyup', (e) => {
    if (e.key in keys) { keys[e.key] = false; updateKeyUI(); }
});

// ─── Controls ───
function toggleRun() {
    isRunning = !isRunning;
    const btn = document.getElementById('playPauseBtn');
    btn.innerHTML = isRunning ? '⏸️ Pause Simulation' : '▶️ Start Simulation';
    btn.className = isRunning ? 'danger-btn' : 'primary';
    if (isRunning) requestAnimationFrame(loop);
}

function resetSimulation() {
    car.reset();
    survivalFrames = 0;
    updateStatsUI();
}

function changeTrack(e) {
    track = new Track(e.target.value);
    const starts = { oval: [400, 100, 0], scurve: [100, 410, -Math.PI / 2], obstacles: [400, 80, 0] };
    const s = starts[e.target.value] || starts.oval;
    car.startX = s[0]; car.startY = s[1]; car.startAngle = s[2];
    car.reset();
    survivalFrames = 0;
    updateStatsUI();
}

function changeTrainingMode(e) { trainingMode = e.target.value; }

function toggleManualDrive() {
    manualDrive = !manualDrive;
    const btn = document.getElementById('manualDriveBtn');
    btn.innerHTML = manualDrive ? '🎮 Autopilot Off (Manual)' : '🧠 Autopilot On (Autonomous)';
    btn.style.borderColor = manualDrive ? '#3b82f6' : 'rgba(255,255,255,0.08)';
    btn.style.background = manualDrive ? 'rgba(59,130,246,0.15)' : 'rgba(255,255,255,0.05)';
}

function clearWeights() { prnn.init(); }

function updateKeyUI() {
    document.getElementById('keyL').className = keys.ArrowLeft ? 'key-cap active' : 'key-cap';
    document.getElementById('keyR').className = keys.ArrowRight ? 'key-cap active' : 'key-cap';
}

function updateStatsUI() {
    document.getElementById('crashVal').innerText = crashCount;
    document.getElementById('distVal').innerText = (totalDistance / 100).toFixed(1) + ' m';
    const surv = (survivalFrames / 60).toFixed(1);
    document.getElementById('survVal').innerText = surv + 's';
    document.getElementById('maxSurvVal').innerText = (maxSurvivalTime / 60).toFixed(1) + 's';
}

// ─── Game Loop ───
function loop(timestamp) {
    if (!isRunning) return;
    update();
    draw();
    requestAnimationFrame(loop);
}

function update() {
    totalFrames++;

    // 1. Sensors
    car.castRays(track.walls);

    // 2. Build input phases (distance → phase: 0 = safe, π = wall)
    const clampedPhases = {};
    const clampedNodes = [0, 1, 2, 3];
    clampedPhases[0] = (1.0 - car.sensorDistances[0]) * Math.PI;
    clampedPhases[1] = (1.0 - car.sensorDistances[1]) * Math.PI;
    clampedPhases[2] = (1.0 - car.sensorDistances[2]) * Math.PI;
    clampedPhases[3] = 0.0;

    // 3. Noise annealing
    let effectiveNoise = noiseLevel;
    if (noiseScheduleActive && trainingEnabled) {
        effectiveNoise = initialNoise * Math.max(0.01, 1.0 - totalFrames / 8000);
        document.getElementById('noiseVal').innerText = effectiveNoise.toFixed(3);
    }

    // 4. Autopilot settle → get steering
    prnn.settle(clampedNodes, clampedPhases, 100, 0.05, effectiveNoise);

    const autopilotSteer = Math.sin(prnn.phases[7]);

    // 5. Steering decision
    let steering = manualDrive
        ? (keys.ArrowLeft ? -1.0 : (keys.ArrowRight ? 1.0 : 0.0))
        : autopilotSteer;

    // 6. Speed adaptation: slower near walls, faster in open
    const minDist = Math.min(...car.sensorDistances);
    car.speed = (car.baseSpeed || baseSpeedDefault) * (0.3 + 0.7 * Math.min(1.0, minDist / dangerThreshold));

    // 7. Move car
    const prevX = car.x, prevY = car.y;
    car.update(steering);
    totalDistance += Math.hypot(car.x - prevX, car.y - prevY);
    survivalFrames++;

    // 8. ─── TRAINING ───
    if (trainingEnabled) {
        if (trainingMode === 'hebbian') {
            let reward = 0.6;
            if (minDist < dangerThreshold) reward -= 1.2 * (1.0 - minDist) ** 2;
            if (manualDrive && minDist > dangerThreshold) reward += 0.2;
            document.getElementById('rewardVal').innerText = (reward >= 0 ? '+' : '') + reward.toFixed(2);
            document.getElementById('rewardVal').className = reward >= 0 ? 'stat-value success' : 'stat-value danger';
            prnn.updateRewardHebbian(reward);

        } else if (trainingMode === 'contrastive') {
            // ── Continuous Contrastive Hebbian Learning (every frame) ──
            // Save negative phase (free-running state before clamp)
            const negState = prnn.saveState();

            // Compute the "correct" steering for the current sensor reading
            let correctSteer = 0.0;
            const dLeft = car.sensorDistances[0];
            const dRight = car.sensorDistances[2];

            if (dLeft < dRight - 0.05) {
                correctSteer = 1.0;   // steer right (away from left wall)
            } else if (dRight < dLeft - 0.05) {
                correctSteer = -1.0;  // steer left (away from right wall)
            } else {
                correctSteer = 0.0;   // symmetric → go straight
            }

            const targetPhase = correctSteer * Math.PI / 2;
            const posNodes = [0, 1, 2, 3, 7];
            const posPhases = { ...clampedPhases, 7: targetPhase };

            // Settle under positive-phase clamp (teaching signal)
            prnn.settle(posNodes, posPhases, 150, 0.05, 0.005);

            // Apply CHL using stored negative state and current positive state
            const posState = prnn.saveState();
            prnn.updateContrastive(posState, negState);

            // Restore negative state so the autopilot continues from free dynamics
            prnn.restoreState(negState);

            document.getElementById('rewardVal').innerText = 'CHL ✓';
            document.getElementById('rewardVal').className = 'stat-value success';
        }
    } else {
        document.getElementById('rewardVal').innerText = 'Disabled';
        document.getElementById('rewardVal').className = 'stat-value';
    }

    // 9. Collision check
    if (track.checkCollision(car)) {
        if (trainingEnabled) {
            if (trainingMode === 'hebbian') {
                prnn.updateRewardHebbian(-1.5);
            } else {
                // Contrastive crash correction
                const negState = prnn.saveState();
                const correctSteer = car.sensorDistances[0] < car.sensorDistances[2] ? 1.0 : -1.0;
                const posNodes = [0, 1, 2, 3, 7];
                const posPhases = { ...clampedPhases, 7: correctSteer * Math.PI / 2 };
                prnn.settle(posNodes, posPhases, 150, 0.05, 0.0);
                prnn.updateContrastive(prnn.saveState(), negState);
                prnn.restoreState(negState);
            }
        }
        crashCount++;
        if (survivalFrames > maxSurvivalTime) maxSurvivalTime = survivalFrames;
        car.reset();
        survivalFrames = 0;
        flashScreen();
    }

    updateStatsUI();
}

function flashScreen() {
    gameCanvas.style.boxShadow = '0 0 30px rgba(239, 68, 68, 0.7)';
    setTimeout(() => { gameCanvas.style.boxShadow = 'none'; }, 150);
}

function draw() {
    gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
    track.draw(gameCtx);
    drawStartLine(gameCtx);
    car.draw(gameCtx);
    drawNetwork();
}

function drawStartLine(ctx) {
    ctx.beginPath();
    ctx.moveTo(400, 75);
    ctx.lineTo(400, 115);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 4;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
}

// ─── PRNN Visualizer ───
function drawNetwork() {
    networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);

    // Connection weights
    networkCtx.lineWidth = 1;
    for (let i = 0; i < prnn.N; i++) {
        for (let j = i + 1; j < prnn.N; j++) {
            const w = prnn.K[i][j];
            if (w === 0) continue;
            const p1 = nodePositions[i], p2 = nodePositions[j];
            networkCtx.beginPath();
            networkCtx.moveTo(p1.x, p1.y);
            networkCtx.lineTo(p2.x, p2.y);
            const intensity = Math.min(1.0, Math.abs(w) / prnn.maxWeight);
            networkCtx.strokeStyle = w > 0
                ? `rgba(16, 185, 129, ${0.1 + intensity * 0.75})`
                : `rgba(239, 68, 68, ${0.1 + intensity * 0.75})`;
            networkCtx.lineWidth = 0.5 + intensity * 4.5;
            networkCtx.stroke();
        }
    }

    // Nodes
    networkCtx.lineWidth = 2;
    for (let i = 0; i < prnn.N; i++) {
        const node = nodePositions[i];
        const phase = prnn.phases[i];
        const amp = prnn.r[i];
        const isClamped = i <= 3;

        // Outer glow
        networkCtx.beginPath();
        networkCtx.arc(node.x, node.y, 25, 0, 2 * Math.PI);
        networkCtx.fillStyle = '#060913';
        networkCtx.shadowColor = node.color;
        networkCtx.shadowBlur = 8;
        networkCtx.fill();
        networkCtx.shadowBlur = 0;

        // Border — amplitude modulates brightness
        networkCtx.strokeStyle = isClamped ? '#f3f4f6' : node.color;
        networkCtx.lineWidth = isClamped ? 3 : 2;
        networkCtx.globalAlpha = 0.3 + 0.7 * Math.min(1.0, amp);
        networkCtx.stroke();
        networkCtx.globalAlpha = 1.0;

        // Phase arrow
        networkCtx.beginPath();
        networkCtx.moveTo(node.x, node.y);
        const len = 18;
        const ex = node.x + Math.cos(phase) * len;
        const ey = node.y + Math.sin(phase) * len;
        networkCtx.lineTo(ex, ey);
        networkCtx.strokeStyle = node.color;
        networkCtx.lineWidth = 3;
        networkCtx.stroke();

        // Arrow tip (glow if amplitude high)
        networkCtx.beginPath();
        networkCtx.arc(ex, ey, 4, 0, 2 * Math.PI);
        networkCtx.fillStyle = amp > 0.8 ? '#ffffff' : '#9ca3af';
        networkCtx.fill();

        // Label
        networkCtx.font = 'bold 10px Outfit';
        networkCtx.fillStyle = '#f3f4f6';
        networkCtx.textAlign = 'center';
        networkCtx.fillText(node.name, node.x, node.y + 40);

        // Phase + amplitude display
        networkCtx.font = '9px monospace';
        networkCtx.fillStyle = '#9ca3af';
        networkCtx.fillText((phase * 180 / Math.PI).toFixed(0) + '°  |z|=' + amp.toFixed(2), node.x, node.y - 32);
    }

    // Legend
    networkCtx.font = '11px Outfit';
    networkCtx.fillStyle = '#9ca3af';
    networkCtx.textAlign = 'left';
    networkCtx.fillText('Constructive (+)', 30, 450);
    networkCtx.fillText('Destructive (-)', 240, 450);
}

// Initial draw
draw();
