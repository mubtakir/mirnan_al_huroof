const prnn = new PRNN();
const state = new GameState();
let isRunning = false;
let manualOverride = false;
let trainingEnabled = true;
let noiseLevel = 0.01;
let totalFrames = 0;

const gameCanvas = document.getElementById('gameCanvas');
const gc = gameCanvas.getContext('2d');
const netCanvas = document.getElementById('networkCanvas');
const nc = netCanvas.getContext('2d');

const keys = { ArrowLeft: false, ArrowRight: false, ' ': false, a: false, d: false };

// Node positions for 10-neuron network viz
const nodePos = [
    { name: 'S1: Threat', x: 50, y: 80, color: '#ef4444' },
    { name: 'S2: Freq D1', x: 50, y: 170, color: '#ef4444' },
    { name: 'S3: Freq D2', x: 50, y: 260, color: '#ef4444' },
    { name: 'B4: Bias', x: 150, y: 170, color: '#60a5fa' },
    { name: 'I5: Player', x: 150, y: 340, color: '#10b981' },
    { name: 'H6: Hidden', x: 270, y: 80, color: '#8b5cf6' },
    { name: 'H7: Hidden', x: 270, y: 210, color: '#8b5cf6' },
    { name: 'H8: Hidden', x: 270, y: 340, color: '#8b5cf6' },
    { name: 'O9: Freq', x: 400, y: 130, color: '#ec4899' },
    { name: 'O10: Shield', x: 400, y: 280, color: '#ec4899' }
];

// Events
document.getElementById('playPauseBtn').addEventListener('click', () => {
    isRunning = !isRunning;
    const btn = document.getElementById('playPauseBtn');
    btn.innerHTML = isRunning ? '⏸️ Pause' : '▶️ Start Game';
    btn.className = isRunning ? 'danger-btn' : 'primary';
    if (isRunning) requestAnimationFrame(loop);
});

document.getElementById('resetBtn').addEventListener('click', () => {
    state.player.health = 100;
    state.player.shieldEnergy = 100;
    state.player.score = 0;
    state.enemies = [];
    state.particles = [];
    state.waveDifficulty = 1;
    state.spawnInterval = 45;
    state.spawnTimer = 0;
    updateUI();
    draw();
});

document.getElementById('autopilotBtn').addEventListener('click', () => {
    state.player.autopilot = !state.player.autopilot;
    const btn = document.getElementById('autopilotBtn');
    btn.innerHTML = state.player.autopilot ? '🧠 Autopilot ON' : '🎮 Manual Control';
});

document.getElementById('clearWeightsBtn').addEventListener('click', () => prnn.init());

document.getElementById('lrSlider').addEventListener('input', (e) => {
    prnn.learningRate = parseFloat(e.target.value);
    document.getElementById('lrVal').innerText = prnn.learningRate.toFixed(3);
});

document.getElementById('noiseSlider').addEventListener('input', (e) => {
    noiseLevel = parseFloat(e.target.value);
    document.getElementById('noiseVal').innerText = noiseLevel.toFixed(3);
});

window.addEventListener('keydown', (e) => {
    if (e.key === ' ') { e.preventDefault(); keys[' '] = true; }
    if (e.key === 'a' || e.key === 'A') { keys.a = true; e.preventDefault(); }
    if (e.key === 'd' || e.key === 'D') { keys.d = true; e.preventDefault(); }
    if (e.key === 'm' || e.key === 'M') { state.player.autopilot = !state.player.autopilot; }
    if (e.key === 't' || e.key === 'T') {
        prnn.learningMode = prnn.learningMode === 'contrastive' ? 'hebbian' : 'contrastive';
        document.getElementById('learnBadge').innerText = prnn.learningMode === 'contrastive' ? 'CHL' : 'Hebbian';
        document.getElementById('learnBadge').style.background = prnn.learningMode === 'contrastive'
            ? 'rgba(16,185,129,0.2)' : 'rgba(236,72,153,0.2)';
        document.getElementById('learnBadge').style.color = prnn.learningMode === 'contrastive' ? '#10b981' : '#ec4899';
    }
    if (e.key === 'r' || e.key === 'R') { state.reportActive = !state.reportActive; state.creditsActive = false; }
    if (e.key === 'c' || e.key === 'C') { state.creditsActive = !state.creditsActive; state.reportActive = false; }
});
window.addEventListener('keyup', (e) => {
    if (e.key === ' ') { keys[' '] = false; }
    if (e.key === 'a' || e.key === 'A') keys.a = false;
    if (e.key === 'd' || e.key === 'D') keys.d = false;
});

function updateUI() {
    const p = state.player;
    document.getElementById('healthVal').innerText = Math.round(p.health) + '%';
    document.getElementById('healthVal').className = 'stat-value ' + (p.health > 40 ? 'success' : 'danger');
    document.getElementById('shieldVal').innerText = Math.round(p.shieldEnergy) + '%';
    document.getElementById('shieldVal').className = 'stat-value ' + (p.shieldActive ? 'success' : '');
    document.getElementById('freqVal').innerText = p.frequency.toFixed(2) + ' Hz';
    document.getElementById('scoreVal').innerText = 'Score: ' + p.score;
    document.getElementById('diffVal').innerText = 'Diff: ' + state.waveDifficulty;

    if (state.enemies.length > 0) {
        const closest = state.enemies.reduce((a, b) => a.distance < b.distance ? a : b);
        const freqDiff = Math.abs(p.frequency - closest.frequency);
        const label = freqDiff <= 0.08 ? (Math.cos(p.phase - closest.phase) > 0 ? '🔥 CONSTRUCTIVE' : '⚠ DESTRUCTIVE') : '❌ MISMATCH';
        document.getElementById('resonanceVal').innerText = label;
        document.getElementById('resonanceVal').className = 'stat-value ' +
            (freqDiff <= 0.08 ? (Math.cos(p.phase - closest.phase) > 0 ? 'success' : 'danger') : '');
    } else {
        document.getElementById('resonanceVal').innerText = '—';
        document.getElementById('resonanceVal').className = 'stat-value';
    }
}

function loop(ts) {
    if (!isRunning) return;
    update();
    draw();
    updateUI();
    requestAnimationFrame(loop);
}

function update() {
    totalFrames++;

    const dt = 0.05;
    const clampedNodes = [0, 1, 2, 3, 4];
    const clampedPhases = {};

    let closest = null, secondClosest = null;
    if (state.enemies.length > 0) {
        const sorted = [...state.enemies].sort((a, b) => a.distance - b.distance);
        closest = sorted[0];
        if (sorted.length > 1) secondClosest = sorted[1];
    }

    clampedPhases[3] = 0.0;
    clampedPhases[4] = state.player.frequency * Math.PI;

    if (closest) {
        clampedPhases[0] = (1.2 - Math.max(0, Math.min(1.2, closest.distance))) / 1.2 * Math.PI;
        clampedPhases[1] = Math.abs(state.player.frequency - closest.frequency) * Math.PI;
    } else { clampedPhases[0] = 0; clampedPhases[1] = 0; }
    clampedPhases[2] = secondClosest ? Math.abs(state.player.frequency - secondClosest.frequency) * Math.PI : 0;

    const targetFreq = closest ? closest.frequency : state.player.frequency;
    const targetShield = closest && closest.distance < 0.65 ? Math.PI / 2 : 0;

    if (prnn.learningMode === 'contrastive') {
        const neg = prnn.saveState();
        prnn.settle(clampedNodes, clampedPhases, 40, 0.02, noiseLevel);
        const zNeg = prnn.saveState();

        prnn.restoreState(neg);
        const posNodes = [...clampedNodes, 8, 9];
        const posPhases = { ...clampedPhases, 8: targetFreq * Math.PI, 9: targetShield };
        prnn.settle(posNodes, posPhases, 40, 0.02, 0.005);
        const zPos = prnn.saveState();

        prnn.updateContrastive(zPos, zNeg);
        prnn.restoreState(neg);
        prnn.settle(clampedNodes, clampedPhases, 40, 0.02, noiseLevel);
    } else {
        const neg = prnn.saveState();
        prnn.settle(clampedNodes, clampedPhases, 40, 0.02, noiseLevel);

        let reward = 0.6;
        if (closest) {
            const danger = 1 - Math.max(0, Math.min(1.2, closest.distance)) / 1.2;
            reward -= danger;
            if (Math.abs(state.player.frequency - closest.frequency) <= 0.08) reward += 0.5;
            if ((closest.distance < 0.65) === state.player.shieldActive) reward += 0.3;
        }
        prnn.updateRewardHebbian(reward);
        prnn.restoreState(neg);
        prnn.settle(clampedNodes, clampedPhases, 40, 0.02, noiseLevel);
    }

    if (state.player.autopilot) {
        const phaseOut = prnn.phases[8];
        state.player.frequency = Math.max(0.1, Math.min(1.0, Math.abs(phaseOut) / Math.PI));
        const shieldAmp = prnn.r[9];
        if (shieldAmp > 0.8 && state.player.shieldEnergy > 15) state.player.shieldActive = true;
        else if (shieldAmp < 0.4) state.player.shieldActive = false;
    } else {
        if (keys[' ']) {
            if (!state.player.shieldActive && state.player.shieldEnergy > 15) state.player.shieldActive = true;
            else if (state.player.shieldActive) state.player.shieldActive = false;
        }
        if (keys.a) state.player.frequency = Math.max(0.1, state.player.frequency - 0.05);
        if (keys.d) state.player.frequency = Math.min(1.0, state.player.frequency + 0.05);
    }

    updateGame(state, dt);

    if (state.player.health <= 0) {
        state.player.health = 100;
        state.player.shieldEnergy = 100;
        state.enemies = [];
        state.particles = [];
        state.player.score = 0;
        prnn.init();
    }
}

function draw() {
    drawRadar();
    drawNetwork();
}

function drawRadar() {
    gc.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
    const cx = 350, cy = 250;

    // Radar rings
    for (let r = 3; r >= 1; r--) {
        gc.beginPath();
        gc.arc(cx, cy, r * 60, 0, 2 * Math.PI);
        gc.strokeStyle = `rgba(139,92,246,${0.05 + r * 0.05})`;
        gc.lineWidth = 1;
        gc.stroke();
    }

    // Crosshairs
    gc.strokeStyle = 'rgba(139,92,246,0.08)';
    gc.lineWidth = 1;
    gc.beginPath(); gc.moveTo(cx - 180, cy); gc.lineTo(cx + 180, cy); gc.stroke();
    gc.beginPath(); gc.moveTo(cx, cy - 180); gc.lineTo(cx, cy + 180); gc.stroke();

    // Player
    gc.beginPath();
    gc.arc(cx, cy, state.player.shieldActive ? 20 : 14, 0, 2 * Math.PI);
    gc.fillStyle = state.player.shieldActive ? '#22d3ee' : '#3b82f6';
    gc.shadowColor = state.player.shieldActive ? '#22d3ee' : '#3b82f6';
    gc.shadowBlur = state.player.shieldActive ? 20 : 10;
    gc.fill();
    gc.shadowBlur = 0;

    if (state.player.shieldActive) {
        gc.beginPath();
        gc.arc(cx, cy, 28, 0, 2 * Math.PI);
        gc.strokeStyle = 'rgba(34,211,238,0.4)';
        gc.lineWidth = 2;
        gc.setLineDash([4, 4]);
        gc.stroke();
        gc.setLineDash([]);
    }

    // Enemies
    for (const enemy of state.enemies) {
        const ex = cx + enemy.distance * Math.cos(enemy.angle) * 180;
        const ey = cy + enemy.distance * Math.sin(enemy.angle) * 180;

        const color = enemy.color === 'red' ? '#ef4444' : (enemy.color === 'yellow' ? '#fbbf24' :
            (enemy.color === 'magenta' ? '#ec4899' : '#10b981'));

        gc.beginPath();
        gc.arc(ex, ey, enemy.radius, 0, 2 * Math.PI);
        gc.fillStyle = color;
        gc.shadowColor = color;
        gc.shadowBlur = 12;
        gc.fill();
        gc.shadowBlur = 0;

        // Frequency label
        gc.font = 'bold 10px monospace';
        gc.fillStyle = '#fff';
        gc.textAlign = 'center';
        gc.textBaseline = 'middle';
        const label = enemy.isChameleon ? 'C' : (enemy.frequency === 0.2 ? '2' : (enemy.frequency === 0.5 ? '5' : '8'));
        gc.fillText(label, ex, ey);

        // Health bar
        const bw = 20, bh = 3;
        gc.fillStyle = 'rgba(0,0,0,0.5)';
        gc.fillRect(ex - bw / 2, ey - enemy.radius - 6, bw, bh);
        gc.fillStyle = enemy.health > 50 ? '#10b981' : '#ef4444';
        gc.fillRect(ex - bw / 2, ey - enemy.radius - 6, bw * (enemy.health / 100), bh);
    }

    // Particles
    for (const p of state.particles) {
        const alpha = p.life / p.maxLife;
        gc.globalAlpha = alpha;
        gc.fillStyle = p.color;
        gc.beginPath();
        gc.arc(p.x, p.y, 2 + (1 - alpha) * 2, 0, 2 * Math.PI);
        gc.fill();
    }
    gc.globalAlpha = 1;

    // Player freq arc
    gc.beginPath();
    gc.arc(cx, cy, 50, -state.player.phase, -state.player.phase + 0.3);
    gc.strokeStyle = '#10b981';
    gc.lineWidth = 3;
    gc.stroke();
}

function drawNetwork() {
    nc.clearRect(0, 0, netCanvas.width, netCanvas.height);

    // Connections
    for (let i = 0; i < prnn.N; i++) {
        for (let j = i + 1; j < prnn.N; j++) {
            const w = prnn.K[i][j];
            if (w === 0) continue;
            const p1 = nodePos[i], p2 = nodePos[j];
            nc.beginPath();
            nc.moveTo(p1.x, p1.y);
            nc.lineTo(p2.x, p2.y);
            const intensity = Math.min(1, Math.abs(w) / prnn.maxWeight);
            nc.strokeStyle = w > 0
                ? `rgba(16,185,129,${0.1 + intensity * 0.7})`
                : `rgba(239,68,68,${0.1 + intensity * 0.7})`;
            nc.lineWidth = 0.5 + intensity * 4;
            nc.stroke();
        }
    }

    // Nodes
    for (let i = 0; i < prnn.N; i++) {
        const node = nodePos[i];
        const phase = prnn.phases[i];
        const amp = prnn.r[i];

        nc.beginPath();
        nc.arc(node.x, node.y, 22, 0, 2 * Math.PI);
        nc.fillStyle = '#060913';
        nc.shadowColor = node.color;
        nc.shadowBlur = 8;
        nc.fill();
        nc.shadowBlur = 0;

        const isClamped = i <= 4;
        nc.strokeStyle = isClamped ? '#f3f4f6' : node.color;
        nc.lineWidth = isClamped ? 3 : 2;
        nc.globalAlpha = 0.3 + 0.7 * Math.min(1, amp);
        nc.stroke();
        nc.globalAlpha = 1;

        // Phase arrow
        nc.beginPath();
        nc.moveTo(node.x, node.y);
        const len = 16;
        nc.lineTo(node.x + Math.cos(phase) * len, node.y + Math.sin(phase) * len);
        nc.strokeStyle = node.color;
        nc.lineWidth = 3;
        nc.stroke();

        nc.beginPath();
        nc.arc(node.x + Math.cos(phase) * len, node.y + Math.sin(phase) * len, 4, 0, 2 * Math.PI);
        nc.fillStyle = amp > 0.8 ? '#fff' : '#9ca3af';
        nc.fill();

        nc.font = 'bold 9px sans-serif';
        nc.fillStyle = '#f3f4f6';
        nc.textAlign = 'center';
        nc.textBaseline = 'top';
        nc.fillText(node.name, node.x, node.y + 26);

        nc.font = '8px monospace';
        nc.fillStyle = '#9ca3af';
        nc.textBaseline = 'bottom';
        nc.fillText((phase * 180 / Math.PI).toFixed(0) + '° |z|=' + amp.toFixed(2), node.x, node.y - 26);
    }

    // Legend or credits overlay
    if (state.creditsActive) {
        nc.fillStyle = 'rgba(6,9,19,0.85)';
        nc.fillRect(0, 0, netCanvas.width, netCanvas.height);
        nc.font = '11px monospace';
        nc.fillStyle = '#22d3ee';
        nc.textAlign = 'center';
        nc.textBaseline = 'middle';
        const lines = [
            '🧠 PRNN SURVIVAL SYMPHONY',
            '─'.repeat(30),
            'Dynamics: Stuart-Landau',
            'dz/dt = (μ − a − g·⟨|z|²⟩ − |z|² + iω)·z + ΣK·(zⱼ−zᵢ)',
            'μ=1.0  g_inh=0.4  γ=2.0  τ_a=1.5',
            'Learning: CHL + Hebbian',
            '10 oscillators: 3 in → 3 hid → 2 out',
            'Resonance: cos(Δφ) · damage',
        ];
        lines.forEach((l, idx) => {
            nc.fillStyle = idx === 0 ? '#22d3ee' : (idx === 1 ? '#374151' : '#9ca3af');
            nc.fillText(l, netCanvas.width / 2, 40 + idx * 22);
        });
    }

    if (state.reportActive) {
        nc.fillStyle = 'rgba(6,9,19,0.85)';
        nc.fillRect(0, 0, netCanvas.width, netCanvas.height);
        nc.font = '11px monospace';
        nc.textAlign = 'center';
        nc.textBaseline = 'middle';
        const sAmp = (prnn.r[0] + prnn.r[1] + prnn.r[2]) / 3;
        const hAmp = (prnn.r[5] + prnn.r[6] + prnn.r[7]) / 3;
        const oAmp = (prnn.r[8] + prnn.r[9]) / 2;

        const lines = [
            '📊 NETWORK REPORT',
            '─'.repeat(30),
            `Avg |z| Sensors: ${sAmp.toFixed(2)}`,
            `Avg |z| Hidden:  ${hAmp.toFixed(2)}`,
            `Avg |z| Outputs: ${oAmp.toFixed(2)}`,
            `Shield amp:      ${prnn.r[9].toFixed(2)}`,
            `Freq control φ:  ${(prnn.phases[8]*180/Math.PI).toFixed(0)}°`,
        ];
        lines.forEach((l, idx) => {
            nc.fillStyle = idx === 0 ? '#22d3ee' : '#9ca3af';
            nc.fillText(l, netCanvas.width / 2, 40 + idx * 22);
        });
    }
}

// Initial draw
draw();
updateUI();
