document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const promptInput = document.getElementById('prompt-input');
    const chatContainer = document.getElementById('chat-container');
    const modeSelect = document.getElementById('mode-select');
    const sendBtn = document.getElementById('send-btn');
    const typingIndicator = document.getElementById('typing-indicator');
    const betaSlider = document.getElementById('beta-slider');
    const betaValue = document.getElementById('beta-value');
    const kBSlider = document.getElementById('kB-slider');
    const kBValue = document.getElementById('kB-value');
    const meterSelect = document.getElementById('meter-select');
    const rhymeInput = document.getElementById('rhyme-input');
    const poeticOptions = document.getElementById('poetic-options');
    const cascadeToggle = document.getElementById('cascade-toggle');
    const cascadeStrengthSlider = document.getElementById('cascade-strength-slider');
    const cascadeStrengthValue = document.getElementById('cascade-strength-value');
    const dialogueToggle = document.getElementById('dialogue-toggle');
    const relaxationToggle = document.getElementById('relaxation-toggle');
    const physTraceContainer = document.getElementById('phys-trace-container');
    const traceSteps = document.getElementById('trace-steps');
    const traceSummary = document.getElementById('trace-summary');

    // New split workspace element handles
    const fieldResultsContainer = document.getElementById('field-results-container');
    const sioResultsContainer = document.getElementById('sio-results-container');
    const calibrationSidebar = document.getElementById('calibration-sidebar');
    const letterGrid = document.getElementById('letter-grid');
    const selectedLetterHeader = document.getElementById('selected-letter-header');
    const slidersContainer = document.getElementById('sliders-container');
    const saveCalibrationBtn = document.getElementById('save-calibration-btn');
    const newBenchmarkWordInput = document.getElementById('new-benchmark-word');
    const addBenchmarkWordBtn = document.getElementById('add-benchmark-word-btn');
    const benchmarkWordsList = document.getElementById('benchmark-words-list');
    const vocabTypeSelect = document.getElementById('vocab-type-select');
    const hebbianBoostToggle = document.getElementById('hebbian-boost-toggle');
    const attractedWordsList = document.getElementById('attracted-words-list');
    const repelledWordsList = document.getElementById('repelled-words-list');

    // Sidebar & calibration state
    let currentLetter = null;
    let letterVectors = {};
    let dimNames = [];
    let isDirty = false;
    let benchmarkWords = [];

    const ARABIC_DIM_LABELS = {
        "concentration": "تركيز",
        "internal_external": "داخلي/خارجي",
        "stability_motion": "ثبات/حركة",
        "density": "كثافة",
        "temperature": "حرارة",
        "time_accumulation": "تراكم زمني",
        "time_peak": "ذروة زمنية",
        "time_discharge": "تفريغ زمني",
        "motion_linear": "حركة خطية",
        "motion_rotary": "حركة دورانية",
        "motion_pulse": "حركة نبضية",
        "motion_stretch": "حركة تمددية",
        "motion_slip": "حركة انزلاقية",
        "motion_air": "حركة هوائية",
        "axis_v": "محور عمودي",
        "mass": "كتلة",
        "hardness_solid": "صلابة",
        "penetration": "تغلغل/اختراق",
        "charge": "شحنة",
        "reference_self": "مرجعية ذاتية",
        "space_extensionality": "امتداد مكاني",
        "time_causality": "سببية زمنية"
    };

    betaSlider.addEventListener('input', () => {
        betaValue.textContent = parseFloat(betaSlider.value).toFixed(1);
    });
    kBSlider.addEventListener('input', () => {
        kBValue.textContent = parseFloat(kBSlider.value).toFixed(1);
    });

    cascadeToggle.addEventListener('change', () => {
        cascadeStrengthSlider.disabled = !cascadeToggle.checked;
    });
    cascadeStrengthSlider.addEventListener('input', () => {
        cascadeStrengthValue.textContent = parseFloat(cascadeStrengthSlider.value).toFixed(1);
    });

    // Handle Generation mode dynamic layout updates
    modeSelect.addEventListener('change', () => {
        poeticOptions.style.display = modeSelect.value === 'poetic' ? 'flex' : 'none';
        const isChat = !['attract', 'synthesize', 'physics'].includes(modeSelect.value);
        chatContainer.style.display = isChat ? 'flex' : 'none';
        fieldResultsContainer.style.display = modeSelect.value === 'attract' ? 'flex' : 'none';
        sioResultsContainer.style.display = modeSelect.value === 'synthesize' ? 'flex' : 'none';
        physTraceContainer.style.display = modeSelect.value === 'physics' ? 'block' : 'none';
        calibrationSidebar.style.display = modeSelect.value === 'attract' ? 'flex' : 'none';

        if (modeSelect.value === 'attract') {
            promptInput.placeholder = "أدخل كلمة أو حرفاً لفحص حقل الجذب والتنافر الفيزيائي...";
            loadLetters();
            loadBenchmarkVocab();
        } else if (modeSelect.value === 'synthesize') {
            promptInput.placeholder = "اكتب هدف المشروع... مثال: ابنِ لي API لإدارة المهام مع واجهة ويب";
            document.getElementById('sio-status').textContent = 'جاهز';
        } else if (modeSelect.value === 'physics') {
            promptInput.placeholder = "أدخل نصاً لتتبع فيزياء التوليد...";
        } else if (['root', 'root_list', 'root_poetic'].includes(modeSelect.value)) {
            promptInput.placeholder = "أدخل كلمة واحدة لاستكشاف حقل جذرها الصرفي...";
        } else {
            promptInput.placeholder = "اكتب رسالتك هنا...";
        }
    });

    // Letter Calibration APIs
    async function loadLetters() {
        try {
            const response = await fetch('/api/letters');
            const data = await response.json();
            dimNames = data.dim_names;
            letterVectors = {};
            for (const [ch, info] of Object.entries(data.letters)) {
                letterVectors[ch] = info.v;
            }
            renderLetterGrid();
        } catch (error) {
            console.error("Failed to load letters:", error);
        }
    }

    const ARABIC_LETTERS_ORDER = "أبتثجحخدذرزسشصضطظعغفقكلمنهوي".split("");
    function renderLetterGrid() {
        letterGrid.innerHTML = '';
        ARABIC_LETTERS_ORDER.forEach(ch => {
            if (letterVectors[ch]) {
                const btn = document.createElement('button');
                btn.className = 'letter-btn';
                btn.textContent = ch;
                btn.type = 'button';
                if (currentLetter === ch) {
                    btn.classList.add('active');
                }
                btn.addEventListener('click', () => selectLetter(ch));
                letterGrid.appendChild(btn);
            }
        });
    }

    function selectLetter(letter) {
        currentLetter = letter;
        document.querySelectorAll('.letter-btn').forEach(btn => {
            btn.classList.toggle('active', btn.textContent === letter);
        });
        selectedLetterHeader.textContent = `معايرة الحرف: ${letter}`;
        renderSliders(letter);
        fetchRichLetterInfo(letter);
        saveCalibrationBtn.disabled = true;
        isDirty = false;
    }

    async function fetchRichLetterInfo(letter) {
        const card = document.getElementById('rich-letter-card');
        try {
            const response = await fetch(`/api/letters/rich/${encodeURIComponent(letter)}`);
            if (!response.ok) {
                card.style.display = 'none';
                return;
            }
            const data = await response.json();
            renderRichLetterCard(data);
            card.style.display = 'block';
        } catch (error) {
            console.error("Failed to fetch rich letter info:", error);
            card.style.display = 'none';
        }
    }

    function renderRichLetterCard(data) {
        document.getElementById('meaning-core').textContent = data.core_meaning || '?';
        
        const branchesDiv = document.getElementById('meaning-branches');
        branchesDiv.innerHTML = '';
        if (data.branches && data.branches.length > 0) {
            data.branches.forEach(b => {
                const tag = document.createElement('span');
                tag.className = 'branch-tag';
                tag.textContent = b;
                branchesDiv.appendChild(tag);
            });
        }

        document.getElementById('meaning-opposite').innerHTML = 
            `<span class="pair-label">ضد</span><span>${data.opposite || '?'}</span>`;
        document.getElementById('meaning-standard').innerHTML = 
            `<span class="pair-label">ميزان</span><span>${data.standard_of || '?'}</span>`;

        document.getElementById('phys-omega').textContent = data.omega_0 || '?';
        document.getElementById('phys-operator').textContent = data.operator || '?';

        const interpDiv = document.getElementById('vector-interp-section');
        interpDiv.innerHTML = '';
        if (data.vector_interpretation) {
            const interp = data.vector_interpretation;
            if (interp.dominant && interp.dominant.length > 0) {
                const domDiv = document.createElement('div');
                domDiv.className = 'interp-block dominant';
                domDiv.innerHTML = '<span class="interp-label">ينشط</span>';
                interp.dominant.forEach(([name, desc]) => {
                    const item = document.createElement('span');
                    item.className = 'interp-item';
                    item.textContent = `${name}`;
                    item.title = desc;
                    domDiv.appendChild(item);
                });
                interpDiv.appendChild(domDiv);
            }
            if (interp.opposite && interp.opposite.length > 0) {
                const oppDiv = document.createElement('div');
                oppDiv.className = 'interp-block opposite';
                oppDiv.innerHTML = '<span class="interp-label">ينخفض</span>';
                interp.opposite.forEach(([name, desc]) => {
                    const item = document.createElement('span');
                    item.className = 'interp-item';
                    item.textContent = `${name}`;
                    item.title = desc;
                    oppDiv.appendChild(item);
                });
                interpDiv.appendChild(oppDiv);
            }
        }
    }

    function renderSliders(letter) {
        slidersContainer.innerHTML = '';
        const vector = letterVectors[letter];
        dimNames.forEach((name, idx) => {
            const val = vector[idx];
            const group = document.createElement('div');
            group.className = 'dim-slider-group';
            
            const labelText = ARABIC_DIM_LABELS[name] || name;
            
            group.innerHTML = `
                <div class="dim-slider-label">
                    <span>${labelText}</span>
                    <span class="dim-val" id="val-${name}">${val.toFixed(2)}</span>
                </div>
                <input type="range" class="dim-slider" id="slider-${name}" min="-1" max="1" step="0.05" value="${val}">
            `;
            
            const slider = group.querySelector('input');
            slider.addEventListener('input', (e) => {
                const newVal = parseFloat(e.target.value);
                document.getElementById(`val-${name}`).textContent = newVal.toFixed(2);
                
                // Update internal values
                letterVectors[letter][idx] = newVal;
                isDirty = true;
                saveCalibrationBtn.disabled = false;
                
                updateSliderTrack(slider, newVal);
                triggerFieldUpdate();
            });
            
            slidersContainer.appendChild(group);
            updateSliderTrack(slider, val);
        });
    }

    function updateSliderTrack(slider, value) {
        const pct = (value + 1) / 2 * 100;
        slider.style.background = `linear-gradient(to right, #ef4444 0%, #ef4444 ${pct}%, #10b981 ${pct}%, #10b981 100%)`;
    }

    let updateTimeout = null;
    function triggerFieldUpdate() {
        if (updateTimeout) return;
        updateTimeout = setTimeout(() => {
            updateTimeout = null;
            sendLetterUpdate(currentLetter, letterVectors[currentLetter]);
        }, 45);
    }

    async function sendLetterUpdate(letter, vector) {
        try {
            await fetch('/api/letters/update', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({letter, vector})
            });
            recalculateField();
        } catch (error) {
            console.error("Failed to update letter:", error);
        }
    }

    saveCalibrationBtn.addEventListener('click', () => {
        if (!currentLetter || !isDirty) return;
        saveCalibrationBtn.disabled = true;
        isDirty = false;
        saveCalibrationBtn.innerHTML = '<i class="fa-solid fa-check"></i> تم الحفظ!';
        setTimeout(() => {
            saveCalibrationBtn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> حفظ المعايرة';
        }, 1500);
    });

    // Benchmark Vocabulary Manager APIs
    async function loadBenchmarkVocab() {
        try {
            const response = await fetch('/api/benchmark');
            const data = await response.json();
            benchmarkWords = data.words;
            renderBenchmarkVocab();
        } catch (error) {
            console.error("Failed to load benchmark vocabulary:", error);
        }
    }

    function renderBenchmarkVocab() {
        benchmarkWordsList.innerHTML = '';
        benchmarkWords.forEach(w => {
            const tag = document.createElement('div');
            tag.className = 'benchmark-tag';
            tag.innerHTML = `
                <span>${w}</span>
                <span class="delete-tag"><i class="fa-solid fa-xmark"></i></span>
            `;
            tag.querySelector('.delete-tag').addEventListener('click', () => deleteBenchmarkWord(w));
            benchmarkWordsList.appendChild(tag);
        });
    }

    async function deleteBenchmarkWord(word) {
        benchmarkWords = benchmarkWords.filter(w => w !== word);
        renderBenchmarkVocab();
        await saveBenchmarkVocab();
        recalculateField();
    }

    async function addBenchmarkWord() {
        const word = newBenchmarkWordInput.value.trim();
        if (!word) return;
        if (!benchmarkWords.includes(word)) {
            benchmarkWords.push(word);
            renderBenchmarkVocab();
            await saveBenchmarkVocab();
            recalculateField();
        }
        newBenchmarkWordInput.value = '';
    }

    async function saveBenchmarkVocab() {
        try {
            await fetch('/api/benchmark/update', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({words: benchmarkWords})
            });
        } catch (error) {
            console.error("Failed to save benchmark vocabulary:", error);
        }
    }

    addBenchmarkWordBtn.addEventListener('click', addBenchmarkWord);
    newBenchmarkWordInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            addBenchmarkWord();
        }
    });

    // Field Calculation Updates
    async function recalculateField() {
        const target = promptInput.dataset.lastTarget || "";
        if (!target) return;
        
        const vocabType = vocabTypeSelect ? vocabTypeSelect.value : 'all';
        const useHebbian = hebbianBoostToggle ? hebbianBoostToggle.checked : false;
        
        try {
            const response = await fetch(`/api/field?target=${encodeURIComponent(target)}&vocab_type=${vocabType}&use_hebbian=${useHebbian}`);
            const data = await response.json();
            renderFieldResults(data);
        } catch (error) {
            console.error("Failed to calculate attraction field:", error);
        }
    }

    if (vocabTypeSelect) vocabTypeSelect.addEventListener('change', recalculateField);
    if (hebbianBoostToggle) hebbianBoostToggle.addEventListener('change', recalculateField);

    function renderFieldResults(data) {
        attractedWordsList.innerHTML = '';
        repelledWordsList.innerHTML = '';
        
        if (data.attracted && data.attracted.length > 0) {
            data.attracted.forEach(([word, score]) => {
                attractedWordsList.appendChild(createWordCard(word, score));
            });
        } else {
            attractedWordsList.innerHTML = '<div class="word-card"><div class="word-card-header"><span class="word-name">لا توجد نتائج</span></div></div>';
        }
        
        if (data.repelled && data.repelled.length > 0) {
            data.repelled.forEach(([word, score]) => {
                repelledWordsList.appendChild(createWordCard(word, score));
            });
        } else {
            repelledWordsList.innerHTML = '<div class="word-card"><div class="word-card-header"><span class="word-name">لا توجد نتائج</span></div></div>';
        }
    }

    function createWordCard(word, score) {
        const card = document.createElement('div');
        card.className = 'word-card';
        const pct = Math.min(Math.max(Math.abs(score) * 100, 0), 100);
        
        card.innerHTML = `
            <div class="word-card-header">
                <span class="word-name">${word}</span>
                <span class="word-score">${score >= 0 ? '+' : ''}${score.toFixed(3)}</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: ${pct}%"></div>
            </div>
        `;
        return card;
    }

    // Markdown Parser
    function parseMarkdown(text) {
        if (!text) return "";
        let html = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');

        let lines = html.split('\n');
        let inList = false;
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i].trim();
            if (line.startsWith('- ')) {
                if (!inList) {
                    lines[i] = '<ul><li>' + line.substring(2) + '</li>';
                    inList = true;
                } else {
                    lines[i] = '<li>' + line.substring(2) + '</li>';
                }
            } else {
                if (inList) {
                    lines[i] = '</ul>' + (line ? '<p>' + line + '</p>' : '');
                    inList = false;
                } else if (line) {
                    if (!line.startsWith('<h') && !line.startsWith('<ul') && !line.startsWith('<li')) {
                        lines[i] = '<p>' + line + '</p>';
                    }
                }
            }
        }
        if (inList) {
            lines.push('</ul>');
        }
        return lines.join('\n');
    }

    function looksLikeCodeBlock(content) {
        if (!content) return false;
        const text = String(content).trim();
        if (!text) return false;
        const lines = text.split('\n');
        return lines.some(line =>
            /^\s*(def|class|for|while|if|elif|else|try|except|finally|with|import|from|return|print|function|struct|module)\b/.test(line) ||
            /^\s*[A-Za-z_]\w*\s*=/.test(line)
        );
    }

    function renderCodeBlock(contentDiv, content) {
        contentDiv.classList.add('code-message-content');
        const pre = document.createElement('pre');
        pre.className = 'message-code-block';
        pre.dir = 'ltr';
        pre.textContent = String(content).replace(/\r\n/g, '\n');
        contentDiv.appendChild(pre);
    }

    function addMessage(content, type) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}-message`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (type === 'bot' || type === 'system') {
            if (looksLikeCodeBlock(content)) {
                renderCodeBlock(contentDiv, content);
            } else {
                contentDiv.innerHTML = parseMarkdown(content);
            }
        } else {
            contentDiv.textContent = content;
        }

        msgDiv.appendChild(contentDiv);
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return msgDiv;
    }

    function addPhysicsReport(msgDiv, report, timeTaken) {
        if (!report) return;

        const reportDiv = document.createElement('div');
        reportDiv.className = 'physics-report';

        const addStat = (icon, label) => {
            const s = document.createElement('div');
            s.className = 'physics-stat';
            s.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${label}</span>`;
            reportDiv.appendChild(s);
        };

        addStat('fa-stopwatch', `${timeTaken.toFixed(3)}s`);
        addStat('fa-wave-square', `S=${report.entropy}`);
        addStat('fa-fire', `β=${report.beta}`);
        addStat('fa-temperature-high', `k_B=${report.k_B}`);

        if (report.dccf_coupling !== undefined) {
            addStat('fa-link', `DCCF=${report.dccf_coupling}`);
        }
        if (report.ppm_field !== undefined) {
            addStat('fa-magnet', `PPM=${report.ppm_field}`);
        }
        if (report.amfs_centrality !== undefined) {
            addStat('fa-weight-scale', `AMFS=${report.amfs_centrality}`);
        }
        if (report.cascade_enabled !== undefined) {
            addStat('fa-circle-down', `CASCADE=${report.cascade_enabled ? 'ON' : 'OFF'}(${report.cascade_strength})`);
        }
        if (report.dialogue_mode) {
            addStat('fa-comments', `DIALOGUE=${report.dialogue_intent}(${report.dialogue_confidence})`);
        }
        if (report.mass_mean) {
            addStat('fa-weight-hanging', `M=${report.mass_mean.toFixed(2)}`);
        }
        if (report.mode) {
            addStat('fa-microchip', report.mode);
        }
        if (report.phase_coherence !== undefined) {
            addStat('fa-rotate', `Φ=${report.phase_coherence.toFixed(3)}`);
        }
        if (report.temperature !== undefined && report.temperature > 0) {
            addStat('fa-temperature-half', `T=${report.temperature.toFixed(2)}`);
        }
        if (report.S_crit !== undefined) {
            addStat('fa-gauge-high', `S_crit=${report.S_crit}`);
        }
        if (report.heterodyne_active !== undefined && report.heterodyne_active > 0) {
            addStat('fa-tower-broadcast', `HET=${report.heterodyne_active.toFixed(3)}`);
        }
        if (report.oscillator_active !== undefined && report.oscillator_active > 0) {
            addStat('fa-arrows-spin', `OSC=${report.oscillator_active.toFixed(3)}`);
        }
        if (report.gravity_vector !== undefined && report.gravity_vector > 0) {
            addStat('fa-arrows-down-to-line', `GV=${report.gravity_vector.toFixed(3)}`);
        }
        if (report.carrier_active !== undefined && report.carrier_active > 0) {
            addStat('fa-broadcast-tower', `CR=${report.carrier_active.toFixed(3)}`);
        }
        if (report.beamform_active !== undefined && report.beamform_active > 0) {
            addStat('fa-crosshairs', `BF=${report.beamform_active.toFixed(3)}`);
        }
        if (report.refractory_active !== undefined && report.refractory_active > 0) {
            addStat('fa-ban', `RF=${report.refractory_active.toFixed(3)}`);
        }
        if (report.macro_wave_active !== undefined && report.macro_wave_active > 0) {
            addStat('fa-cubes', `MW=${report.macro_wave_active.toFixed(3)}`);
        }

        if (report.beamformer && report.beamformer.weights) {
            const radarDiv = document.createElement('div');
            radarDiv.className = 'radar-container';
            radarDiv.innerHTML = `<div class="radar-title"><i class="fa-solid fa-crosshairs"></i> الرادار الطوري (الهدف: <strong>${report.beamformer.candidate}</strong>)</div>`;
            
            const barsContainer = document.createElement('div');
            barsContainer.className = 'radar-bars';
            
            const ctxWords = report.beamformer.context;
            const weights = report.beamformer.weights;
            
            for (let i = 0; i < ctxWords.length; i++) {
                const w = ctxWords[i];
                const weight = weights[i];
                const heightPercent = Math.min(100, Math.max(5, weight * 100 * 2.0)); 
                
                const barItem = document.createElement('div');
                barItem.className = 'radar-item';
                barItem.innerHTML = `
                    <div class="radar-bar-wrapper">
                        <div class="radar-bar" style="height: ${heightPercent}%"></div>
                    </div>
                    <div class="radar-word">${w}</div>
                    <div class="radar-val">${(weight).toFixed(2)}</div>
                `;
                barsContainer.appendChild(barItem);
            }
            radarDiv.appendChild(barsContainer);
            reportDiv.appendChild(radarDiv);
        }

        addV8Pillars(report, reportDiv);

        msgDiv.insertAdjacentElement('afterend', reportDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // SIO Results Renderer
    function renderSioResults(data) {
        document.getElementById('sio-status').textContent = 
            `اكتمل في ${data.time_elapsed}s | ${data.phases_completed.length}/${data.total_phases} مراحل`;
        
        const planDiv = document.getElementById('sio-plan');
        planDiv.innerHTML = `<strong>الهدف:</strong> ${data.goal}<br><small>${data.summary}</small>`;
        
        const phasesDiv = document.getElementById('sio-phases');
        phasesDiv.innerHTML = '<h3>المراحل</h3>';
        data.phase_details.forEach(p => {
            const icon = p.coherence > 0.5 ? 'fa-circle-check' : 'fa-circle-exclamation';
            const color = p.coherence > 0.5 ? '#10b981' : '#f59e0b';
            phasesDiv.innerHTML += `
                <div class="sio-phase-card">
                    <i class="fa-solid ${icon}" style="color:${color}"></i>
                    <span class="phase-name">${p.name}</span>
                    <span class="phase-info">${p.iterations} محاولات | تماسك: ${(p.coherence*100).toFixed(0)}%</span>
                    <span class="phase-diag">${p.diagnosis}</span>
                </div>`;
        });
        
        if (data.phases_failed && data.phases_failed.length > 0) {
            phasesDiv.innerHTML += `<div class="sio-phase-card failed">فشلت: ${data.phases_failed.join(', ')}</div>`;
        }
        
        const delivDiv = document.getElementById('sio-deliverable');
        delivDiv.innerHTML = '<h3>المنتج النهائي</h3>';
        delivDiv.innerHTML += parseMarkdown(data.deliverable);
        
        const progress = data.phases_completed.length / Math.max(data.total_phases, 1) * 100;
        document.getElementById('sio-progress-bar').style.width = progress + '%';
        
        document.getElementById('sio-status').textContent += 
            ` | مراقب: ${data.monitor.corrections_applied} تصحيحات`;
    }

    // V8 Pillars display
    function addV8Pillars(report, reportDiv) {
        const pilDiv = document.createElement('div');
        pilDiv.className = 'v8-pillars';
        const pillars = [
            {key: 'dm_coherence', label: 'DM', desc: 'كثافة كمومية'},
            {key: 'pps_cross', label: 'PPS', desc: 'فصل جسيمي'},
            {key: 'cff_score', label: 'CFF', desc: 'تدفق سببي'},
            {key: 'hwm_level', label: 'HWM', desc: 'تعديل هرمي'},
        ];
        pillars.forEach(p => {
            const s = document.createElement('span');
            const val = report[p.key] || report[p.key.toLowerCase()] || 0;
            const color = val > 0.3 ? '#10b981' : val > 0.1 ? '#f59e0b' : '#6b7280';
            s.className = 'v8-pillar';
            s.innerHTML = `<strong style="color:${color}">${p.label}</strong><small>${p.desc}</small>`;
            s.title = `${p.desc}: ${val}`;
            pilDiv.appendChild(s);
        });
        reportDiv.appendChild(pilDiv);
    }

    // Entropy chart drawing
    function drawEntropyChart(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !data.length) return;
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 2;
        ctx.beginPath();
        const pad = 10, w = W - 2 * pad, h = H - 2 * pad;
        for (let i = 0; i < data.length; i++) {
            const x = pad + (i / Math.max(data.length - 1, 1)) * w;
            const y = pad + (1 - data[i]) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.fillStyle = 'rgba(139, 92, 246, 0.15)';
        ctx.lineTo(pad + w, pad + h);
        ctx.lineTo(pad, pad + h);
        ctx.closePath();
        ctx.fill();
    }

    function renderPhysicsTrace(prompt, data) {
        const report = data.physics_report || {};
        traceSummary.innerHTML = `
            <div class="trace-summary-row">
                <span><strong>المدخل:</strong> ${prompt}</span>
                <span><strong>المخرج:</strong> ${data.result || '—'}</span>
                <span><strong>الزمن:</strong> ${data.time_taken ? data.time_taken.toFixed(3) + 's' : '—'}</span>
                <span><strong>الإنتروبيا:</strong> ${report.entropy || '—'}</span>
                <span><strong>β:</strong> ${report.beta || '—'}</span>
                <span><strong>الكلمات:</strong> ${report.word_count || '—'}</span>
            </div>`;

        traceSteps.innerHTML = '';
        if (report.word_masses) {
            const words = Object.entries(report.word_masses);
            words.forEach(([w, mass], i) => {
                const angle = report.phase_angles ? report.phase_angles[i] : '—';
                const align = report.alignments ? report.alignments[i] : '—';
                const step = document.createElement('div');
                step.className = 'trace-step';
                step.innerHTML = `
                    <div class="trace-step-num">${i + 1}</div>
                    <div class="trace-step-word">${w}</div>
                    <div class="trace-step-metrics">
                        <span title="الكتلة">⚖️ ${Number(mass).toFixed(3)}</span>
                        <span title="زاوية الطور">∠ ${angle}</span>
                        <span title="المحاذاة">${align}</span>
                    </div>`;
                traceSteps.appendChild(step);
            });
        }

        if (report.top_k && report.top_k.length) {
            const topDiv = document.createElement('div');
            topDiv.className = 'trace-topk';
            topDiv.innerHTML = '<strong>أعلى المرشحين:</strong> ' + report.top_k.join(' · ');
            traceSteps.appendChild(topDiv);
        }

        // Draw charts
        const entropyData = report.entropy_history || [report.entropy || 0.5];
        drawEntropyChart('entropy-chart', Array.isArray(entropyData) ? entropyData : [entropyData]);
        drawEntropyChart('coherence-chart', [report.phase_coherence || report.dccf_coupling || 0.5, 0.6, 0.55]);

        // V8 pillars
        const chartDiv = document.querySelector('.trace-charts');
        if (chartDiv) addV8Pillars(report, chartDiv);
    }

    // ═══ Chat submit handler ═══
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const prompt = promptInput.value.trim();
        if (!prompt) return;

        const mode = modeSelect.value;
        const beta = parseFloat(betaSlider.value);
        const k_B = parseFloat(kBSlider.value);

        // Custom handling for attraction field mode
        if (mode === 'attract') {
            promptInput.dataset.lastTarget = prompt;
            promptInput.value = '';
            recalculateField();
            if (prompt.length === 1 && letterVectors[prompt]) {
                selectLetter(prompt);
            }
            return;
        }

        // Custom handling for SIO synthesis mode
        if (mode === 'synthesize') {
            promptInput.disabled = true;
            sendBtn.disabled = true;
            
            document.getElementById('sio-status').textContent = 'يحلل الهدف...';
            document.getElementById('sio-plan').innerHTML = '';
            document.getElementById('sio-phases').innerHTML = '';
            document.getElementById('sio-deliverable').innerHTML = '';
            
            try {
                const response = await fetch('/api/synthesize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ goal: prompt, max_phase_retries: 5 }),
                });
                
                if (!response.ok) {
                    throw new Error('SIO synthesis failed');
                }
                
                const data = await response.json();
                renderSioResults(data);
                
            } catch (error) {
                document.getElementById('sio-status').textContent = 'فشل';
                document.getElementById('sio-deliverable').innerHTML = `<div class="sio-error">خطأ: ${error.message}</div>`;
            } finally {
                promptInput.value = '';
                promptInput.disabled = false;
                sendBtn.disabled = false;
                promptInput.focus();
            }
            return;
        }

        const effectiveMode = mode === 'physics'
            ? 'standard'
            : (mode === 'auto' && dialogueToggle.checked ? 'dialogue' : mode);

        // Build the standard request body (used by both physics-trace and regular chat)
        const body = {
            prompt: prompt,
            mode: effectiveMode,
            max_words: 30,
            beta: beta,
            k_B: k_B,
            cascade: cascadeToggle.checked,
            cascade_strength: cascadeToggle.checked ? parseFloat(cascadeStrengthSlider.value) : null,
            dialogue: effectiveMode === 'dialogue',
        };

        if (mode === 'poetic') {
            body.poetic_meter = meterSelect.value;
            if (rhymeInput.value.trim()) {
                body.poetic_rhyme = rhymeInput.value.trim();
            }
        }

        // Physics trace mode — shows trace panel instead of chat bubble
        if (mode === 'physics') {
            promptInput.disabled = true;
            sendBtn.disabled = true;
            typingIndicator.style.display = 'flex';
            try {
                const resp = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body),
                });
                if (!resp.ok) {
                    const errorData = await resp.json();
                    throw new Error(errorData.message || 'Failed to get physics trace');
                }
                const data = await resp.json();
                typingIndicator.style.display = 'none';
                renderPhysicsTrace(prompt, data);
            } catch (error) {
                typingIndicator.style.display = 'none';
                addMessage('خطأ: ' + error.message, 'system');
            } finally {
                promptInput.disabled = false;
                sendBtn.disabled = false;
                promptInput.focus();
                promptInput.value = '';
            }
            return;
        }

        // Standard / Creative / Wave / Quantum / Multiverse / Poetic / Code / Math / Auto / Dialogue modes
        promptInput.value = '';
        promptInput.disabled = true;
        sendBtn.disabled = true;

        addMessage(prompt, 'user');
        typingIndicator.style.display = 'flex';
        chatContainer.scrollTop = chatContainer.scrollHeight;

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minutes

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || errorData.message || 'Failed to generate response');
            }

            const data = await response.json();
            typingIndicator.style.display = 'none';

            const botMsg = addMessage(data.result || '(لم يُنتج النموذج أي نص — قد تكون الكلمة خارج المعجم)', 'bot');
            addPhysicsReport(botMsg, data.physics_report, data.time_taken);

        } catch (error) {
            clearTimeout(timeoutId);
            typingIndicator.style.display = 'none';
            if (error.name === 'AbortError') {
                addMessage('انتهت مهلة الانتظار (10 دقائق). قد يكون الخادم لا يزال يُجمّع الكود للمرة الأولى. أعد المحاولة.', 'system');
            } else {
                addMessage(`خطأ النظام: ${error.message}`, 'system');
            }
        } finally {
            promptInput.disabled = false;
            sendBtn.disabled = false;
            promptInput.focus();
        }
    });
});
