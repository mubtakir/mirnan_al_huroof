document.addEventListener('DOMContentLoaded', () => {
    // عناصر المحلل الرئيسية
    const analysisForm = document.getElementById('analysis-form');
    const wordInput = document.getElementById('word-input');
    const spaceSelect = document.getElementById('space-select');
    const vocabSelect = document.getElementById('vocab-select');
    const hebbianToggle = document.getElementById('hebbian-toggle');
    const kSlider = document.getElementById('k-slider');
    const kValue = document.getElementById('k-value');
    
    const welcomeBox = document.getElementById('welcome-box');
    const resultsGrid = document.getElementById('results-grid');
    const defineResults = document.getElementById('define-results');
    const defineSummary = document.getElementById('define-summary');
    const defineLettersGrid = document.getElementById('define-letters-grid');
    const attractedList = document.getElementById('attracted-list');
    const repelledList = document.getElementById('repelled-list');
    
    const loadingIndicator = document.getElementById('loading-indicator');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeBtnText = document.getElementById('analyze-btn-text');
    const statusText = document.getElementById('status-text');
    const timeTakenSpan = document.getElementById('time-taken');

    // عناصر الشريط الجانبي للمعايرة
    const lettersSelectorGrid = document.getElementById('letters-selector-grid');
    const currentLetterTitle = document.getElementById('current-letter-title');
    const slidersContainerGrid = document.getElementById('sliders-container-grid');
    const applyCalibrationBtn = document.getElementById('apply-calibration-btn');
    const saveCalibrationBtn = document.getElementById('save-calibration-btn');

    // عناصر مدير معجم المعايرة المصغر
    const newBenchmarkInput = document.getElementById('new-benchmark-word');
    const addBenchmarkBtn = document.getElementById('add-benchmark-btn');
    const benchmarkWordsListContainer = document.getElementById('benchmark-words-list-container');
    const saveBenchmarkBtn = document.getElementById('save-benchmark-btn');

    // عناصر مبدل الوضع وتفسير الحروف
    const modeFieldBtn = document.getElementById('mode-field-btn');
    const modeDefineBtn = document.getElementById('mode-define-btn');
    const letterInterpPanel = document.getElementById('letter-interpretation-panel');
    const letterRichMeanings = document.getElementById('letter-rich-meanings');
    const letterVectorInterp = document.getElementById('letter-vector-interp');

    // البيانات المخزنة محلياً
    let lettersData = {};
    let dimNames = [];
    let selectedLetter = "";
    let lastAnalyzedWord = "";
    let benchmarkWords = [];
    let updateTimeout = null;
    let currentMode = "field"; // "field" or "define"

    // تعريب أسماء الأبعاد الـ 22 لبرنامج الرقيم
    const dimTranslations = {
        "concentration": "التركيز / التكثيف",
        "internal_external": "الداخلي / الخارجي",
        "stability_motion": "الاستقرار / الحركة",
        "density": "الكثافة",
        "temperature": "الحرارة والشدة",
        "time_accumulation": "التراكم الزمني",
        "time_peak": "الذروة الزمنية",
        "time_discharge": "التفريغ الزمني (العطاء)",
        "motion_linear": "الحركة الخطية",
        "motion_rotary": "الحركة الدائرية",
        "motion_pulse": "الحركة النبضية",
        "motion_stretch": "الحركة الامتدادية",
        "motion_slip": "الحركة الانزلاقية",
        "motion_air": "الحركة الهوائية / النفخ",
        "axis_v": "المحور الرأسي / الطولي",
        "mass": "الكتلة",
        "hardness_solid": "الصلابة / الليونة",
        "penetration": "النفاذية / الاختراق",
        "charge": "الشحنة والجهد",
        "reference_self": "المرجعية الذاتية",
        "space_extensionality": "الامتداد الفضائي",
        "time_causality": "السببية الزمنية"
    };

    // البدء الفوري بتحميل البيانات
    init();

    async function init() {
        await loadLetters();
        await loadBenchmarkWords();
        setupModeSwitcher();
    }

    // مبدل الوضع: تحليل المجال / تفكيك دلالي
    function setupModeSwitcher() {
        modeFieldBtn.addEventListener('click', () => switchMode('field'));
        modeDefineBtn.addEventListener('click', () => switchMode('define'));
    }

    function switchMode(mode) {
        currentMode = mode;
        modeFieldBtn.classList.toggle('active', mode === 'field');
        modeDefineBtn.classList.toggle('active', mode === 'define');

        if (mode === 'field') {
            analyzeBtnText.textContent = 'قياس المجال';
            wordInput.placeholder = 'أدخل كلمة واحدة باللغة العربية (مثال: علم)...';
        } else {
            analyzeBtnText.textContent = 'تفكيك دلالي';
            wordInput.placeholder = 'أدخل كلمة لتفكيكها إلى دلالات حروفها (مثال: عدل)...';
        }

        if (lastAnalyzedWord) {
            triggerAnalysis(lastAnalyzedWord);
        }
    }

    // تحديث رقم عرض النتائج عند تحريك الشريط
    kSlider.addEventListener('input', () => {
        kValue.textContent = kSlider.value;
    });

    // إعادة التحليل فورا عند تغيير الفضاء أو حالة التعزيز أو حجم المعجم
    spaceSelect.addEventListener('change', () => {
        if (lastAnalyzedWord && currentMode === 'field') triggerAnalysis(lastAnalyzedWord);
    });

    vocabSelect.addEventListener('change', () => {
        if (lastAnalyzedWord && currentMode === 'field') triggerAnalysis(lastAnalyzedWord);
    });

    hebbianToggle.addEventListener('change', () => {
        if (lastAnalyzedWord && currentMode === 'field') triggerAnalysis(lastAnalyzedWord);
    });

    // إضافة الكليك للأمثلة السريعة
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            wordInput.value = btn.textContent;
            triggerAnalysis(btn.textContent);
        });
    });

    // جلب أبعاد الحروف من السيرفر
    async function loadLetters() {
        try {
            const res = await fetch('/api/letters');
            if (!res.ok) throw new Error("فشل جلب الحروف");
            const data = await res.json();
            
            lettersData = data.letters || {};
            dimNames = data.dim_names || [];
            
            renderLetterSelector();
            
            // اختيار الحرف الأول تلقائياً
            const firstLetter = Object.keys(lettersData)[0];
            if (firstLetter) {
                selectLetter(firstLetter);
            }
        } catch (err) {
            console.error(err);
            showStatus(`خطأ في جلب الحروف: ${err.message}`, "var(--ruby)");
        }
    }

    // جلب الكلمات المصغرة من السيرفر
    async function loadBenchmarkWords() {
        try {
            const res = await fetch('/api/benchmark');
            if (!res.ok) throw new Error("فشل جلب الكلمات المختصرة");
            benchmarkWords = await res.json();
            renderBenchmarkTags();
        } catch (err) {
            console.error(err);
            showStatus(`خطأ في جلب معجم المعايرة: ${err.message}`, "var(--ruby)");
        }
    }

    // بناء شبكة اختيار الحروف
    function renderLetterSelector() {
        lettersSelectorGrid.innerHTML = '';
        Object.keys(lettersData).forEach(ch => {
            const btn = document.createElement('button');
            btn.className = 'letter-btn';
            btn.textContent = ch;
            btn.dataset.letter = ch;
            btn.addEventListener('click', () => selectLetter(ch));
            lettersSelectorGrid.appendChild(btn);
        });
    }

    // اختيار حرف معين وعرض منزلقاته + تفسيره
    async function selectLetter(letter) {
        selectedLetter = letter;
        currentLetterTitle.textContent = letter;
        
        // تمييز الزر النشط بصرياً
        document.querySelectorAll('.letter-btn').forEach(btn => {
            if (btn.dataset.letter === letter) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        renderSliders(letter);
        await loadLetterInterpretation(letter);
    }

    // تحميل تفسير المتجه والمعاني المتفرعة للحرف
    async function loadLetterInterpretation(letter) {
        try {
            const res = await fetch(`/api/letter/${encodeURIComponent(letter)}`);
            if (!res.ok) throw new Error("فشل جلب تفسير الحرف");
            const info = await res.json();

            // عرض المعاني المتفرعة
            let meaningsHtml = `<div class="rich-meaning-card">
                <span class="meaning-core">${info.core_meaning}</span>
                <div class="meaning-branches">
                    <span class="meaning-label">التفرعات:</span>
                    <span class="meaning-chain">${info.branches.join(' > ')}</span>
                </div>
                <div class="meaning-opposite">
                    <span class="meaning-label">الضد:</span> ${info.opposite}
                </div>
                <div class="meaning-standard">
                    <span class="meaning-label">المعيار:</span> ${info.standard_of}
                </div>
                <div class="meaning-meta">
                    <span>المؤثر: ${info.operator}</span>
                    <span>التردد الذاتي: ${info.omega_0}</span>
                </div>
            </div>`;
            letterRichMeanings.innerHTML = meaningsHtml;

            // عرض تفسير المتجه
            const interp = info.vector_interpretation;
            let interpHtml = `<p class="interp-summary">${interp.summary}</p>`;

            if (interp.dominant && interp.dominant.length > 0) {
                interpHtml += '<div class="interp-list"><span class="interp-label positive">+ نشط:</span><ul>';
                interp.dominant.forEach(([name, desc, val]) => {
                    interpHtml += `<li><strong>${name}</strong> (${val.toFixed(2)}): ${desc}</li>`;
                });
                interpHtml += '</ul></div>';
            }
            if (interp.opposite && interp.opposite.length > 0) {
                interpHtml += '<div class="interp-list"><span class="interp-label negative">- منخفض:</span><ul>';
                interp.opposite.forEach(([name, desc, val]) => {
                    interpHtml += `<li><strong>${name}</strong> (${val.toFixed(2)}): ${desc}</li>`;
                });
                interpHtml += '</ul></div>';
            }
            letterVectorInterp.innerHTML = interpHtml;
            letterInterpPanel.style.display = 'block';

        } catch (err) {
            console.error(err);
            letterInterpPanel.style.display = 'none';
        }
    }

    // بناء منزلقات أبعاد الحرف
    function renderSliders(letter) {
        slidersContainerGrid.innerHTML = '';
        const v = lettersData[letter].v;
        
        dimNames.forEach((dim, idx) => {
            const val = v[idx];
            const card = document.createElement('div');
            card.className = 'slider-card';
            
            const trans = dimTranslations[dim] || dim;
            
            card.innerHTML = `
                <div class="slider-info-row">
                    <div class="slider-names">
                        <span class="slider-name-en">${dim}</span>
                        <span class="slider-name-ar">${trans}</span>
                    </div>
                    <span class="slider-val" id="val-${dim}">${val.toFixed(2)}</span>
                </div>
                <input type="range" min="-1" max="1" step="0.1" value="${val}" id="input-${dim}">
            `;
            
            const rangeInput = card.querySelector('input');
            const valSpan = card.querySelector('.slider-val');
            
            // تحديث فوري أثناء السحب (أداء عالي + سحب حي)
            rangeInput.addEventListener('input', (e) => {
                const numericVal = parseFloat(e.target.value);
                valSpan.textContent = numericVal.toFixed(2);
                
                // تحديث المتجه المحلي الكاش
                lettersData[letter].v[idx] = numericVal;
                
                // جدولة طلب التحديث التفاعلي فورا خلال 40 مللي ثانية
                throttleUpdate();
            });
            
            slidersContainerGrid.appendChild(card);
        });
    }

    // آلية تصفية وتحديد معدل تحديث السحب (Throttling)
    function throttleUpdate() {
        if (updateTimeout) clearTimeout(updateTimeout);
        updateTimeout = setTimeout(async () => {
            await applyLetterCalibration(false); // تحديث مؤقت في الخلفية دون إيقاظ مؤشرات الحظر
        }, 45);
    }

    // إرسال التعديلات الطورية للحرف المختار للذاكرة في الخلفية
    async function applyLetterCalibration(showText = true) {
        if (!selectedLetter) return;
        
        if (showText) {
            applyCalibrationBtn.disabled = true;
            showStatus(`جاري تطبيق التعديل للحرف "${selectedLetter}" في الذاكرة...`, "var(--text-muted)");
        }
        
        try {
            const v = lettersData[selectedLetter].v;
            const res = await fetch('/api/letters/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    letter: selectedLetter,
                    v: v
                })
            });
            
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "فشل التطبيق");
            }
            
            if (showText) {
                showStatus(`تم تطبيق تعديلات الحرف "${selectedLetter}" مؤقتاً!`, "var(--emerald)");
            }
            
            // حلقة التغذية الراجعة اللحظية: إعادة التحليل فورا للكلمة النشطة
            if (lastAnalyzedWord) {
                triggerAnalysisSilent(lastAnalyzedWord);
            }
        } catch (err) {
            console.error(err);
            showStatus(`خطأ في المعايرة: ${err.message}`, "var(--ruby)");
        } finally {
            if (showText) applyCalibrationBtn.disabled = false;
        }
    }

    // الحفظ النهائي للحروف في JSON
    saveCalibrationBtn.addEventListener('click', async () => {
        saveCalibrationBtn.disabled = true;
        showStatus("جاري حفظ مصفوفة الحروف نهائياً إلى JSON...", "var(--text-muted)");
        try {
            const res = await fetch('/api/letters/save', { method: 'POST' });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "فشل الحفظ");
            }
            showStatus("تم حفظ مصفوفة الحروف بنجاح وتحديث JSON المصدري!", "var(--emerald)");
        } catch (err) {
            console.error(err);
            showStatus(`خطأ أثناء حفظ الحروف: ${err.message}`, "var(--ruby)");
        } finally {
            saveCalibrationBtn.disabled = false;
        }
    });

    // ربط زر المعايرة العادية يدوياً
    applyCalibrationBtn.addEventListener('click', () => applyLetterCalibration(true));

    // --- إدارة الكلمات المصغرة (Benchmark Vocab) ---
    
    // رندرة الكلمات المصغرة كـ تاقات قابلة للحذف
    function renderBenchmarkTags() {
        benchmarkWordsListContainer.innerHTML = '';
        benchmarkWords.forEach(word => {
            const tag = document.createElement('span');
            tag.className = 'benchmark-tag';
            tag.innerHTML = `
                ${word}
                <i class="fa-solid fa-xmark" title="حذف"></i>
            `;
            
            // زر الحذف
            tag.querySelector('i').addEventListener('click', () => {
                benchmarkWords = benchmarkWords.filter(w => w !== word);
                renderBenchmarkTags();
            });
            
            benchmarkWordsListContainer.appendChild(tag);
        });
    }

    // إضافة كلمة للـ Benchmark محلياً
    addBenchmarkBtn.addEventListener('click', addWordFromInput);
    newBenchmarkInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            addWordFromInput();
        }
    });

    function addWordFromInput() {
        const word = newBenchmarkInput.value.trim();
        if (!word) return;
        if (benchmarkWords.includes(word)) {
            newBenchmarkInput.value = '';
            return;
        }
        benchmarkWords.push(word);
        renderBenchmarkTags();
        newBenchmarkInput.value = '';
        newBenchmarkInput.focus();
    }

    // حفظ الكلمات المصغرة نهائياً في ملف JSON وإعادة بناء الموجات
    saveBenchmarkBtn.addEventListener('click', async () => {
        saveBenchmarkBtn.disabled = true;
        showStatus("جاري حفظ قائمة معجم المعايرة المصغر للـ JSON وإعادة البناء...", "var(--text-muted)");
        try {
            const res = await fetch('/api/benchmark/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ words: benchmarkWords })
            });
            
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "فشل التحديث");
            }
            
            showStatus("تم حفظ وتحديث معجم المعايرة بنجاح وإعادة بناء متجهات الكلمات!", "var(--emerald)");
            
            // إعادة تحليل الكلمة الحالية لتنعكس بالمعجم الجديد
            if (lastAnalyzedWord) {
                triggerAnalysis(lastAnalyzedWord);
            }
        } catch (err) {
            console.error(err);
            showStatus(`خطأ في حفظ قائمة المعايرة: ${err.message}`, "var(--ruby)");
        } finally {
            saveBenchmarkBtn.disabled = false;
        }
    });


    // --- منطق تحليل المجال الرئيسي ---

    analysisForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const word = wordInput.value.trim();
        if (!word) return;

        if (word.split(/\s+/).length > 1) {
            showStatus("تنبيه: يرجى إدخال كلمة واحدة فقط بدون مسافات.", "var(--ruby)");
            return;
        }

        await triggerAnalysis(word);
    });

    async function triggerAnalysis(word) {
        if (currentMode === 'define') {
            await triggerDefinition(word);
        } else {
            await triggerFieldAnalysis(word);
        }
    }

    // تفكيك دلالي
    async function triggerDefinition(word) {
        wordInput.disabled = true;
        analyzeBtn.disabled = true;
        loadingIndicator.style.display = 'flex';
        showStatus("جاري التفكيك الدلالي للحروف...", "var(--text-muted)");
        timeTakenSpan.textContent = "";

        try {
            const res = await fetch('/api/define', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ word: word })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "حدث خطأ");
            }

            const data = await res.json();
            lastAnalyzedWord = word;
            wordInput.value = word;

            welcomeBox.style.display = 'none';
            resultsGrid.style.display = 'none';
            defineResults.style.display = 'block';
            renderDefinition(data);

            showStatus(`تم تفكيك "${word}" إلى ${data.letters.length} حروف`, "var(--emerald)");

        } catch (err) {
            console.error(err);
            showStatus(`خطأ: ${err.message}`, "var(--ruby)");
        } finally {
            wordInput.disabled = false;
            analyzeBtn.disabled = false;
            loadingIndicator.style.display = 'none';
            wordInput.focus();
        }
    }

    // عرض نتائج التفكيك الدلالي
    function renderDefinition(data) {
        defineSummary.innerHTML = `<span class="define-word">${data.word}</span>
            <span class="define-composition">= ${data.definition}</span>`;

        defineLettersGrid.innerHTML = '';
        data.letters.forEach(item => {
            const card = document.createElement('div');
            card.className = 'define-letter-card';
            card.innerHTML = `
                <div class="dlc-header">
                    <span class="dlc-letter">${item.letter}</span>
                    <span class="dlc-core">${item.meaning}</span>
                </div>
                <div class="dlc-branches">
                    <span class="dlc-label">التفرعات:</span>
                    <span>${item.branches.join(' > ')}</span>
                </div>
                <div class="dlc-opposite">
                    <span class="dlc-label">الضد:</span> ${item.opposite}
                </div>
                <div class="dlc-standard">
                    <span class="dlc-label">المعيار:</span> ${item.standard_of}
                </div>
            `;
            defineLettersGrid.appendChild(card);
        });
    }

    // تحليل المجال (القديم)
    async function triggerFieldAnalysis(word) {
        const spaceType = spaceSelect.value;
        const topK = parseInt(kSlider.value);
        const vocabType = vocabSelect.value;

        wordInput.disabled = true;
        analyzeBtn.disabled = true;
        loadingIndicator.style.display = 'flex';
        showStatus("جاري التحليل الفيزيائي وسبر المتجهات...", "var(--text-muted)");
        timeTakenSpan.textContent = "";

        try {
            const response = await fetch('/api/field', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    word: word,
                    space_type: spaceType,
                    top_k: topK,
                    use_hebbian: hebbianToggle.checked,
                    vocab_type: vocabType
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "حدث خطأ في السيرفر");
            }

            const data = await response.json();
            
            lastAnalyzedWord = word;
            wordInput.value = word;
            
            welcomeBox.style.display = 'none';
            defineResults.style.display = 'none';
            resultsGrid.style.display = 'grid';

            renderResults(data.attracted, data.repelled);

            showStatus(`تم فحص حقل الكلمة "${data.word}" بنجاح في فضاء ${getSpaceNameArabic(data.space_type)}`, "var(--emerald)");
            timeTakenSpan.textContent = `الوقت: ${data.time_taken.toFixed(3)} ثانية`;

        } catch (error) {
            console.error(error);
            showStatus(`خطأ: ${error.message}`, "var(--ruby)");
        } finally {
            wordInput.disabled = false;
            analyzeBtn.disabled = false;
            loadingIndicator.style.display = 'none';
            wordInput.focus();
        }
    }

    // دالة تحليل صامتة (سريعة جداً - بدون حظر إدخال أو مؤشر تحميل) تُستخدم أثناء سحب السلايدر
    async function triggerAnalysisSilent(word) {
        if (currentMode !== 'field') return;
        
        const spaceType = spaceSelect.value;
        const topK = parseInt(kSlider.value);
        const vocabType = vocabSelect.value;

        try {
            const response = await fetch('/api/field', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    word: word,
                    space_type: spaceType,
                    top_k: topK,
                    use_hebbian: hebbianToggle.checked,
                    vocab_type: vocabType
                })
            });

            if (!response.ok) return;

            const data = await response.json();
            renderResults(data.attracted, data.repelled);
            timeTakenSpan.textContent = `الوقت: ${data.time_taken.toFixed(3)} ثانية`;
        } catch (error) {
            console.error(error);
        }
    }

    // رندرة الكلمات المتجاذبة والمتنافرة
    function renderResults(attracted, repelled) {
        attractedList.innerHTML = '';
        repelledList.innerHTML = '';

        if (attracted.length === 0) {
            attractedList.innerHTML = `<div class="word-card" style="text-align:center; color:var(--text-muted);">لا توجد نتائج تجاذب كافية</div>`;
        } else {
            attracted.forEach(item => {
                const card = createWordCard(item.word, item.score, 'attracted');
                attractedList.appendChild(card);
            });
        }

        if (repelled.length === 0) {
            repelledList.innerHTML = `<div class="word-card" style="text-align:center; color:var(--text-muted);">لا توجد نتائج تنافر كافية</div>`;
        } else {
            repelled.forEach(item => {
                const card = createWordCard(item.word, item.score, 'repelled');
                repelledList.appendChild(card);
            });
        }
    }

    // إنشاء بطاقة الكلمة مع شريط التشابه الطوري
    function createWordCard(word, score, type) {
        const card = document.createElement('div');
        card.className = 'word-card';

        let percentage = 0;
        if (type === 'attracted') {
            percentage = Math.max(0, Math.min(100, score * 100));
        } else {
            percentage = score < 0 ? Math.min(100, Math.abs(score) * 100) : 0;
        }

        card.innerHTML = `
            <div class="word-info-row">
                <span class="word-text">${word}</span>
                <span class="score-text">${score.toFixed(3)}</span>
            </div>
            <div class="bar-container">
                <div class="bar-fill" style="width: 0%"></div>
            </div>
        `;

        setTimeout(() => {
            const fill = card.querySelector('.bar-fill');
            if (fill) fill.style.width = `${percentage}%`;
        }, 50);

        return card;
    }

    function showStatus(text, color = "var(--text-muted)") {
        statusText.textContent = text;
        statusText.style.color = color;
    }

    function getSpaceNameArabic(type) {
        switch (type) {
            case 'physical': return 'الطور المادي الأصواتي';
            case 'philosophical': return 'الطور الفلسفي الدلالي';
            default: return 'الطور المدمج الكامل';
        }
    }
});
