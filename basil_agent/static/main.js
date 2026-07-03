// Majnoon v27.0 — Sovereign UI Engine
// Live Monitor + Web Search + Process Manager + Full Dashboard

document.addEventListener('DOMContentLoaded', () => {

    // ===== Core Elements =====
    const chatMessages   = document.getElementById('chat-messages');
    const thoughtLog     = document.getElementById('thought-log');
    const userInput      = document.getElementById('user-input');
    const sendBtn        = document.getElementById('send-btn');
    const stopBtn        = document.getElementById('stop-btn');
    const agentPulse     = document.getElementById('agent-pulse');
    const agentStatus    = document.getElementById('agent-status-text');
    const resetSessionBtn= document.getElementById('reset-session-btn');
    const gitBadge       = document.getElementById('git-status-badge');
    const toastContainer = document.getElementById('toast-container');
    const togglePreviewBtn  = document.getElementById('toggle-preview-btn');
    const previewPanel      = document.getElementById('preview-panel');
    const closePreviewBtn   = document.getElementById('close-preview-btn');
    const refreshPreviewBtn = document.getElementById('refresh-preview-btn');
    const previewUrl        = document.getElementById('preview-url');
    const previewFrame      = document.getElementById('preview-frame');

    // Tabs (v29.5 — cognitive/qalam removed)
    const tabBtns        = document.querySelectorAll('.tab-btn');
    const tabContents    = document.querySelectorAll('.tab-content');

    // Live Monitor
    const agentScreen  = document.getElementById('agent-screen');
    const lmPulse      = document.getElementById('lm-pulse');
    const lmTurn       = document.getElementById('lm-turn');
    const lmAutoBadge  = document.getElementById('lm-auto-badge');
    const sbTools      = document.getElementById('sb-tools');
    const sbReplan     = document.getElementById('sb-replan');
    const sbRag        = document.getElementById('sb-rag');
    const sbCpu        = document.getElementById('sb-cpu');
    const sbRam        = document.getElementById('sb-ram');

    // World State
    const wsStepBadge    = document.getElementById('ws-step-badge');
    const wsProgressVal  = document.getElementById('ws-progress-val');
    const wsBar          = document.getElementById('ws-bar');
    const wsToolsVal     = document.getElementById('ws-tools-val');
    const wsBacktrackVal = document.getElementById('ws-backtrack-val');
    const wsIssuesVal    = document.getElementById('ws-issues-val');
    const wsSnapshotsVal = document.getElementById('ws-snapshots-val');
    const wsFilesVal     = document.getElementById('ws-files-val');
    const wsMonologueText= document.getElementById('ws-monologue-text');

    let isProcessing = false;
    let currentAbortController = null;
    let turnCount = 0;
    const MAX_SCREEN_LINES = 100;

    // ═══════════════════════════════════════════
    //  LIVE AGENT SCREEN
    // ═══════════════════════════════════════════

    function lmAddLine(text, type = 'system') {
        if (!agentScreen) return;
        const line = document.createElement('div');
        line.className = `line ${type}`;

        // Icon prefix
        const icons = {
            thought: '💭',
            action:  '⚡',
            result:  '✅',
            error:   '❌',
            system:  '▸',
            auto:    '◈',
        };
        const prefix = icons[type] || '▸';

        // Truncate long lines
        const display = text.length > 120 ? text.slice(0, 120) + '…' : text;
        line.textContent = `${prefix} ${display}`;
        agentScreen.appendChild(line);

        // Keep max lines
        const lines = agentScreen.querySelectorAll('.line');
        if (lines.length > MAX_SCREEN_LINES) lines[0].remove();

        agentScreen.scrollTop = agentScreen.scrollHeight;
    }

    function lmSetActive(active) {
        if (!lmPulse) return;
        lmPulse.className = `lm-pulse${active ? '' : ' idle'}`;
    }

    function lmUpdateTurn(n, max = 50) {
        if (lmTurn) lmTurn.textContent = `دورة ${n} / ${max}`;
    }

    // ═══════════════════════════════════════════
    //  TOAST NOTIFICATIONS
    // ═══════════════════════════════════════════

    function showToast(level, title, body) {
        const icons = { success: 'fa-circle-check', warning: 'fa-triangle-exclamation', error: 'fa-circle-xmark', info: 'fa-circle-info' };
        const toast = document.createElement('div');
        toast.className = `toast toast-${level}`;
        toast.innerHTML = `
            <i class="fa-solid ${icons[level] || 'fa-bell'}"></i>
            <div class="toast-body">
                <div class="toast-title">${title}</div>
                <div class="toast-msg">${body}</div>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()"><i class="fa-solid fa-xmark"></i></button>
        `;
        toastContainer.appendChild(toast);
        setTimeout(() => { if (toast.parentElement) toast.remove(); }, 6000);
    }

    function connectNotifications() {
        const es = new EventSource('/notifications/stream');
        es.onmessage = (e) => {
            try {
                const n = JSON.parse(e.data);
                showToast(n.level || 'info', n.title || '', n.body || '');
                if (n.title === 'استمرار تلقائي' && lmAutoBadge) {
                    lmAutoBadge.style.display = 'inline';
                    setTimeout(() => { lmAutoBadge.style.display = 'none'; }, 5000);
                }
            } catch (_) {}
        };
        es.onerror = () => { es.close(); setTimeout(connectNotifications, 5000); };
    }
    connectNotifications();

    // ═══════════════════════════════════════════
    //  WORLD STATE + STATUS BAR
    // ═══════════════════════════════════════════

    function updateWorldState() {
        fetch('/worldstate')
            .then(r => r.json())
            .then(data => {
                if (data.status !== 'ok') return;
                const ws   = data.world_state || {};
                const bt   = data.backtracking || {};
                const prog = data.progress || {};
                const sess = data.session || {};
                const mono = data.inner_monologue || [];

                wsStepBadge.textContent = `خطوة ${ws.step || 0}`;
                const pct = prog.percent ? Math.round(prog.percent) : 0;
                wsProgressVal.textContent = `${pct}%`;
                wsBar.style.width = `${pct}%`;
                wsBar.style.background = pct >= 80 ? '#10b981' : pct >= 40 ? '#f59e0b' : '#6366f1';
                wsToolsVal.textContent    = sess.total_tool_calls ?? 0;
                wsBacktrackVal.textContent= `${bt.backtrack_count ?? 0} / ${bt.max_backtracks ?? 5}`;
                wsIssuesVal.textContent   = bt.detected_issues ?? 0;
                wsSnapshotsVal.textContent= bt.available_snapshots ?? 0;
                const files = sess.files_modified || [];
                wsFilesVal.textContent = files.length ? files.map(f => f.split(/[/\\]/).pop()).join(', ') : 'لا شيء بعد';
                if (mono.length) wsMonologueText.textContent = mono[mono.length-1].thought || '...';

                // Status bar
                if (sbTools)  sbTools.textContent  = sess.total_tool_calls ?? 0;
                if (sbReplan) sbReplan.textContent  = sess.replans ?? 0;
            })
            .catch(() => {});
    }

    let wsPollInterval = setInterval(updateWorldState, 10000);
    updateWorldState();

    // ═══════════════════════════════════════════
    //  TASK BOARD
    // ═══════════════════════════════════════════

    const STATUS_ICONS = {
        completed:  '<i class="fa-solid fa-circle-check" style="color:#4ade80"></i>',
        in_progress:'<i class="fa-solid fa-spinner fa-spin" style="color:#fbbf24"></i>',
        failed:     '<i class="fa-solid fa-circle-xmark" style="color:#f87171"></i>',
        pending:    '<i class="fa-regular fa-circle" style="color:#475569"></i>'
    };

    function renderTaskBoard(data) {
        if (!data || data.status !== 'ok') return;
        const tbProgressFill  = document.getElementById('tb-progress-fill');
        const tbProgressBadge = document.getElementById('tb-progress-badge');
        const tbGoal          = document.getElementById('tb-goal');
        const tbPhases        = document.getElementById('tb-phases');
        const tbTimeline      = document.getElementById('tb-timeline');

        const pct = data.progress?.percent ?? 0;
        if (tbProgressFill) tbProgressFill.style.width = pct + '%';
        if (tbProgressBadge) tbProgressBadge.textContent = pct + '%';
        if (tbGoal) tbGoal.textContent = (data.goal || 'لا توجد مهمة نشطة').slice(0, 90);

        if (tbPhases && data.phases?.length) {
            tbPhases.innerHTML = '';
            data.phases.forEach(phase => {
                const ph = document.createElement('div');
                ph.className = 'tb-phase';
                ph.innerHTML = `<div class="tb-phase-name">${phase.name}</div>`;
                (phase.tasks || []).forEach(task => {
                    const t = document.createElement('div');
                    t.className = `tb-task ${task.status}`;
                    t.innerHTML = `<span class="tb-task-icon">${STATUS_ICONS[task.status] || ''}</span><span class="tb-task-text">${task.description}</span>`;
                    ph.appendChild(t);
                });
                tbPhases.appendChild(ph);
            });
        }

        const res = data.resources || {};
        const tbCpu = document.getElementById('tb-cpu');
        const tbRam = document.getElementById('tb-ram');
        const tbTools= document.getElementById('tb-tools');
        const tbRag  = document.getElementById('tb-rag');
        if (tbCpu)   tbCpu.textContent   = (res.cpu_percent ?? '--') + '%';
        if (tbRam)   tbRam.textContent   = res.ram_used_gb ? res.ram_used_gb.toFixed(1) + 'GB' : '--GB';
        if (tbTools) tbTools.textContent = data.session?.tool_calls ?? 0;
        if (tbRag)   tbRag.textContent   = (data.rag?.summary || '').split('\n')[0].replace('[RAG STATS] Total memories: ', '') || '--';
        if (sbCpu)   sbCpu.textContent   = (res.cpu_percent ?? '--') + '%';
        if (sbRam)   sbRam.textContent   = res.ram_used_gb ? res.ram_used_gb.toFixed(1) + 'GB' : '--GB';
        if (sbRag)   sbRag.textContent   = tbRag?.textContent || '--';

        if (tbTimeline && data.recent_actions?.length) {
            tbTimeline.innerHTML = '';
            [...data.recent_actions].reverse().forEach(a => {
                const item = document.createElement('div');
                item.className = 'tb-timeline-item';
                item.innerHTML = `<span class="tb-timeline-tool">${a.tool}</span><span class="tb-timeline-preview">${a.preview}</span>`;
                tbTimeline.appendChild(item);
            });
        }
    }

    function updateTaskBoard() {
        fetch('/api/taskboard').then(r => r.json()).then(renderTaskBoard).catch(() => {});
    }

    let tbPollInterval = setInterval(updateTaskBoard, 8000);
    updateTaskBoard();

    // ═══════════════════════════════════════════
    //  v29.0 COMMAND CENTER
    // ═══════════════════════════════════════════

    function updateCmdFiles(fileList) {
        const list = document.getElementById('cmd-file-list');
        const count = document.getElementById('cmd-file-count');
        if (!list) return;
        if (!fileList || fileList.length === 0) {
            list.innerHTML = '<div class="file-item" style="color:#6b7280">لا توجد ملفات</div>';
            if (count) count.textContent = '0';
            return;
        }
        if (count) count.textContent = fileList.length;
        const extIcon = { py:'py', js:'js', html:'html', css:'css', json:'json', md:'md', txt:'txt' };
        list.innerHTML = fileList.slice(0, 20).map(f => {
            const ext = f.split('.').pop();
            const icon = extIcon[ext] || 'file';
            const isDir = f.endsWith('/');
            return `<div class="file-item" onclick="quickCmd('read_file(\"${f}\")')" title="اضغط للقراءة">
                <i class="fa-solid fa-${isDir ? 'folder' : 'file-code'} icon ${icon}"></i> ${f}
            </div>`;
        }).join('');
        if (fileList.length > 20) {
            list.innerHTML += `<div class="file-item" style="color:#6b7280">... و ${fileList.length - 20} ملف آخر</div>`;
        }
    }

    function updateCmdDebug(debugData) {
        const list = document.getElementById('cmd-debug-list');
        const count = document.getElementById('cmd-debug-count');
        if (!list) return;
        if (!debugData || debugData.length === 0) {
            list.innerHTML = '<div class="debug-entry"><span class="ts">--</span> لا توجد تشخيصات</div>';
            if (count) count.textContent = '0';
            return;
        }
        if (count) count.textContent = debugData.length;
        list.innerHTML = debugData.slice(-10).reverse().map(d => `
            <div class="debug-entry">
                <span class="ts">${d.ts || '--'}</span>
                <span class="type">${d.type || 'INFO'}</span>
                <span class="msg">${d.msg || ''}</span>
            </div>
        `).join('');
    }

    function updateCmdTerminal(termLine) {
        const list = document.getElementById('cmd-term-list');
        const count = document.getElementById('cmd-term-count');
        if (!list) return;
        if (!termLine) return;
        const maxLines = 50;
        const entry = document.createElement('div');
        entry.textContent = termLine;
        list.appendChild(entry);
        while (list.children.length > maxLines) list.removeChild(list.firstChild);
        list.scrollTop = list.scrollHeight;
        if (count) count.textContent = list.children.length;
    }

    function updateCmdCenter() {
        Promise.all([
            fetch('/api/taskboard').then(r => r.json()).catch(() => ({})),
            fetch('/cmd/center').then(r => r.json()).catch(() => ({})),
        ]).then(([tb, cmd]) => {
            const actions = tb?.recent_actions || [];
            actions.forEach(a => {
                if (a.tool && a.preview) {
                    updateCmdTerminal(`[${a.tool}] ${a.preview.slice(0, 100)}`);
                }
            });
            if (cmd?.files) updateCmdFiles(cmd.files);
            if (cmd?.debug_history) updateCmdDebug(cmd.debug_history);
        });
    }

    // Command Center poll
    let cmdPollInterval = setInterval(updateCmdCenter, 5000);
    setTimeout(updateCmdCenter, 1000);

    function showHilControls(show, context) {
        const hil = document.getElementById('hil-controls');
        if (!hil) return;
        if (show) {
            hil.style.display = 'flex';
            hil.dataset.context = context || '';
            showToast('warning', 'موافقة المستخدم مطلوبة', context ? context.slice(0, 80) : 'الوكيل ينتظر قرارك');
        } else {
            hil.style.display = 'none';
            hil.dataset.context = '';
        }
    }

    window.hilAction = function(action) {
        const hil = document.getElementById('hil-controls');
        const context = hil?.dataset?.context || '';
        showHilControls(false);

        let msg = '';
        if (action === 'approve') {
            msg = 'متابعة: ' + (context || 'تمت الموافقة، أكمل.');
        } else if (action === 'reject') {
            msg = 'توقف: ' + (context || 'غير موافق على هذا المسار. جرب بديلاً.');
        } else if (action === 'modify') {
            const mod = prompt('عدّل التعليمات للوكيل:', context || '');
            if (mod) msg = 'تعديل: ' + mod;
        }
        if (msg) {
            userInput.value = msg;
            sendMessage();
        }
    };

    // ═══════════════════════════════════════════
    //  PROCESSING MODE
    // ═══════════════════════════════════════════

    function setProcessingMode(active) {
        isProcessing = active;
        document.body.classList.toggle('processing', active);
        agentPulse.classList.toggle('active', active);
        lmSetActive(active);
        agentStatus.innerHTML = active
            ? '<i class="fa-solid fa-spinner fa-spin"></i> الوكيل يعمل...'
            : '<i class="fa-solid fa-eye"></i> جاهز — 100% Sovereign Prime (Majnoon)';
        clearInterval(wsPollInterval);
        clearInterval(tbPollInterval);
        wsPollInterval = setInterval(updateWorldState, active ? 3000 : 10000);
        tbPollInterval = setInterval(updateTaskBoard,  active ? 2500 : 8000);
        if (!active) { turnCount = 0; lmUpdateTurn(0); }
    }

    // ═══════════════════════════════════════════
    //  CHAT FUNCTIONS
    // ═══════════════════════════════════════════

    function addMessage(text, isUser = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = isUser ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
        const content = document.createElement('div');
        content.className = 'content';
        content.innerHTML = text
            .replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msgDiv;
    }

    function addLog(text, type = 'system') {
        const logDiv = document.createElement('div');
        logDiv.className = `log-entry ${type}`;
        const icons = { system:'fa-info-circle', tool:'fa-wrench', obs:'fa-eye', error:'fa-triangle-exclamation', success:'fa-check-circle' };
        logDiv.innerHTML = `<i class="fa-solid ${icons[type] || 'fa-circle'}"></i> ${text}`;
        thoughtLog.appendChild(logDiv);
        thoughtLog.scrollTop = thoughtLog.scrollHeight;
        if (type === 'error' && text.includes('CRITICAL')) showToast('error', 'خطأ حرج', text.slice(0, 100));
        if (text.includes('[BACKTRACKING]')) showToast('warning', 'تراجع استراتيجي', 'الوكيل يعيد التخطيط');
    }

    function createStreamMessage() {
        document.querySelector('.typing-indicator')?.closest('.message')?.remove();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant-message';
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';
        const content = document.createElement('div');
        content.className = 'content';
        content.id = 'streaming-content-' + Date.now();
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return content;
    }

    async function sendMessage(retryCount = 0) {
        if (isProcessing && retryCount === 0) return;
        const rawText = retryCount === 0 ? userInput.value.trim() : window._lastMsg;
        if (!rawText) return;

        if (retryCount === 0) {
            window._lastMsg = rawText;
            addMessage(rawText, true);
            lmAddLine(rawText.slice(0, 80), 'system');
            userInput.value = '';
            userInput.style.height = 'auto';
        }

        setProcessingMode(true);
        currentAbortController = new AbortController();
        let indicator = retryCount === 0
            ? addMessage('<div class="typing-indicator"><span></span><span></span><span></span></div>', false)
            : null;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Request-ID': Date.now() + '' },
                signal: currentAbortController.signal,
                body: JSON.stringify({ message: rawText })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.trim().startsWith('data: ')) continue;
                    const jsonStr = line.substring(6).trim();
                    if (!jsonStr || jsonStr === '[DONE]') continue;
                    try {
                        const data = JSON.parse(jsonStr);
                        const content = data.content || '';

                        // ── Command Center feed ──
                        if (data.type === 'system') {
                            if (content.includes('[AUTO_DEBUG]') || content.includes('[DEBUG_INFER]')) {
                                updateCmdDebug([{ ts: new Date().toLocaleTimeString(), type: 'DEBUG', msg: content.slice(0, 100) }]);
                            }
                            if (content.includes('Tool Result') || content.includes('🛠️')) {
                                updateCmdTerminal(content.replace(/<[^>]+>/g, '').slice(0, 120));
                            }
                            if (content.includes('Human-in-the-loop') || content.includes('موافقة')) {
                                showHilControls(true, content);
                            }

                            // ── Live Monitor routing ──
                            addLog(content, 'system');
                            if (content.includes('Thought:')) {
                                lmAddLine(content.replace('Thought:', '').trim(), 'thought');
                            } else if (content.includes('ACTION:') || content.includes('Tool Result')) {
                                lmAddLine(content, 'action');
                                turnCount++;
                                lmUpdateTurn(turnCount);
                            } else if (content.includes('AUTO-CONTINUE')) {
                                lmAddLine(content, 'auto');
                                lmAutoBadge && (lmAutoBadge.style.display = 'inline');
                            } else if (content.includes('[OK]') || content.includes('TASK DONE')) {
                                lmAddLine(content, 'result');
                            } else if (content.includes('[FAIL]') || content.includes('ERROR')) {
                                lmAddLine(content, 'error');
                            } else {
                                lmAddLine(content, 'system');
                            }
                        }
                        else if (data.type === 'error') {
                            addLog(content, 'error');
                            lmAddLine(content, 'error');
                            updateCmdDebug([{ ts: new Date().toLocaleTimeString(), type: 'ERROR', msg: content.slice(0, 100) }]);
                        }
                        else if (data.type === 'assistant_start') {
                            indicator?.remove();
                            window._streamTarget = createStreamMessage();
                        }
                        else if (data.type === 'assistant_token') {
                            if (window._streamTarget) {
                                window._streamTarget.innerHTML += content.replace(/\n/g, '<br>');
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                        }
                        else if (data.type === 'assistant') {
                            indicator?.remove();
                            addMessage(content, false);
                        }
                    } catch (_) {}
                }
            }

            window._lastMsg = null;
            lmAddLine('المهمة اكتملت.', 'result');
            showToast('success', 'اكتملت الاستجابة', 'الوكيل أنهى معالجة طلبك');
            updateWorldState();
            updateTaskBoard();
            updateCmdCenter();

        } catch (err) {
            if (err.name === 'AbortError') {
                addLog('تم إلغاء العملية.', 'error');
                lmAddLine('إيقاف يدوي.', 'error');
            } else if (retryCount < 3) {
                const delay = Math.pow(2, retryCount) * 1000;
                addLog(`انقطع الاتصال. إعادة المحاولة بعد ${delay/1000}ث...`, 'error');
                setTimeout(() => sendMessage(retryCount + 1), delay);
                return;
            } else {
                addLog(`فشل الاتصال: ${err.message}`, 'error');
                showToast('error', 'فشل الاتصال', err.message);
            }
            indicator?.remove();
        } finally {
            if (retryCount === 0 || !isProcessing) {
                setProcessingMode(false);
                currentAbortController = null;
                lmAutoBadge && (lmAutoBadge.style.display = 'none');
            }
        }
    }

    stopBtn.addEventListener('click', () => {
        if (isProcessing && currentAbortController) {
            currentAbortController.abort();
            setProcessingMode(false);
            currentAbortController = null;
            addLog('تم إيقاف الوكيل يدوياً.', 'error');
            lmAddLine('إيقاف.', 'error');
            document.querySelector('.typing-indicator')?.closest('.message')?.remove();
        }
    });

    sendBtn.addEventListener('click', () => sendMessage());
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        setTimeout(() => {
            userInput.style.height = 'auto';
            userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
        }, 0);
    });

    // ═══════════════════════════════════════════
    //  QUICK ACTIONS & WEB SEARCH
    // ═══════════════════════════════════════════

    window.quickCmd = function(text) {
        userInput.value = text;
        sendMessage();
    };

    window.stopAgent = function() {
        document.getElementById('stop-btn')?.click();
    };

    window.webSearch = function() {
        const q = document.getElementById('ws-search-input')?.value?.trim();
        if (!q) return;
        userInput.value = `ابحث في الويب عن: ${q}`;
        sendMessage();
        document.getElementById('ws-search-input').value = '';
    };

    document.getElementById('ws-search-input')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') window.webSearch();
    });

    // ═══════════════════════════════════════════
    //  RESET + GIT + PREVIEW
    // ═══════════════════════════════════════════

    resetSessionBtn.addEventListener('click', async () => {
        if (isProcessing) currentAbortController?.abort();
        if (!confirm('هل تريد تصفير الجلسة والبدء من جديد؟')) return;
        try {
            const res = await fetch('/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
            const data = await res.json();
            if (data.status === 'success') {
                chatMessages.innerHTML = '';
                thoughtLog.innerHTML = '';
                agentScreen && (agentScreen.innerHTML = '<div class="line system">▸ جلسة جديدة...</div>');
                addMessage('مرحباً من جديد! جلسة جديدة نظيفة. كيف يمكنني مساعدتك؟', false);
                updateWorldState();
                showToast('info', 'جلسة جديدة', 'تم تصفير الوكيل بالكامل');
            }
        } catch (err) { addLog(`فشل التصفير: ${err.message}`, 'error'); }
    });

    async function updateGitBadge() {
        try {
            const res = await fetch('/git/status');
            const data = await res.json();
            gitBadge.className = `git-badge ${data.status}`;
            gitBadge.querySelector('span').textContent = `Git: ${data.branch}`;
        } catch { gitBadge.querySelector('span').textContent = 'Git: غير متصل'; }
    }
    setInterval(updateGitBadge, 8000);
    updateGitBadge();

    togglePreviewBtn.addEventListener('click', () => {
        const isHidden = previewPanel.style.display === 'none';
        previewPanel.style.display = isHidden ? 'flex' : 'none';
        if (isHidden && (previewFrame.src === 'about:blank' || previewFrame.src === window.location.href)) {
            previewFrame.src = previewUrl.value;
        }
    });
    closePreviewBtn.addEventListener('click', () => { previewPanel.style.display = 'none'; });
    refreshPreviewBtn.addEventListener('click', () => {
        previewFrame.src = 'about:blank';
        setTimeout(() => {
            let url = previewUrl.value;
            if (!url.startsWith('http')) url = 'http://' + url;
            previewUrl.value = url;
            previewFrame.src = url;
        }, 100);
    });
    previewUrl.addEventListener('keypress', e => { if (e.key === 'Enter') refreshPreviewBtn.click(); });

    // ═══════════════════════════════════════════
    //  TAB SWITCHING (v29.5 — Mufakir/Qalam removed)
    // ═══════════════════════════════════════════
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            tabContents.forEach(c => c.classList.remove('active'));
            const target = document.getElementById(`tab-${tabId}`);
            if (target) target.classList.add('active');
        });
    });

    // Init
    lmSetActive(false);
    lmAddLine('Majnoon v27 جاهز — Sovereign Prime %100', 'system');
    setTimeout(updateCmdCenter, 2000);
});
