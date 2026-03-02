// ===== CONFIGURATION =====
const API_BASE = '/api';
const REFRESH_INTERVAL = 2000; // 2 seconds

// ===== CLOCK =====
function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
}
setInterval(updateClock, 1000);
updateClock();

// ===== FETCH STATS =====
async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        if (!res.ok) return;
        const data = await res.json();

        // Update counters with animation
        animateValue('stat-total', data.total_requests);
        animateValue('stat-blocked', data.blocked);
        animateValue('stat-allowed', data.allowed);

        // Block rate
        const rate = data.total_requests > 0
            ? ((data.blocked / data.total_requests) * 100).toFixed(1)
            : 0;
        document.getElementById('stat-rate').textContent = rate + '%';

        // Attack bars
        const maxAttack = Math.max(
            data.attack_types.sqli,
            data.attack_types.xss,
            data.attack_types.path_traversal,
            data.attack_types.encoding,
            data.attack_types.other,
            1
        );

        updateBar('bar-sqli', 'count-sqli', data.attack_types.sqli, maxAttack);
        updateBar('bar-xss', 'count-xss', data.attack_types.xss, maxAttack);
        updateBar('bar-path', 'count-path', data.attack_types.path_traversal, maxAttack);
        updateBar('bar-encoding', 'count-encoding', data.attack_types.encoding, maxAttack);
        updateBar('bar-other', 'count-other', data.attack_types.other, maxAttack);

    } catch (e) {
        console.error('Stats fetch failed:', e);
    }
}

function updateBar(barId, countId, value, max) {
    const pct = max > 0 ? (value / max) * 100 : 0;
    document.getElementById(barId).style.width = pct + '%';
    document.getElementById(countId).textContent = value;
}

function animateValue(id, newValue) {
    const el = document.getElementById(id);
    const current = parseInt(el.textContent) || 0;
    if (current !== newValue) {
        el.textContent = newValue;
        el.style.transform = 'scale(1.15)';
        setTimeout(() => { el.style.transform = 'scale(1)'; }, 200);
    }
}

// ===== FETCH LOGS =====
async function fetchLogs() {
    try {
        const res = await fetch(`${API_BASE}/logs`);
        if (!res.ok) return;
        const logs = await res.json();

        const tbody = document.getElementById('log-body');
        document.getElementById('log-count').textContent = logs.length + ' entries';

        if (logs.length === 0) {
            tbody.innerHTML = '<tr class="empty-row"><td colspan="7">Waiting for requests...</td></tr>';
            return;
        }

        tbody.innerHTML = logs.map(log => {
            const rowClass = log.status === 'blocked' ? 'row-blocked' : 'row-allowed';
            const statusClass = log.status === 'blocked' ? 'status-blocked' : 'status-allowed';
            const statusLabel = log.status === 'blocked' ? '🚫 BLOCKED' : '✅ ALLOWED';

            return `<tr class="${rowClass}">
                <td>${log.timestamp}</td>
                <td>${escapeHtml(log.ip)}</td>
                <td>${log.method}</td>
                <td title="${escapeHtml(log.uri)}">${escapeHtml(log.uri)}</td>
                <td><span class="status-badge ${statusClass}">${statusLabel}</span></td>
                <td>${escapeHtml(log.reason)}</td>
                <td>${log.probability.toFixed(4)}</td>
            </tr>`;
        }).join('');

    } catch (e) {
        console.error('Logs fetch failed:', e);
    }
}

// ===== TEST URL =====
async function testUrl() {
    const input = document.getElementById('test-url');
    const url = input.value.trim();
    if (!url) return;

    const btn = document.getElementById('test-btn');
    btn.textContent = '⏳ Analyzing...';
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        const data = await res.json();

        const resultDiv = document.getElementById('test-result');
        resultDiv.classList.remove('hidden', 'blocked', 'allowed');
        resultDiv.classList.add(data.status === 'blocked' ? 'blocked' : 'allowed');

        document.getElementById('result-icon').textContent = data.status === 'blocked' ? '🚫' : '✅';
        document.getElementById('result-status').textContent = data.status === 'blocked' ? 'BLOCKED' : 'ALLOWED';
        document.getElementById('result-reason').textContent = 'Reason: ' + data.reason;
        document.getElementById('result-prob').textContent = 'Malicious Probability: ' + (data.probability * 100).toFixed(2) + '%';

        document.getElementById('result-status').style.color = data.status === 'blocked' ? '#ef4444' : '#10b981';

        // Refresh stats and logs immediately
        fetchStats();
        fetchLogs();
    } catch (e) {
        console.error('Test failed:', e);
    } finally {
        btn.textContent = 'Analyze';
        btn.disabled = false;
    }
}

function quickTest(url) {
    document.getElementById('test-url').value = url;
    testUrl();
}

// ===== UTILS =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== ENTER KEY =====
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('test-url').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') testUrl();
    });
});

// ===== POLLING =====
setInterval(() => {
    fetchStats();
    fetchLogs();
}, REFRESH_INTERVAL);

// Initial fetch
fetchStats();
fetchLogs();
