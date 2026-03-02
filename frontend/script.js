// ===== CONFIG =====
const API_BASE = '/api';
const REFRESH_MS = 2000;

// ===== CLOCK =====
function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
}
setInterval(updateClock, 1000);
updateClock();
document.getElementById('year').textContent = new Date().getFullYear();

// ===== CHART.JS — DOUGHNUT =====
const doughnutCtx = document.getElementById('attackDoughnut').getContext('2d');
const doughnutChart = new Chart(doughnutCtx, {
    type: 'doughnut',
    data: {
        labels: ['SQL Injection', 'XSS', 'Path Traversal', 'Encoding', 'Other'],
        datasets: [{
            data: [0, 0, 0, 0, 0],
            backgroundColor: ['#ef4444', '#f59e0b', '#06b6d4', '#8b5cf6', '#64748b'],
            borderColor: '#0b0f19',
            borderWidth: 3,
            hoverOffset: 8
        }]
    },
    options: {
        responsive: true,
        cutout: '65%',
        plugins: {
            legend: {
                position: 'bottom',
                labels: { color: '#94a3b8', padding: 14, font: { size: 11, family: 'Inter' } }
            }
        }
    }
});

// ===== CHART.JS — TIMELINE =====
const MAX_TIMELINE_POINTS = 30;
const timelineLabels = [];
const blockedData = [];
const allowedData = [];
let prevTotal = 0;
let prevBlocked = 0;
let prevAllowed = 0;

const timelineCtx = document.getElementById('timelineChart').getContext('2d');
const timelineChart = new Chart(timelineCtx, {
    type: 'line',
    data: {
        labels: timelineLabels,
        datasets: [
            {
                label: 'Blocked',
                data: blockedData,
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239,68,68,0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 2,
                borderWidth: 2
            },
            {
                label: 'Allowed',
                data: allowedData,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16,185,129,0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 2,
                borderWidth: 2
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
            x: {
                ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 10 },
                grid: { color: 'rgba(255,255,255,0.04)' }
            },
            y: {
                beginAtZero: true,
                ticks: { color: '#64748b', font: { size: 10 }, precision: 0 },
                grid: { color: 'rgba(255,255,255,0.04)' }
            }
        },
        plugins: {
            legend: {
                labels: { color: '#94a3b8', font: { size: 11, family: 'Inter' } }
            }
        }
    }
});

// ===== FETCH STATS =====
async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        if (!res.ok) return;
        const data = await res.json();

        // Animate stat values
        animateValue('stat-total', data.total_requests);
        animateValue('stat-blocked', data.blocked);
        animateValue('stat-allowed', data.allowed);
        const rate = data.total_requests > 0
            ? ((data.blocked / data.total_requests) * 100).toFixed(1)
            : 0;
        document.getElementById('stat-rate').textContent = rate + '%';

        // Update doughnut chart
        const at = data.attack_types;
        doughnutChart.data.datasets[0].data = [at.sqli, at.xss, at.path_traversal, at.encoding, at.other];
        doughnutChart.update('none');

        // Update progress bars
        const maxAtk = Math.max(at.sqli, at.xss, at.path_traversal, at.encoding, at.other, 1);
        updateBar('bar-sqli', 'count-sqli', at.sqli, maxAtk);
        updateBar('bar-xss', 'count-xss', at.xss, maxAtk);
        updateBar('bar-path', 'count-path', at.path_traversal, maxAtk);
        updateBar('bar-encoding', 'count-encoding', at.encoding, maxAtk);
        updateBar('bar-other', 'count-other', at.other, maxAtk);

        // Update timeline chart
        const now = new Date();
        const label = now.toLocaleTimeString('en-US', { hour12: false, minute: '2-digit', second: '2-digit' });
        const newBlocked = data.blocked - prevBlocked;
        const newAllowed = data.allowed - prevAllowed;

        timelineLabels.push(label);
        blockedData.push(Math.max(newBlocked, 0));
        allowedData.push(Math.max(newAllowed, 0));

        if (timelineLabels.length > MAX_TIMELINE_POINTS) {
            timelineLabels.shift();
            blockedData.shift();
            allowedData.shift();
        }
        timelineChart.update('none');

        prevTotal = data.total_requests;
        prevBlocked = data.blocked;
        prevAllowed = data.allowed;

    } catch (e) {
        console.error('Stats fetch error:', e);
    }
}

function updateBar(barId, countId, value, max) {
    const pct = max > 0 ? (value / max) * 100 : 0;
    document.getElementById(barId).style.width = pct + '%';
    document.getElementById(countId).textContent = value;
}

function animateValue(id, newVal) {
    const el = document.getElementById(id);
    const cur = parseInt(el.textContent) || 0;
    if (cur !== newVal) {
        el.textContent = newVal;
        el.style.transform = 'scale(1.12)';
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
            tbody.innerHTML = `<tr><td colspan="7" class="text-center text-secondary py-5 fst-italic">
                <i class="bi bi-hourglass-split me-2"></i>Waiting for requests...</td></tr>`;
            return;
        }

        tbody.innerHTML = logs.map(log => {
            const isBlocked = log.status === 'blocked';
            const badgeClass = isBlocked ? 'bg-danger' : 'bg-success';
            const icon = isBlocked ? 'bi-shield-x' : 'bi-shield-check';
            const label = isBlocked ? 'BLOCKED' : 'ALLOWED';
            const rowClass = isBlocked ? 'text-danger' : 'text-success';

            return `<tr class="${rowClass}">
                <td>${log.timestamp}</td>
                <td>${escapeHtml(log.ip)}</td>
                <td><span class="badge bg-secondary bg-opacity-50">${log.method}</span></td>
                <td title="${escapeHtml(log.uri)}">${escapeHtml(log.uri)}</td>
                <td><span class="badge ${badgeClass} bg-opacity-25 text-${isBlocked ? 'danger' : 'success'}">
                    <i class="bi ${icon} me-1"></i>${label}</span></td>
                <td>${escapeHtml(log.reason)}</td>
                <td class="font-monospace">${log.probability.toFixed(4)}</td>
            </tr>`;
        }).join('');

    } catch (e) {
        console.error('Logs fetch error:', e);
    }
}

// ===== TEST URL =====
async function testUrl() {
    const input = document.getElementById('test-url');
    const url = input.value.trim();
    if (!url) return;

    const btn = document.getElementById('test-btn');
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Analyzing...';
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        const data = await res.json();
        const isBlocked = data.status === 'blocked';

        const resultDiv = document.getElementById('test-result');
        resultDiv.classList.remove('d-none');

        const alertDiv = document.getElementById('result-alert');
        alertDiv.className = `alert d-flex align-items-center gap-3 mb-0 alert-${isBlocked ? 'danger' : 'success'}`;

        document.getElementById('result-icon').className = `bi ${isBlocked ? 'bi-shield-x' : 'bi-shield-check'} fs-1`;
        document.getElementById('result-status').textContent = isBlocked ? 'BLOCKED' : 'ALLOWED';
        document.getElementById('result-reason').textContent = 'Reason: ' + data.reason;
        document.getElementById('result-prob').textContent = 'Malicious Probability: ' + (data.probability * 100).toFixed(2) + '%';

        // Refresh data
        fetchStats();
        fetchLogs();
    } catch (e) {
        console.error('Test error:', e);
    } finally {
        btn.innerHTML = '<i class="bi bi-search me-1"></i> Analyze';
        btn.disabled = false;
    }
}

function quickTest(url) {
    document.getElementById('test-url').value = url;
    testUrl();
}

// ===== UTILS =====
function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

// ===== ENTER KEY =====
document.getElementById('test-url').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') testUrl();
});

// ===== POLLING =====
setInterval(() => { fetchStats(); fetchLogs(); }, REFRESH_MS);
fetchStats();
fetchLogs();
