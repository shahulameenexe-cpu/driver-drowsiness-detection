/**
 * Driver Drowsiness & Fatigue Detection System - Main Application Logic
 * Integrates MediaPipe FaceMesh, Real-time EAR/MAR math, Chart.js telemetry, and Demo Engine.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- UI Elements ---
    const videoWebcam = document.getElementById('webcamVideo');
    const canvasOverlay = document.getElementById('overlayCanvas');
    const ctx = canvasOverlay.getContext('2d');

    const btnStartCamera = document.getElementById('btnStartCamera');
    const btnStopCamera = document.getElementById('btnStopCamera');
    const btnToggleDemo = document.getElementById('btnToggleDemo');

    // Controls
    const inputEarThresh = document.getElementById('inputEarThresh');
    const txtEarThreshVal = document.getElementById('txtEarThreshVal');
    const lblEarThresh = document.getElementById('lblEarThresh');

    const inputDuration = document.getElementById('inputDuration');
    const txtDurationVal = document.getElementById('txtDurationVal');

    const inputMarThresh = document.getElementById('inputMarThresh');
    const txtMarThreshVal = document.getElementById('txtMarThreshVal');
    const lblMarThresh = document.getElementById('lblMarThresh');

    const selectSoundType = document.getElementById('selectSoundType');
    const inputVolume = document.getElementById('inputVolume');
    const txtVolumeVal = document.getElementById('txtVolumeVal');
    const chkVoiceAlert = document.getElementById('chkVoiceAlert');
    const btnTestAlarm = document.getElementById('btnTestAlarm');
    const btnTestVoice = document.getElementById('btnTestVoice');

    // Metrics Displays
    const valEAR = document.getElementById('valEAR');
    const barEAR = document.getElementById('barEAR');
    const valMAR = document.getElementById('valMAR');
    const barMAR = document.getElementById('barMAR');
    const valAttention = document.getElementById('valAttention');
    const barAttention = document.getElementById('barAttention');
    const valBpm = document.getElementById('valBpm');
    const barBpm = document.getElementById('barBpm');
    const systemStatusBadge = document.getElementById('systemStatusBadge');
    const statusText = document.getElementById('statusText');

    // Table & Logs
    const logTableBody = document.getElementById('logTableBody');
    const btnExportCsv = document.getElementById('btnExportCsv');
    const btnClearLog = document.getElementById('btnClearLog');

    // Demo Scenario Buttons
    const btnSimNormal = document.getElementById('btnSimNormal');
    const btnSimDrowsy = document.getElementById('btnSimDrowsy');
    const btnSimYawn = document.getElementById('btnSimYawn');
    const btnSimNod = document.getElementById('btnSimNod');

    // --- System State Variables ---
    let isCameraRunning = false;
    let isDemoMode = false;
    let cameraStream = null;
    let mediaPipeCamera = null;
    let faceMesh = null;
    let animationFrameId = null;

    // Thresholds
    let earThreshold = parseFloat(inputEarThresh.value);
    let marThreshold = parseFloat(inputMarThresh.value);
    let triggerDurationSec = parseFloat(inputDuration.value);

    // Tracking Counters
    let drowsyFrameCounter = 0;
    let yawnFrameCounter = 0;
    let headNodFrameCounter = 0;
    let blinkCountSession = 0;
    let lastBlinkTime = 0;
    let sessionStartTime = Date.now();
    let isAlarmActive = false;
    let isYawnWarningActive = false;
    let alarmStartTime = null;

    // Logs Array
    const incidentLogs = [];

    // --- Chart.js Setup ---
    let telemetryChart = null;
    const maxChartPoints = 30;
    const chartLabels = Array(maxChartPoints).fill('');
    const earDataBuffer = Array(maxChartPoints).fill(0.30);
    const marDataBuffer = Array(maxChartPoints).fill(0.18);

    function initChart() {
        const chartCtx = document.getElementById('telemetryChart').getContext('2d');
        telemetryChart = new Chart(chartCtx, {
            type: 'line',
            data: {
                labels: chartLabels,
                datasets: [
                    {
                        label: 'EAR (Eye Aspect Ratio)',
                        data: earDataBuffer,
                        borderColor: '#00f3ff',
                        backgroundColor: 'rgba(0, 243, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 0
                    },
                    {
                        label: 'MAR (Yawn Ratio)',
                        data: marDataBuffer,
                        borderColor: '#ffb700',
                        backgroundColor: 'rgba(255, 183, 0, 0.05)',
                        borderWidth: 2,
                        borderDash: [4, 4],
                        tension: 0.3,
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: false },
                    y: {
                        min: 0,
                        max: 0.8,
                        grid: { color: 'rgba(255, 255, 255, 0.08)' },
                        ticks: { color: '#94a3b8', font: { family: 'Rajdhani' } }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#f1f5f9', font: { family: 'Rajdhani', size: 12 } }
                    }
                }
            }
        });
    }
    initChart();

    function updateChart(ear, mar) {
        if (!telemetryChart) return;
        earDataBuffer.shift();
        earDataBuffer.push(ear);

        marDataBuffer.shift();
        marDataBuffer.push(mar);

        telemetryChart.update('none');
    }

    // --- Math Calculations ---
    function euclideanDistance(pt1, pt2) {
        return Math.sqrt(Math.pow(pt1.x - pt2.x, 2) + Math.pow(pt1.y - pt2.y, 2));
    }

    // EAR Calculation for 6 landmarks [p1, p2, p3, p4, p5, p6]
    function calculateEAR(eye) {
        const v1 = euclideanDistance(eye[1], eye[5]);
        const v2 = euclideanDistance(eye[2], eye[4]);
        const h = euclideanDistance(eye[0], eye[3]);
        if (h === 0) return 0.3;
        return (v1 + v2) / (2.0 * h);
    }

    // MAR Calculation for mouth landmarks [left, right, top, bottom]
    function calculateMAR(mouth) {
        const vertical = euclideanDistance(mouth.top, mouth.bottom);
        const horizontal = euclideanDistance(mouth.left, mouth.right);
        if (horizontal === 0) return 0.2;
        return vertical / horizontal;
    }

    // MediaPipe Landmark Index mappings
    const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144];
    const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380];
    const MOUTH_INDICES = { left: 61, right: 291, top: 13, bottom: 14 };

    // --- Initialize MediaPipe FaceMesh ---
    function initMediaPipe() {
        if (window.FaceMesh) {
            faceMesh = new window.FaceMesh({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
            });

            faceMesh.setOptions({
                maxNumFaces: 1,
                refineLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            faceMesh.onResults(onFaceMeshResults);
        }
    }
    initMediaPipe();

    // Handle MediaPipe Landmark Results
    function onFaceMeshResults(results) {
        if (!isCameraRunning) return;

        // Resize overlay canvas to match video
        canvasOverlay.width = videoWebcam.videoWidth || 640;
        canvasOverlay.height = videoWebcam.videoHeight || 480;

        ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            updateStatus('NO FACE DETECTED', 'warning');
            return;
        }

        const landmarks = results.multiFaceLandmarks[0];
        const width = canvasOverlay.width;
        const height = canvasOverlay.height;

        // Extract pixel coordinates for eyes & mouth
        const leftEye = LEFT_EYE_INDICES.map(idx => ({
            x: landmarks[idx].x * width,
            y: landmarks[idx].y * height
        }));

        const rightEye = RIGHT_EYE_INDICES.map(idx => ({
            x: landmarks[idx].x * width,
            y: landmarks[idx].y * height
        }));

        const mouth = {
            left: { x: landmarks[MOUTH_INDICES.left].x * width, y: landmarks[MOUTH_INDICES.left].y * height },
            right: { x: landmarks[MOUTH_INDICES.right].x * width, y: landmarks[MOUTH_INDICES.right].y * height },
            top: { x: landmarks[MOUTH_INDICES.top].x * width, y: landmarks[MOUTH_INDICES.top].y * height },
            bottom: { x: landmarks[MOUTH_INDICES.bottom].x * width, y: landmarks[MOUTH_INDICES.bottom].y * height }
        };

        const earLeft = calculateEAR(leftEye);
        const earRight = calculateEAR(rightEye);
        const earAvg = (earLeft + earRight) / 2.0;

        const mar = calculateMAR(mouth);

        // Pitch / Head droop estimation using nose tip (4) and chin (152)
        const noseTip = landmarks[4];
        const chin = landmarks[152];
        const headPitchDiff = (chin.y - noseTip.y);
        const isHeadDrooping = headPitchDiff < 0.18; // chin closer to nose in 2D projection

        // Process Telemetry Frame
        processFrameMetrics(earAvg, mar, isHeadDrooping);

        // Draw HUD visuals overlay on canvas
        drawMeshOverlay(leftEye, rightEye, mouth, earAvg, mar, isHeadDrooping);
    }

    // --- Process Telemetry Metrics & Trigger Alarms ---
    function processFrameMetrics(ear, mar, headDroop) {
        // Update metric UI text
        valEAR.innerText = ear.toFixed(2);
        barEAR.style.width = Math.min(100, (ear / 0.4) * 100) + '%';

        valMAR.innerText = mar.toFixed(2);
        barMAR.style.width = Math.min(100, (mar / 0.8) * 100) + '%';

        updateChart(ear, mar);

        // Calculate Attention Score
        let attentionScore = 100;

        // Check Eye Closure (EAR)
        if (ear < earThreshold) {
            drowsyFrameCounter++;
            attentionScore -= Math.min(60, drowsyFrameCounter * 3);
        } else {
            // Blink detection logic (Quick closure & re-open)
            if (drowsyFrameCounter >= 1 && drowsyFrameCounter <= 5) {
                blinkCountSession++;
                valBpm.innerText = Math.round((blinkCountSession / Math.max(1, (Date.now() - sessionStartTime) / 60000)));
            }
            drowsyFrameCounter = Math.max(0, drowsyFrameCounter - 2);
        }

        // Check Yawning (MAR)
        if (mar > marThreshold) {
            yawnFrameCounter++;
            attentionScore -= 25;
        } else {
            yawnFrameCounter = Math.max(0, yawnFrameCounter - 1);
        }

        // Check Head Nodding
        if (headDroop) {
            headNodFrameCounter++;
            attentionScore -= 30;
        } else {
            headNodFrameCounter = Math.max(0, headNodFrameCounter - 1);
        }

        attentionScore = Math.max(0, Math.min(100, attentionScore));
        valAttention.innerText = `${Math.round(attentionScore)}%`;
        barAttention.style.width = `${attentionScore}%`;

        // Set bar color based on score
        if (attentionScore > 70) {
            barAttention.style.background = 'var(--success-green)';
            valAttention.style.color = 'var(--success-green)';
        } else if (attentionScore > 35) {
            barAttention.style.background = 'var(--warning-yellow)';
            valAttention.style.color = 'var(--warning-yellow)';
        } else {
            barAttention.style.background = 'var(--alert-red)';
            valAttention.style.color = 'var(--alert-red)';
        }

        // Frame duration threshold calculation (assumes ~25-30 fps)
        const triggerFrames = triggerDurationSec * 25;

        // Drowsiness Emergency Trigger
        if (drowsyFrameCounter >= triggerFrames || headNodFrameCounter >= triggerFrames * 0.8) {
            if (!isAlarmActive) {
                triggerEmergencyAlarm(ear, mar, 'Critical Drowsiness Detected');
            }
        } else if (yawnFrameCounter > 20) {
            if (!isYawnWarningActive && !isAlarmActive) {
                triggerYawnWarning(mar);
            }
        } else {
            // Reset to normal if no active conditions
            if (isAlarmActive && drowsyFrameCounter === 0 && headNodFrameCounter === 0) {
                clearEmergencyAlarm();
            } else if (isYawnWarningActive && yawnFrameCounter === 0) {
                clearYawnWarning();
            } else if (!isAlarmActive && !isYawnWarningActive) {
                updateStatus('DRIVER VIGILANT', 'normal');
            }
        }
    }

    // --- Emergency & Warning Trigger Handlers ---
    function triggerEmergencyAlarm(ear, mar, reason) {
        isAlarmActive = true;
        alarmStartTime = new Date();
        document.body.classList.add('alert-active');

        updateStatus('🚨 DROWSINESS ALARM!', 'alert');
        window.audioEngine.startAlarm();

        if (chkVoiceAlert.checked) {
            window.audioEngine.speakVoiceWarning("Drowsiness warning! Driver wake up! Pull over safely!");
        }

        logIncident('CRITICAL DROWSINESS', ear, mar, 'Alarm Sound & Voice Alert Triggered');
    }

    function clearEmergencyAlarm() {
        isAlarmActive = false;
        document.body.classList.remove('alert-active');
        window.audioEngine.stopAlarm();
        updateStatus('DRIVER VIGILANT', 'normal');
    }

    function triggerYawnWarning(mar) {
        isYawnWarningActive = true;
        updateStatus('⚠️ YAWN / FATIGUE DETECTED', 'warning');
        window.audioEngine.playShortBeep(650, 0.2);

        if (chkVoiceAlert.checked) {
            window.audioEngine.speakVoiceWarning("Fatigue warning. Frequent yawning detected.");
        }

        logIncident('YAWN WARNING', parseFloat(valEAR.innerText), mar, 'Fatigue Beep Warned');
    }

    function clearYawnWarning() {
        isYawnWarningActive = false;
        if (!isAlarmActive) {
            updateStatus('DRIVER VIGILANT', 'normal');
        }
    }

    function updateStatus(text, state) {
        statusText.innerText = text;
        systemStatusBadge.className = 'status-badge';
        if (state === 'warning') systemStatusBadge.classList.add('status-warning');
        else if (state === 'alert') systemStatusBadge.classList.add('status-alert');
    }

    // --- HUD Mesh Overlay Renderer ---
    function drawMeshOverlay(leftEye, rightEye, mouth, ear, mar, headDroop) {
        // Eye Color (Green if open, Red if closed)
        const eyeColor = ear < earThreshold ? '#ff0055' : '#00ff9d';

        // Draw Left Eye outline
        ctx.strokeStyle = eyeColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        leftEye.forEach((pt, i) => {
            if (i === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
        });
        ctx.closePath();
        ctx.stroke();

        // Draw Right Eye outline
        ctx.beginPath();
        rightEye.forEach((pt, i) => {
            if (i === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
        });
        ctx.closePath();
        ctx.stroke();

        // Draw Mouth outline
        ctx.strokeStyle = mar > marThreshold ? '#ffb700' : '#00f3ff';
        ctx.beginPath();
        ctx.moveTo(mouth.left.x, mouth.left.y);
        ctx.lineTo(mouth.top.x, mouth.top.y);
        ctx.lineTo(mouth.right.x, mouth.right.y);
        ctx.lineTo(mouth.bottom.x, mouth.bottom.y);
        ctx.closePath();
        ctx.stroke();

        // HUD Text Header on Video
        ctx.fillStyle = '#00f3ff';
        ctx.font = '14px Orbitron';
        ctx.fillText(`EAR: ${ear.toFixed(2)}`, 20, 30);
        ctx.fillText(`MAR: ${mar.toFixed(2)}`, 20, 50);

        if (headDroop) {
            ctx.fillStyle = '#ff0055';
            ctx.fillText('HEAD DROOP DETECTED', 20, 75);
        }
    }

    // --- Incident Logging & CSV Export ---
    function logIncident(type, ear, mar, action) {
        const timeStr = new Date().toLocaleTimeString();
        const durationStr = alarmStartTime ? `${((new Date() - alarmStartTime) / 1000).toFixed(1)}s` : '1.0s';

        const incident = { timestamp: timeStr, type, ear: ear.toFixed(2), mar: mar.toFixed(2), duration: durationStr, action };
        incidentLogs.push(incident);

        // Update Table UI
        renderLogTable();
    }

    function renderLogTable() {
        if (incidentLogs.length === 0) {
            logTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 1rem;">No fatigue incidents logged yet. Stay alert!</td></tr>`;
            return;
        }

        logTableBody.innerHTML = incidentLogs.map(log => {
            let typeClass = 'log-type-drowsy';
            if (log.type.includes('YAWN')) typeClass = 'log-type-yawn';
            else if (log.type.includes('NOD')) typeClass = 'log-type-nod';

            return `
                <tr>
                    <td>${log.timestamp}</td>
                    <td class="${typeClass}">${log.type}</td>
                    <td>${log.ear}</td>
                    <td>${log.mar}</td>
                    <td>${log.duration}</td>
                    <td>${log.action}</td>
                </tr>
            `;
        }).reverse().join('');
    }

    btnClearLog.addEventListener('click', () => {
        incidentLogs.length = 0;
        renderLogTable();
    });

    btnExportCsv.addEventListener('click', () => {
        if (incidentLogs.length === 0) {
            alert('No logs available to export.');
            return;
        }

        let csv = 'Timestamp,Event Type,EAR,MAR,Duration,Action Taken\n';
        incidentLogs.forEach(row => {
            csv += `"${row.timestamp}","${row.type}",${row.ear},${row.mar},"${row.duration}","${row.action}"\n`;
        });

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `drowsiness_log_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // --- Demo Mode Simulation Engine ---
    let demoSimMode = 'normal'; // 'normal', 'drowsy', 'yawn', 'nod'
    let demoFrameTick = 0;

    function startDemoSimulation() {
        isDemoMode = true;
        isCameraRunning = false;
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
        }
        videoWebcam.style.display = 'none';

        btnStartCamera.style.display = 'inline-flex';
        btnStopCamera.style.display = 'none';
        btnToggleDemo.innerHTML = '<span>🎬 Stop Demo Mode</span>';

        cancelAnimationFrame(animationFrameId);
        runDemoSimulationLoop();
    }

    function stopDemoSimulation() {
        isDemoMode = false;
        cancelAnimationFrame(animationFrameId);
        btnToggleDemo.innerHTML = '<span>🎬 Start Demo Mode</span>';
        ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
        clearEmergencyAlarm();
    }

    function runDemoSimulationLoop() {
        if (!isDemoMode) return;

        demoFrameTick++;

        canvasOverlay.width = 640;
        canvasOverlay.height = 480;

        ctx.fillStyle = '#070c18';
        ctx.fillRect(0, 0, 640, 480);

        // Draw Synthetic Avatar Head
        const cx = 320;
        let cy = 220;
        let ear = 0.32;
        let mar = 0.18;
        let isHeadDroop = false;

        // Apply scenario dynamics
        if (demoSimMode === 'normal') {
            // Periodic natural blink
            if (demoFrameTick % 90 < 8) ear = 0.12;
            else ear = 0.30 + Math.sin(demoFrameTick * 0.05) * 0.02;
        } else if (demoSimMode === 'drowsy') {
            // Eyes slowly closing and remaining closed
            ear = Math.max(0.08, 0.30 - (demoFrameTick % 120) * 0.003);
        } else if (demoSimMode === 'yawn') {
            // Mouth opening wide
            mar = 0.20 + Math.abs(Math.sin(demoFrameTick * 0.08)) * 0.55;
            ear = 0.22;
        } else if (demoSimMode === 'nod') {
            // Head drooping down
            cy = 220 + Math.min(80, (demoFrameTick % 100) * 1.5);
            if (cy > 270) isHeadDroop = true;
            ear = 0.15;
        }

        // Render Avatar Graphics
        ctx.fillStyle = '#1e293b';
        ctx.strokeStyle = '#00f3ff';
        ctx.lineWidth = 3;

        // Head Circle
        ctx.beginPath();
        ctx.arc(cx, cy, 100, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Eyes
        const eyeColor = ear < earThreshold ? '#ff0055' : '#00ff9d';
        ctx.fillStyle = eyeColor;

        const eyeHeight = Math.max(2, ear * 60);

        // Left Eye
        ctx.fillRect(cx - 45, cy - 20 - eyeHeight / 2, 30, eyeHeight);
        // Right Eye
        ctx.fillRect(cx + 15, cy - 20 - eyeHeight / 2, 30, eyeHeight);

        // Mouth
        const mouthHeight = Math.max(6, mar * 80);
        ctx.fillStyle = mar > marThreshold ? '#ffb700' : '#00f3ff';
        ctx.beginPath();
        ctx.ellipse(cx, cy + 45, 25, mouthHeight / 2, 0, 0, Math.PI * 2);
        ctx.fill();

        // HUD Watermark
        ctx.fillStyle = 'rgba(0, 243, 255, 0.6)';
        ctx.font = '16px Orbitron';
        ctx.fillText(`[DEMO MODE: ${demoSimMode.toUpperCase()}]`, 20, 40);

        // Process logic
        processFrameMetrics(ear, mar, isHeadDroop);

        animationFrameId = requestAnimationFrame(runDemoSimulationLoop);
    }

    // --- Webcam Controls ---
    btnStartCamera.addEventListener('click', async () => {
        if (isDemoMode) stopDemoSimulation();

        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' },
                audio: false
            });

            videoWebcam.srcObject = cameraStream;
            await videoWebcam.play();
            videoWebcam.style.display = 'block';

            isCameraRunning = true;
            btnStartCamera.style.display = 'none';
            btnStopCamera.style.display = 'inline-flex';

            // Start MediaPipe Frame Loop
            if (window.Camera && faceMesh) {
                mediaPipeCamera = new window.Camera(videoWebcam, {
                    onFrame: async () => {
                        if (isCameraRunning) {
                            await faceMesh.send({ image: videoWebcam });
                        }
                    },
                    width: 640,
                    height: 480
                });
                mediaPipeCamera.start();
            }

            updateStatus('CAMERA ACTIVE', 'normal');
        } catch (err) {
            console.error('Camera Access Error:', err);
            alert('Unable to access webcam. Switching to Demo Simulation Mode.');
            startDemoSimulation();
        }
    });

    btnStopCamera.addEventListener('click', () => {
        isCameraRunning = false;
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
            cameraStream = null;
        }
        if (mediaPipeCamera) {
            mediaPipeCamera.stop();
        }
        videoWebcam.style.display = 'none';
        btnStartCamera.style.display = 'inline-flex';
        btnStopCamera.style.display = 'none';
        clearEmergencyAlarm();
        updateStatus('SYSTEM READY', 'normal');
    });

    btnToggleDemo.addEventListener('click', () => {
        if (isDemoMode) {
            stopDemoSimulation();
        } else {
            startDemoSimulation();
        }
    });

    // Demo Scenario Switchers
    btnSimNormal.addEventListener('click', () => { demoSimMode = 'normal'; demoFrameTick = 0; if(!isDemoMode) startDemoSimulation(); });
    btnSimDrowsy.addEventListener('click', () => { demoSimMode = 'drowsy'; demoFrameTick = 0; if(!isDemoMode) startDemoSimulation(); });
    btnSimYawn.addEventListener('click', () => { demoSimMode = 'yawn'; demoFrameTick = 0; if(!isDemoMode) startDemoSimulation(); });
    btnSimNod.addEventListener('click', () => { demoSimMode = 'nod'; demoFrameTick = 0; if(!isDemoMode) startDemoSimulation(); });

    // --- Inputs & Event Listeners ---
    inputEarThresh.addEventListener('input', (e) => {
        earThreshold = parseFloat(e.target.value);
        txtEarThreshVal.innerText = earThreshold.toFixed(2);
        lblEarThresh.innerText = earThreshold.toFixed(2);
    });

    inputDuration.addEventListener('input', (e) => {
        triggerDurationSec = parseFloat(e.target.value);
        txtDurationVal.innerText = `${triggerDurationSec.toFixed(1)} s`;
    });

    inputMarThresh.addEventListener('input', (e) => {
        marThreshold = parseFloat(e.target.value);
        txtMarThreshVal.innerText = marThreshold.toFixed(2);
        lblMarThresh.innerText = marThreshold.toFixed(2);
    });

    selectSoundType.addEventListener('change', (e) => {
        window.audioEngine.setSoundType(e.target.value);
    });

    inputVolume.addEventListener('input', (e) => {
        const vol = parseFloat(e.target.value);
        txtVolumeVal.innerText = `${Math.round(vol * 100)}%`;
        window.audioEngine.setVolume(vol);
    });

    btnTestAlarm.addEventListener('click', () => {
        window.audioEngine.startAlarm();
        setTimeout(() => window.audioEngine.stopAlarm(), 1500);
    });

    btnTestVoice.addEventListener('click', () => {
        window.audioEngine.speakVoiceWarning("Test warning! Drowsiness detection alarm system operational.");
    });
});
