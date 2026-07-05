document.addEventListener('DOMContentLoaded', () => {

    // ── Mode switching ───────────────────────────────────────────────
    const tabAslToText  = document.getElementById('tab-asl-to-text');
    const tabTextToAsl  = document.getElementById('tab-text-to-asl');
    const modeAslToText = document.getElementById('mode-asl-to-text');
    const modeTextToAsl = document.getElementById('mode-text-to-asl');

    tabAslToText?.addEventListener('click', () => {
        tabAslToText.classList.add('active');    tabTextToAsl.classList.remove('active');
        modeAslToText.classList.add('active');   modeTextToAsl.classList.remove('active');
    });
    tabTextToAsl?.addEventListener('click', () => {
        tabTextToAsl.classList.add('active');    tabAslToText.classList.remove('active');
        modeTextToAsl.classList.add('active');   modeAslToText.classList.remove('active');
    });

    // ── ASL → Text elements ─────────────────────────────────────────
    const btnStartCamera    = document.getElementById('btn-start-camera');
    const videoFeed         = document.getElementById('webcam-feed');
    const videoPlaceholder  = document.getElementById('video-placeholder');
    const cameraStatus      = document.getElementById('camera-status');
    const captureCanvas     = document.getElementById('capture-canvas');
    const landmarkCanvas    = document.getElementById('landmark-canvas');
    const predictionOverlay = document.getElementById('prediction-overlay');
    const predictionWord    = document.getElementById('prediction-word');
    const bufferBar         = document.getElementById('buffer-bar');
    const confidencePct     = document.getElementById('confidence-pct');
    const translatedText    = document.getElementById('translated-text');
    const transcriptStrip   = document.getElementById('transcript-strip');

    let stream          = null;
    let captureTimeout  = null; // Changed from interval to timeout tracker
    let lastWord        = null;
    let lastWordTime    = 0;
    const WORD_GAP_MS   = 2500;   // silence gap → space in transcript
    const CONF_THRESHOLD = 0.40;  // minimum confidence to accept

    // ── Landmark drawing ────────────────────────────────────────────
    const lmCtx = landmarkCanvas.getContext('2d');

    function resizeLandmarkCanvas() {
        landmarkCanvas.width  = landmarkCanvas.offsetWidth;
        landmarkCanvas.height = landmarkCanvas.offsetHeight;
    }

    function drawLandmarks(landmarks) {
        const W = landmarkCanvas.width;
        const H = landmarkCanvas.height;
        lmCtx.clearRect(0, 0, W, H);

        function dot(pt, color, r = 4) {
            // x is mirrored on the canvas via CSS scaleX(-1) so we draw raw coords
            lmCtx.beginPath();
            lmCtx.arc(pt.x * W, pt.y * H, r, 0, Math.PI * 2);
            lmCtx.fillStyle = color;
            lmCtx.fill();
        }

        function line(a, b, color) {
            lmCtx.beginPath();
            lmCtx.moveTo(a.x * W, a.y * H);
            lmCtx.lineTo(b.x * W, b.y * H);
            lmCtx.strokeStyle = color;
            lmCtx.lineWidth = 2;
            lmCtx.stroke();
        }

        // Pose skeleton (7 selected points)
        const pose = landmarks.pose;
        if (pose.length === 7) {
            // Draw connections: shoulder-shoulder, shoulder-elbow, elbow-wrist
            // Indices in POSE_IDS [0,11,12,13,14,15,16]:
            //   0=nose, 1=L-shoulder, 2=R-shoulder, 3=L-elbow, 4=R-elbow, 5=L-wrist, 6=R-wrist
            const conn = [[1,2],[1,3],[3,5],[2,4],[4,6],[0,1],[0,2]];
            conn.forEach(([a, b]) => { if (pose[a] && pose[b]) line(pose[a], pose[b], '#60a5fa'); });
            pose.forEach(p => dot(p, '#93c5fd', 5));
        }

        // Hands — draw all 21 keypoints + finger connections
        const FINGER_SEGS = [
            [0,1],[1,2],[2,3],[3,4],       // thumb
            [0,5],[5,6],[6,7],[7,8],       // index
            [0,9],[9,10],[10,11],[11,12],  // middle
            [0,13],[13,14],[14,15],[15,16],// ring
            [0,17],[17,18],[18,19],[19,20] // pinky
        ];

        function drawHand(pts, dotColor, lineColor) {
            if (!pts || pts.length < 21) return;
            FINGER_SEGS.forEach(([a, b]) => line(pts[a], pts[b], lineColor));
            pts.forEach(p => dot(p, dotColor, 4));
        }

        drawHand(landmarks.left_hand,  '#f472b6', '#ec4899');
        drawHand(landmarks.right_hand, '#34d399', '#10b981');
    }

    // ── Camera start/stop ────────────────────────────────────────────
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });
            videoFeed.srcObject = stream;
            videoFeed.style.display = 'block';
            videoPlaceholder.style.display = 'none';
            predictionOverlay.classList.add('active');
            btnStartCamera.querySelector('span').textContent = 'Stop Camera';
            cameraStatus?.classList.add('live');

            await new Promise(r => videoFeed.addEventListener('loadedmetadata', r, { once: true }));
            resizeLandmarkCanvas();
            new ResizeObserver(resizeLandmarkCanvas).observe(landmarkCanvas);

            captureCanvas.width  = 640;
            captureCanvas.height = 480;

            // Start the recursive execution loop targeting 30 FPS (33ms)
            captureTimeout = setTimeout(captureAndSendLoop, 33);
        } catch (err) {
            console.error('Camera error:', err);
            alert('Could not access camera. Please allow camera permissions in your browser settings, then reload.');
        }
    }

    function stopCamera() {
        stream?.getTracks().forEach(t => t.stop());
        stream = null;
        
        // Safely clear the active timeout loop
        if (captureTimeout) {
            clearTimeout(captureTimeout);
            captureTimeout = null;
        }

        videoFeed.style.display = 'none';
        videoFeed.srcObject = null;
        videoPlaceholder.style.display = 'flex';
        predictionOverlay.classList.remove('active');
        lmCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
        bufferBar.style.width = '0%';
        predictionWord.textContent = '—';
        predictionWord.classList.remove('has-word');
        confidencePct.textContent = '—';
        btnStartCamera.querySelector('span').textContent = 'Start Camera';
        cameraStatus?.classList.remove('live');
        fetch('/api/reset', { method: 'POST' });
    }

    btnStartCamera?.addEventListener('click', () => {
        stream ? stopCamera() : startCamera();
    });

    // ── Frame capture + predict (Fixed Loop for Stable 30 FPS) ──────
    async function captureAndSendLoop() {
        // Kill switch if camera was disabled during processing flight
        if (!stream || !videoFeed.videoWidth) return;

        const ctx = captureCanvas.getContext('2d');
        // Draw un-mirrored frame (server needs natural orientation)
        ctx.drawImage(videoFeed, 0, 0, 640, 480);
        const b64 = captureCanvas.toDataURL('image/jpeg', 0.40);

        try {
            const res  = await fetch('/api/predict', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ image: b64 }),
            });
            
            if (res.ok) {
                const data = await res.json();

                // Buffer progress bar
                const fill = (data.buffer_fill || 0) / (data.buffer_max || 30);
                bufferBar.style.width = `${Math.round(fill * 100)}%`;

                // Draw landmarks
                if (data.landmarks) drawLandmarks(data.landmarks);

                // Update prediction display
                if (data.prediction && data.confidence >= CONF_THRESHOLD) {
                    const word = data.prediction;
                    const conf = Math.round(data.confidence * 100);

                    predictionWord.textContent = word.toUpperCase();
                    predictionWord.classList.add('has-word');
                    confidencePct.textContent  = `${conf}%`;

                    // Append to transcript
                    const now = Date.now();
                    if (data.is_new && word !== lastWord) {
                        if (lastWordTime > 0 && (now - lastWordTime) > WORD_GAP_MS) {
                            transcriptStrip.textContent += ' ';
                        }
                        transcriptStrip.textContent += word + ' ';
                        translatedText.textContent   = transcriptStrip.textContent;
                        lastWord     = word;
                        lastWordTime = now;
                    }
                } else if (!data.prediction) {
                    predictionWord.textContent = '…';
                    predictionWord.classList.remove('has-word');
                    confidencePct.textContent  = '—';
                }
            }
        } catch (err) {
            console.error('Predict error:', err);
        }

        // Re-schedule loop iteration ONLY after current HTTP request lifecycle completes
        if (stream) {
            captureTimeout = setTimeout(captureAndSendLoop, 33);
        }
    }

    // ── Reset buffer button ──────────────────────────────────────────
    document.getElementById('btn-reset-buffer')?.addEventListener('click', () => {
        fetch('/api/reset', { method: 'POST' });
        bufferBar.style.width = '0%';
    });

    // ── Speak output ─────────────────────────────────────────────────
    document.getElementById('btn-speak-output')?.addEventListener('click', () => {
        const text = translatedText.textContent.trim();
        if (text) window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    });

    // ── Clear output ─────────────────────────────────────────────────
    document.getElementById('btn-clear-output')?.addEventListener('click', () => {
        translatedText.textContent  = '';
        transcriptStrip.textContent = '';
        predictionWord.textContent  = '—';
        predictionWord.classList.remove('has-word');
        confidencePct.textContent   = '—';
        lastWord     = null;
        lastWordTime = 0;
        fetch('/api/reset', { method: 'POST' });
    });

    // ── Text → ASL mode ──────────────────────────────────────────────
    let recognition = null;
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SR();
        recognition.continuous      = true;
        recognition.interimResults  = true;
        recognition.lang            = 'en-US';
        recognition.onresult = e => {
            for (let i = e.resultIndex; i < e.results.length; i++) {
                if (e.results[i].isFinal) {
                    document.getElementById('text-input').value +=
                        (document.getElementById('text-input').value ? ' ' : '') +
                        e.results[i][0].transcript.trim();
                }
            }
        };
        recognition.onend = () => document.getElementById('btn-mic')?.classList.remove('recording');
    }

    document.getElementById('btn-mic')?.addEventListener('click', () => {
        const btn = document.getElementById('btn-mic');
        if (!recognition) { alert('Speech recognition not supported.'); return; }
        if (btn.classList.contains('recording')) { recognition.stop(); }
        else { recognition.start(); btn.classList.add('recording'); }
    });

    document.getElementById('btn-translate-text')?.addEventListener('click', async () => {
        const text = document.getElementById('text-input').value.trim();
        if (!text) return;
        const res  = await fetch('/api/text-to-asl', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        const data = await res.json();
        const display = document.getElementById('asl-signs-display');
        document.getElementById('asl-placeholder').style.display = 'none';
        display.innerHTML = '';
        data.signs.forEach(sign => {
            if (sign.image_url) {
                const card = document.createElement('div');
                card.className = 'asl-sign-card';
                const img = document.createElement('img');
                img.src = sign.image_url; img.alt = sign.letter;
                const lbl = document.createElement('span'); lbl.textContent = sign.letter;
                card.append(img, lbl); display.appendChild(card);
            } else if (sign.letter === ' ') {
                const sp = document.createElement('div'); sp.style.width = '20px';
                display.appendChild(sp);
            }
        });
    });

    document.getElementById('btn-clear-input')?.addEventListener('click', () => {
        document.getElementById('text-input').value = '';
        document.getElementById('asl-signs-display').innerHTML = '';
        document.getElementById('asl-placeholder').style.display = 'flex';
    });
});