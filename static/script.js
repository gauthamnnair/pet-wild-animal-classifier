let stream = null;

async function toggleCamera() {
    const cameraStream = document.getElementById('camera-stream');
    const captureButton = document.getElementById('capture-button');
    const cameraButton = document.getElementById('camera-button');
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        cameraStream.style.display = 'none';
        captureButton.style.display = 'none';
        cameraButton.textContent = 'Open Camera';
        return;
    }
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        cameraStream.srcObject = stream;
        cameraStream.style.display = 'block';
        captureButton.style.display = 'block';
        cameraButton.textContent = 'Close Camera';
    } catch (err) {
        console.error('Error accessing camera:', err);
        alert('Error accessing camera. Please make sure you have granted camera permissions.');
    }
}

function showDangerAlert() {
    const existingAlert = document.querySelector('.alert-overlay');
    if (existingAlert) existingAlert.remove();

    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert-overlay';
    alertDiv.innerHTML = `
        <div class="alert-icon">⚠️</div>
        <div class="alert-content">
            <div class="alert-title">WILD ANIMAL DETECTED!</div>
            <div class="alert-message">Exercise caution! A wild animal has been identified in the detection area.</div>
        </div>
    `;
    document.body.appendChild(alertDiv);
    const alertSound = new Audio('static/wild_alert.wav');
    alertSound.play();
    setTimeout(() => {
        alertDiv.style.right = '-400px';
        setTimeout(() => alertDiv.remove(), 500);
    }, 5000);
}

function captureImage() {
    const cameraStream = document.getElementById('camera-stream');
    const canvas = document.createElement('canvas');
    canvas.width = cameraStream.videoWidth;
    canvas.height = cameraStream.videoHeight;
    canvas.getContext('2d').drawImage(cameraStream, 0, 0);
    canvas.toBlob(blob => {
        const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
        fileQueue.push({ file, status: 'pending', type: 'image' });
        updateQueueDisplay();
        const imgPreview = document.getElementById('uploaded-image');
        imgPreview.src = URL.createObjectURL(blob);
        imgPreview.style.display = 'block';
        document.getElementById('uploaded-video').style.display = 'none';
        document.querySelector('.file-select').value = 'image';
        toggleCamera();
    }, 'image/jpeg');
}

let fileQueue = [];
let isProcessing = false;

function addToQueue(event) {
    const files = event.target.files;
    const fileType = document.querySelector('.file-select').value;
    if (!fileType) {
        alert('Please select a file type first');
        return;
    }
    for (let file of files) {
        if ((fileType === 'image' && file.type.startsWith('image/')) ||
            (fileType === 'video' && file.type.startsWith('video/'))) {
            fileQueue.push({ file, status: 'pending', type: fileType });
        }
    }
    updateQueueDisplay();
    previewFile(event);
}

function updateQueueDisplay() {
    const queueList = document.getElementById('queue-list');
    queueList.innerHTML = '';
    fileQueue.forEach((item, index) => {
        const queueItem = document.createElement('div');
        queueItem.className = `queue-item ${item.status}`;
        queueItem.innerHTML = `<span>${item.file.name} (${item.status})</span>
            <button class="remove-item" onclick="removeFromQueue(${index})">Remove</button>`;
        queueList.appendChild(queueItem);
    });
}

function removeFromQueue(index) {
    fileQueue.splice(index, 1);
    updateQueueDisplay();
}

function clearQueue() {
    fileQueue = [];
    updateQueueDisplay();
    resetAnimalOptions();
}

async function processQueue() {
    if (isProcessing || fileQueue.length === 0) return;
    isProcessing = true;
    for (let item of fileQueue) {
        if (item.status === 'pending') {
            item.status = 'processing';
            updateQueueDisplay();
            resetAnimalOptions();
            try {
                const formData = new FormData();
                formData.append('file', item.file);
                formData.append('type', item.type);
                const response = await fetch('/detect', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                displayAnimalDetails(data.detections);
                if (data.alert_required) showDangerAlert();
                item.status = 'completed';
            } catch (error) {
                console.error('Error processing file:', error);
                item.status = 'error';
            }
            updateQueueDisplay();
        }
    }
    isProcessing = false;
}

function displayAnimalDetails(detections) {
    const detailsElement = document.getElementById('animal-details');
    detailsElement.innerHTML = detections.length ? detections.map(d => `
        <div class="detection-card">
            <h3>${d.class} <span class="confidence-score">${(d.confidence * 100).toFixed(1)}%</span></h3>
            <div><strong>Type:</strong> ${d.type}</div>
            <div><strong>Description:</strong> ${d.description || 'No description available'}</div>
            <div><strong>Safety Tips:</strong> ${d.safety_tips || 'No safety tips available'}</div>
        </div>`).join('') : '<div>No animals detected.</div>';
}

function handleFileTypeChange(fileType) {
    document.getElementById('file-upload').accept = fileType === 'image' ? 'image/*' : fileType === 'video' ? 'video/*' : '';
    document.getElementById('uploaded-image').style.display = 'none';
    document.getElementById('uploaded-video').style.display = 'none';
    resetAnimalOptions();
}

function previewFile(event) {
    const file = event.target.files[0];
    const imgPreview = document.getElementById('uploaded-image');
    const videoPreview = document.getElementById('uploaded-video');
    if (file.type.startsWith('image/')) {
        imgPreview.src = URL.createObjectURL(file);
        imgPreview.style.display = 'block';
        videoPreview.style.display = 'none';
    } else {
        document.getElementById('video-source').src = URL.createObjectURL(file);
        videoPreview.style.display = 'block';
        imgPreview.style.display = 'none';
        videoPreview.load();
    }
}

function resetAnimalOptions() {
    document.querySelectorAll('.animal-option').forEach(el => el.classList.remove('active'));
}
