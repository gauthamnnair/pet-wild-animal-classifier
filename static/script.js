let stream = null;

async function toggleCamera() {
    const cameraStream = document.getElementById('camera-stream');
    const captureButton = document.getElementById('capture-button');
    const cameraButton = document.getElementById('camera-button');
    
    if (stream) {
        // Stop the camera
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        cameraStream.style.display = 'none';
        captureButton.style.display = 'none';
        cameraButton.textContent = 'Open Camera';
        return;
    }
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment'
            }
        });
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
    // Remove existing alert if any
    const existingAlert = document.querySelector('.alert-overlay');
    if (existingAlert) {
        existingAlert.remove();
    }

    // Create and append new alert
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

    // Play alert sound
    const alertSound = new Audio('static/wild_alret.wav');
    alertSound.play();

    // Auto-remove alert after 5 seconds
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

    // Draw the current frame from the video onto the canvas
    canvas.getContext('2d').drawImage(cameraStream, 0, 0);

    // Convert the canvas to a blob
    canvas.toBlob(blob => {
        // Create a File object from the blob
        const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });

        // Add to queue
        fileQueue.push({
            file: file,
            status: 'pending',
            type: 'image'
        });

        // Update queue display
        updateQueueDisplay();

        // Show preview
        const imgPreview = document.getElementById('uploaded-image');
        imgPreview.src = URL.createObjectURL(blob);
        imgPreview.style.display = 'block';
        document.getElementById('uploaded-video').style.display = 'none';

        // Set file type select to image
        document.querySelector('.file-select').value = 'image';

        // Close camera
        toggleCamera();
    }, 'image/jpeg');
}

let fileQueue = [];
let isProcessing = false;
let lastProcessedFile = null;

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
            fileQueue.push({
                file: file,
                status: 'pending',
                type: fileType
            });
        }
    }
    updateQueueDisplay();
    // Show file preview without highlighting animal type
    previewFile(event);
}

function updateQueueDisplay() {
    const queueList = document.getElementById('queue-list');
    console.log('Updating queue display:', fileQueue); // Debug log
    queueList.innerHTML = '';

    fileQueue.forEach((item, index) => {
        const queueItem = document.createElement('div');
        queueItem.className = `queue-item ${item.status}`;
        queueItem.innerHTML = `
            <span>${item.file.name} (${item.status})</span>
            <button class="remove-item" onclick="removeFromQueue(${index})">Remove</button>
        `;
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
    resetAnimalOptions(); // Reset animal options when queue is cleared
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

                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                console.log('Server response:', data);

                if (data.error) {
                    throw new Error(data.error);
                }

                // Update the preview with processed result
                if (item.type === 'image') {
                    document.getElementById('uploaded-image').src = data.processed_image;
                    document.getElementById('uploaded-image').style.display = 'block';
                    document.getElementById('uploaded-video').style.display = 'none';
                } else if (item.type === 'video') {
                    const videoElement = document.getElementById('uploaded-video');
                    const videoSource = document.getElementById('video-source');

                    if (data.processed_video_url) {
                        videoSource.src = data.processed_video_url;
                        videoElement.load();
                        videoElement.style.display = 'block';
                        document.getElementById('uploaded-image').style.display = 'none';
                    } else {
                        alert("No processed video received from the server.");
                    }
                }

                // Display animal details
                displayAnimalDetails(data.detections);

                // Handle alert for wild animals
                if (data.alert_required) {
                    showDangerAlert();
                }

                // Highlight animal type (pet or wild)
                if (data.dominant_type === 'pet') {
                    document.getElementById('pet-animal').classList.add('active');
                } else if (data.dominant_type === 'wild') {
                    document.getElementById('wild-animal').classList.add('active');
                }

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
    console.log('Displaying animal details:', detections); // Debug log
    const detailsElement = document.getElementById('animal-details');

    if (!detections || detections.length === 0) {
        detailsElement.innerHTML = `
            <div id="no-detection" style="text-align: center; color: #666;">
                No animals detected. Upload an image or video to detect animals.
            </div>
        `;
        return;
    }

    // Clear previous content
    detailsElement.innerHTML = '';

    // Display each detection
    detections.forEach((detection, index) => {
        const detectionElement = document.createElement('div');
        detectionElement.className = 'detection-card';

        const confidencePercent = (detection.confidence * 100).toFixed(1);
        const confidenceText = `${confidencePercent}% confident`;

        // Check if confidence is low
        const confidenceClass = detection.confidence < 0.5 ? 'low-confidence' : 'high-confidence';

        detectionElement.innerHTML = `
            <h3>
                ${detection.class.charAt(0).toUpperCase() + detection.class.slice(1)}
                <span class="confidence-score ${confidenceClass}">${confidenceText}</span>
            </h3>
            <div class="detection-info">
                <strong>Type:</strong> ${detection.type.charAt(0).toUpperCase() + detection.type.slice(1)}
            </div>
            <div class="detection-info">
                <strong>Description:</strong> ${detection.description || 'No description available'}
            </div>
            <div class="detection-info">
                <strong>Safety Tips:</strong> ${detection.safety_tips || 'No safety tips available'}
            </div>
        `;

        // Add a separator line between detection cards (except for the last one)
        if (index < detections.length - 1) {
            detectionElement.appendChild(document.createElement('hr')).className = 'detection-divider';
        }

        detailsElement.appendChild(detectionElement);
    });
}

function handleFileTypeChange(fileType) {
    const fileInput = document.getElementById('file-upload');
    if (fileType === 'image') {
        fileInput.accept = 'image/*';
    } else if (fileType === 'video') {
        fileInput.accept = 'video/*';
    } else {
        fileInput.accept = '';
    }
    document.getElementById('uploaded-image').style.display = 'none';
    document.getElementById('uploaded-video').style.display = 'none';
    resetAnimalOptions(); // Reset animal options when file type changes
}

function previewFile(event) {
    const file = event.target.files[0];
    const imgPreview = document.getElementById('uploaded-image');
    const videoPreview = document.getElementById('uploaded-video');
    const videoSource = document.getElementById('video-source');
    const fileType = document.querySelector('.file-select').value;

    resetAnimalOptions(); // Reset animal options when new file is previewed

    if (fileType === 'image' && file.type.startsWith('image/')) {
        imgPreview.src = URL.createObjectURL(file);
        imgPreview.style.display = 'block';
        videoPreview.style.display = 'none';
    } else if (fileType === 'video' && file.type.startsWith('video/')) {
        videoSource.src = URL.createObjectURL(file);
        videoPreview.style.display = 'block';
        imgPreview.style.display = 'none';
        videoPreview.load();
    }
}

function resetAnimalOptions() {
    document.querySelectorAll('.animal-option').forEach(el => {
        el.classList.remove('active');
    });
}
