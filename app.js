class YOLODetector {
    constructor() {
        this.session = null;
        this.isModelLoaded = false;
        this.isDetecting = false;
        
        // COCO class names (adjust if your model uses different classes)
        this.classNames = [
            'controller', 'ac_adaptor', 'bat_charger', 'battery'
        ];
        
        this.colors = this.generateColors(this.classNames.length);
        this.initializeElements();
        this.loadModel();
    }
    
    initializeElements() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.status = document.getElementById('status');
        this.fpsDisplay = document.getElementById('fps');
        this.detectionsDisplay = document.getElementById('detections');
        
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        
        this.lastTime = performance.now();
        this.frameCount = 0;
    }
    
    generateColors(numColors) {
        const colors = [];
        for (let i = 0; i < numColors; i++) {
            const hue = (i * 137.508) % 360; // Golden angle approximation
            colors.push(`hsl(${hue}, 70%, 50%)`);
        }
        return colors;
    }
    
    async loadModel() {
        try {
            this.status.textContent = 'Loading YOLO model...';
            
            // Configure ONNX Runtime for better performance
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
            ort.env.wasm.numThreads = 1; // Single thread for stability
            
            this.session = await ort.InferenceSession.create('./best.onnx', {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            
            this.isModelLoaded = true;
            this.status.textContent = 'Model loaded successfully! Click "Start Camera" to begin.';
            this.startBtn.disabled = false;
            
        } catch (error) {
            console.error('Error loading model:', error);
            this.status.textContent = 'Error loading model. Make sure best.onnx is in the same folder.';
        }
    }
    
    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'environment' // Use back camera on mobile
                }
            });
            
            this.video.srcObject = stream;
            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.startDetection();
            };
            
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.status.textContent = 'Camera started. Detecting objects...';
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.status.textContent = 'Error accessing camera. Please allow camera permissions.';
        }
    }
    
    stopCamera() {
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        
        this.isDetecting = false;
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.status.textContent = 'Camera stopped.';
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    startDetection() {
        this.isDetecting = true;
        this.detectFrame();
    }
    
    async detectFrame() {
        if (!this.isDetecting || !this.isModelLoaded) return;
        
        try {
            const currentTime = performance.now();
            
            // Preprocess image
            const tensor = await this.preprocessImage();
            
            // Run inference
            const results = await this.session.run({ images: tensor });
            const output = results.output0.data;
            
            // Process detections
            const detections = this.processDetections(output);
            
            // Draw results
            this.drawDetections(detections);
            
            // Update FPS
            this.frameCount++;
            if (currentTime - this.lastTime >= 1000) {
                const fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime));
                this.fpsDisplay.textContent = `FPS: ${fps}`;
                this.frameCount = 0;
                this.lastTime = currentTime;
            }
            
            // Update detections count
            this.detectionsDisplay.textContent = `Detections: ${detections.length}`;
            
        } catch (error) {
            console.error('Detection error:', error);
        }
        
        // Continue detection loop
        requestAnimationFrame(() => this.detectFrame());
    }
    
    async preprocessImage() {
        // Create a temporary canvas for preprocessing
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        // Set canvas size to model input size (640x640 for YOLOv8)
        tempCanvas.width = 640;
        tempCanvas.height = 640;
        
        // Draw and resize video frame
        tempCtx.drawImage(this.video, 0, 0, 640, 640);
        
        // Get image data
        const imageData = tempCtx.getImageData(0, 0, 640, 640);
        const data = imageData.data;
        
        // Convert to RGB and normalize (0-1)
        const input = new Float32Array(3 * 640 * 640);
        for (let i = 0; i < data.length; i += 4) {
            const pixelIndex = i / 4;
            input[pixelIndex] = data[i] / 255.0; // R
            input[pixelIndex + 640 * 640] = data[i + 1] / 255.0; // G
            input[pixelIndex + 640 * 640 * 2] = data[i + 2] / 255.0; // B
        }
        
        return new ort.Tensor('float32', input, [1, 3, 640, 640]);
    }
    
   processDetections(output) {
    const detections = [];
    const numDetections = 8400; // Your model outputs 8400 detections
    const numClasses = 8; // Your model has 8 classes (not 80 like COCO)
    
    for (let i = 0; i < numDetections; i++) {
        // YOLOv8 output format: [x, y, w, h, class1_conf, class2_conf, ...]
        const x = output[i];                    // center_x
        const y = output[i + numDetections];    // center_y  
        const w = output[i + numDetections * 2]; // width
        const h = output[i + numDetections * 3]; // height
        
        // Find class with highest confidence
        let maxScore = 0;
        let classId = 0;
        for (let j = 0; j < numClasses; j++) {
            const score = output[i + numDetections * (4 + j)];
            if (score > maxScore) {
                maxScore = score;
                classId = j;
            }
        }
        
        // Filter by confidence threshold
        if (maxScore > 0.5) {
            // Convert to corner coordinates and scale to canvas size
            const scaleX = this.canvas.width / 640;
            const scaleY = this.canvas.height / 640;
            
            const boxX = (x - w / 2) * scaleX;
            const boxY = (y - h / 2) * scaleY;
            const boxW = w * scaleX;
            const boxH = h * scaleY;
            
            detections.push({
                x: boxX,
                y: boxY,
                width: boxW,
                height: boxH,
                confidence: maxScore,
                classId: classId,
                className: `Class ${classId}` // Update with your actual class names
            });
        }
    }
    
    return detections;
}
    
    drawDetections(detections) {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw each detection
        detections.forEach(detection => {
            const color = this.colors[detection.classId % this.colors.length];
            
            // Draw bounding box
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(detection.x, detection.y, detection.width, detection.height);
            
            // Draw label background
            const label = `${detection.className} ${(detection.confidence * 100).toFixed(1)}%`;
            this.ctx.font = '14px Arial';
            const textWidth = this.ctx.measureText(label).width;
            
            this.ctx.fillStyle = color;
            this.ctx.fillRect(detection.x, detection.y - 25, textWidth + 10, 20);
            
            // Draw label text
            this.ctx.fillStyle = 'white';
            this.ctx.fillText(label, detection.x + 5, detection.y - 10);
        });
    }
}

// Initialize the detector when page loads
document.addEventListener('DOMContentLoaded', () => {
    new YOLODetector();
});
