import cv2
import torch
import numpy as np
from queue import Queue
from threading import Thread
import time

class WebcamStream:
    def __init__(self, src=0, buffer_size=2):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        self.queue = Queue(maxsize=buffer_size)
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stop()
                    break
                if self.queue.empty():  # Only add if queue is empty
                    self.queue.put(frame)
            else:
                time.sleep(0.001)  # Prevent CPU overload

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.stream.release()

class ObjectDetector:
    def __init__(self, model_path='yolov5s.pt', conf_thresh=0.25):
        # Load YOLO model optimized for inference
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path=model_path)
        self.model.conf = conf_thresh
        
        # Optimize model for inference
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)
        
        # Enable TensorRT optimization if available
        if torch.cuda.is_available():
            try:
                from torch2trt import TRTModule
                self.model = TRTModule(self.model, fp16_mode=True)
            except ImportError:
                pass
                
        # Enable hardware acceleration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def preprocess(self, frame):
        # Resize and normalize frame
        frame = cv2.resize(frame, (640, 640))
        frame = frame.transpose((2, 0, 1))
        frame = torch.from_numpy(frame).float().div(255.0)
        frame = frame.unsqueeze(0)
        return frame.to(self.device)
        
    @torch.no_grad()  # Disable gradient calculation for inference
    def detect(self, frame):
        # Run inference
        results = self.model(frame)
        return results.xyxy[0]  # Returns (x1, y1, x2, y2, confidence, class)

def main():
    # Initialize webcam stream with minimal buffer
    stream = WebcamStream(src=0, buffer_size=2).start()
    detector = ObjectDetector()
    
    # Create display window
    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    
    # Performance monitoring
    frame_times = []
    
    try:
        while True:
            start_time = time.time()
            
            # Get frame from queue
            frame = stream.read()
            
            # Skip processing if we're falling behind
            if stream.queue.qsize() > 1:
                continue
            
            # Preprocess frame
            input_tensor = detector.preprocess(frame)
            
            # Run detection
            detections = detector.detect(input_tensor)
            
            # Draw results
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                if conf > detector.model.conf:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                (0, 255, 0), 2)
            
            # Calculate and display FPS
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times))
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Object Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()