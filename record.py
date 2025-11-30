import cv2
import threading
import time
import datetime
import os
import uuid
from pathlib import Path

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:
    BlobServiceClient = None

ENV_MODE = os.getenv('ENV', 'dev').lower()
AZURE_CONN_STR = os.getenv('AZURE_CONNECT_STR', '') 
AZURE_CONTAINER = "monitoring-videos"

CAM_INDEX = [0, 2, 4]

MIN_MOTION_AREA = 500 
TIME_AFTER_MOTION = 10
WARMUP_TIME = 10
VIDEO_FOLDER = "recordings"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 10.0 

SEND_QUEUE = []
QUEUE_LOCK = threading.Lock()
CURRENT_BATCH_UUID = None

def upload_worker(files):
    print(f"\nWysyłanie {len(files)} plików...")
    
    blob_service_client = None
    if AZURE_CONN_STR and BlobServiceClient:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        except Exception as e:
            print(e)

    for file in files:
        local_path = file['file_name']
        batch_uuid = file['uuid']
        file_name = os.path.basename(local_path)
        azure_blob_name = f"{batch_uuid}/{file_name}"
        
        if blob_service_client:
            try:
                if os.path.exists(local_path):
                    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER, blob=azure_blob_name)
                    with open(local_path, "rb") as data:
                        blob_client.upload_blob(data)
                    print(f"Wysłano: {azure_blob_name}")
                    # os.remove(local_path)
                else:
                    print(f"Brak pliku: {local_path}")
            except Exception as e:
                print(f"{e}")
        else:
            print(f"Wysłano: {azure_blob_name}")
            time.sleep(0.5)

    print(f"Wysłano batch: {CURRENT_BATCH_UUID}\n")


class motion_detection(threading.Thread):
    def __init__(self, cam_id):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.running = True
        self.recording = False
        self.last_motion_time = 0
        self.writer = None
        self.current_filename = None
        self.start_time = time.time()
        
        Path(VIDEO_FOLDER).mkdir(parents=True, exist_ok=True)

    def run(self):
        print(f"[CAM {self.cam_id}] Inicjalizacja...")
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        if not cap.isOpened():
            print(f"[CAM {self.cam_id}] Nie można zainicjalizować kamery")
            return

        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        while self.running:
            ret, frame = cap.read()
            if not ret: break

            small_frame = cv2.resize(frame, (320, 240))
            blurred = cv2.GaussianBlur(small_frame, (7, 7), 0)
            fgmask = fgbg.apply(blurred, learningRate=-1)
            fgmask = cv2.erode(fgmask, kernel_erode, iterations=1)
            fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)
            
            if time.time() - self.start_time < WARMUP_TIME:
                continue

            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < MIN_MOTION_AREA:
                    continue
                motion_detected = True
                break

            current_time = time.time()

            if motion_detected:
                self.last_motion_time = current_time
                if not self.recording:
                    self.start_recording(frame)
            
            if self.recording:
                if self.writer: 
                    self.writer.write(frame)
                
                if current_time - self.last_motion_time > TIME_AFTER_MOTION:
                    self.stop_recording()

        self.stop_recording()
        cap.release()

    def start_recording(self, frame):
        self.recording = True
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        height, width, _ = frame.shape
        ext = "mkv" if ENV_MODE == 'prod' else "avi"
        self.current_filename = f"{VIDEO_FOLDER}/cam{self.cam_id}_{timestamp}.{ext}"
        
        global CURRENT_BATCH_UUID
        with QUEUE_LOCK:
            if not SEND_QUEUE:
                CURRENT_BATCH_UUID = str(uuid.uuid4())
                print(f"Nowe zdarzenie: {CURRENT_BATCH_UUID}")
            
            my_uuid = CURRENT_BATCH_UUID
            SEND_QUEUE.append({
                'file_name': self.current_filename,
                'active': True,
                'cam_id': self.cam_id,
                'uuid': my_uuid
            })

        print(f"[CAM {self.cam_id}] Rozpoczęto nagrywanie")

        if ENV_MODE == 'prod':
            gst_pipeline = (
                f"appsrc ! videoconvert ! "
                f"video/x-raw,format=I420,width={width},height={height},framerate={int(FPS)}/1 ! "
                f"v4l2h264enc ! video/x-h264,level=(string)3.1,profile=high ! " 
                f"h264parse ! matroskamux ! filesink location={self.current_filename}"
            )
            try:
                self.writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, FPS, (width, height), True)
            except Exception as e:
                print(e)
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.current_filename, fourcc, FPS, (width, height))

    def stop_recording(self):
        if not self.recording: return
        
        self.recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
        print(f"[CAM {self.cam_id}] Zatrzymano nagrywanie")

        files = []
        start_upload = False

        with QUEUE_LOCK:
            for item in SEND_QUEUE:
                if item['file_name'] == self.current_filename:
                    item['active'] = False
                    break
            
            is_active = any(item['active'] for item in SEND_QUEUE)

            if not is_active and SEND_QUEUE:
                print(f"Koniec zdarzenia, wysyłanie plików")
                files = list(SEND_QUEUE)
                SEND_QUEUE.clear()
                start_upload = True

        if start_upload:
            t_upload = threading.Thread(target=upload_worker, args=(files,))
            t_upload.start()

    def stop(self):
        self.running = False

if __name__ == "__main__":
    if ENV_MODE == 'prod' and not AZURE_CONN_STR:
        print("Nie ustawiono klucza API do azure")

    threads = []
    for id in CAM_INDEX:
        t = motion_detection(id)
        t.start()
        threads.append(t)

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        for t in threads: t.stop()
        for t in threads: t.join()