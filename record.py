import cv2
import threading
import time
import datetime
import os
import uuid
import requests
from pathlib import Path

from motion import count_motion_area

ENV_MODE = os.getenv('ENV', 'dev').lower()
API_URL = os.getenv('API_URL', 'http://192.168.55.2:3000/api/upload')


def _parse_sources():
    """Źródła wideo, domyślnie są to kamery USB o id 0, 2, 4, można je zamienić na plik
      VIDEO_SOURCE="plik.mp4"
      VIDEO_SOURCE="0,2,4" (kamery)
    """
    raw = os.getenv('VIDEO_SOURCE', '').strip()
    if not raw:
        return [0, 2, 4]
    sources = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        sources.append(int(part) if part.isdigit() else part)
    return sources


def _safe_label(source):
    if isinstance(source, int):
        return str(source)
    base = os.path.splitext(os.path.basename(str(source)))[0]
    safe = ''.join(c if c.isalnum() else '_' for c in base)
    return safe or 'src'


CAM_INDEX = _parse_sources()

MIN_MOTION_AREA = 1500
TIME_AFTER_MOTION = 5
WARMUP_TIME = float(os.getenv('WARMUP_TIME', '10'))
IMAGE_FOLDER = "captures"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 10.0

SEND_QUEUE = []
QUEUE_LOCK = threading.Lock()
CURRENT_BATCH_UUID = None


def upload_worker(files_data):
    """
    Funkcja wysyła grupę plików (batch) na serwer Node.js
    """
    if not files_data:
        return

    print(f"\n[UPLOAD] Przygotowanie do wysłania {len(files_data)} plików na {API_URL}...")

    batch_uuid = files_data[0]['uuid']

    files_payload = []
    opened_files = []

    try:
        for item in files_data:
            path = item['file_name']
            if os.path.exists(path):
                f = open(path, 'rb')
                opened_files.append(f)
                filename = os.path.basename(path)
                # 'files' to nazwa oczekiwana przez API na Azurze
                files_payload.append(('files', (filename, f, 'image/jpeg')))
            else:
                print(f"[ERROR] Brak pliku lokalnego: {path}")

        if not files_payload:
            print("[UPLOAD] Brak plików do wysłania.")
            return

        payload_data = {'batchID': batch_uuid}

        try:
            response = requests.post(API_URL, files=files_payload, data=payload_data)

            if response.status_code == 200:
                print(f"[SUCCESS] Wysłano batch {batch_uuid}. Odpowiedź serwera: {response.text}")
            else:
                print(f"[ERROR] Serwer zwrócił błąd: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Nie można połączyć się z serwerem {API_URL}")
        except Exception as e:
            print(f"[ERROR] Błąd podczas wysyłania: {e}")

    finally:
        for f in opened_files:
            f.close()


class motion_detection(threading.Thread):
    def __init__(self, cam_id):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.running = True
        self.capturing_event = False
        self.last_motion_time = 0
        self.start_time = time.time()
        self.motion_frames_count = 0
        self.last_batch_uuid = None

        # Zmienne do "najlepszej klatki"
        self.best_frame = None
        self.max_motion_area = 0
        self.current_filename_placeholder = None
        self.cam_label = _safe_label(cam_id)

        Path(IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)

    def run(self):
        print(f"[CAM {self.cam_id}] Inicjalizacja wideo")
        if isinstance(self.cam_id, int):
            # Kamera USB (Linux / Raspberry Pi)
            cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, FPS)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        else:
            # Plik wideo (tryb headless / kontener / CI)
            cap = cv2.VideoCapture(self.cam_id)

        if not cap.isOpened():
            print(f"[CAM {self.cam_id}] Nie można zainicjalizować wideo")
            return

        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"[CAM {self.cam_id}] Koniec strumienia / błąd odczytu klatki")
                break

            small_frame = cv2.resize(frame, (320, 240))
            blurred = cv2.GaussianBlur(small_frame, (7, 7), 0)
            fgmask = fgbg.apply(blurred, learningRate=-1)
            fgmask = cv2.erode(fgmask, kernel_erode, iterations=2)
            fgmask = cv2.dilate(fgmask, kernel_dilate, iterations=2)

            if time.time() - self.start_time < WARMUP_TIME:
                continue

            motion_detected, current_total_area = count_motion_area(fgmask, MIN_MOTION_AREA)

            current_time = time.time()

            if not self.capturing_event:
                join_batch_uuid = None
                with QUEUE_LOCK:
                    if SEND_QUEUE and CURRENT_BATCH_UUID:
                        join_batch_uuid = CURRENT_BATCH_UUID

                if join_batch_uuid and join_batch_uuid != self.last_batch_uuid:
                    self.start_event(frame, current_total_area)
                    self.last_motion_time = current_time

            if motion_detected:
                self.motion_frames_count += 1
                self.last_motion_time = current_time
                if not self.capturing_event:
                    if self.motion_frames_count >= 5:
                        self.start_event(frame, current_total_area)
                else:
                    if current_total_area > self.max_motion_area:
                        self.max_motion_area = current_total_area
                        self.best_frame = frame.copy()
            else:
                if not self.capturing_event:
                    self.motion_frames_count = 0

            if self.capturing_event:
                if current_time - self.last_motion_time > TIME_AFTER_MOTION:
                    self.finish_event()

            time.sleep(0.01)

        self.finish_event()
        cap.release()

    def start_event(self, frame, initial_area):
        self.capturing_event = True
        self.max_motion_area = initial_area
        self.best_frame = frame.copy()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_filename_placeholder = f"{IMAGE_FOLDER}/cam{self.cam_label}_{timestamp}.jpg"

        global CURRENT_BATCH_UUID
        with QUEUE_LOCK:
            if not SEND_QUEUE:
                CURRENT_BATCH_UUID = str(uuid.uuid4())
                print(f"--- NOWY EVENT GRUPOWY: {CURRENT_BATCH_UUID} ---")

            self.last_batch_uuid = CURRENT_BATCH_UUID

            SEND_QUEUE.append({
                'file_name': self.current_filename_placeholder,
                'active': True,
                'cam_id': self.cam_id,
                'uuid': CURRENT_BATCH_UUID
            })

        print(f"[CAM {self.cam_id}] Wykryto ruch -> Start analizy")

    def finish_event(self):
        if not self.capturing_event:
            return

        self.capturing_event = False

        if self.best_frame is not None and self.current_filename_placeholder:
            cv2.imwrite(self.current_filename_placeholder, self.best_frame)
            print(f"[CAM {self.cam_id}] Zapisano: {self.current_filename_placeholder}")

        self.best_frame = None
        self.max_motion_area = 0
        self.motion_frames_count = 0

        files_to_upload = []
        start_upload = False

        with QUEUE_LOCK:
            for item in SEND_QUEUE:
                if item['file_name'] == self.current_filename_placeholder:
                    item['active'] = False
                    break

            self.current_filename_placeholder = None
            is_anyone_active = any(item['active'] for item in SEND_QUEUE)

            if not is_anyone_active and SEND_QUEUE:
                print("Koniec zdarzenia. Wysyłanie do API...")
                files_to_upload = list(SEND_QUEUE)
                SEND_QUEUE.clear()
                start_upload = True

        if start_upload:
            t_upload = threading.Thread(target=upload_worker, args=(files_to_upload,))
            t_upload.start()

    def stop(self):
        self.running = False


if __name__ == "__main__":
    print(f"API URL: {API_URL}")

    threads = []
    for id in CAM_INDEX:
        t = motion_detection(id)
        t.start()
        threads.append(t)

    if all(isinstance(s, int) for s in CAM_INDEX): # Dla kamer USB
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            for t in threads:
                t.stop()
            for t in threads:
                t.join()
    else:
        # Dla nagranego pliku
        for t in threads:
            t.join()
