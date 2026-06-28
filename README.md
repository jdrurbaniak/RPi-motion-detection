# Obsługa kamer

Ta część projektu trójkamerowego systemu fotopułapkowego odpowiada za wykrywanie ruchu z trzech kamer na Raspberry Pi i wysyłaniu obrazu z nich przez API do Azure.

## 1. Wymagania

- Python 3.13.5
- 3 kamery na USB (w wersji jaka jest w projekcie muszą być równo 3 kamery, bo inaczej API tego nie przyjmie)
- Raspberry Pi 3B+ (albo jakikolwiek inny komputer z linuxem)

## 2. Instalacja

- Python: `sudo apt install python3 python3-pip`
- Wymagania do OpenCV: `sudo apt install build-essential cmake git libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev`
- OpenCV: `sudo apt install python3-opencv`
- (Instrukcja do instalacji OpenCV na RPi: https://opencv.org/blog/raspberry-pi-with-opencv/)

## 3. Konfiguracja kamer
   - Domyślnie skrypt używa indeksów kamer 0, 2 i 4. Aby sprawdzić jakie indeksy mają podłączone kamery, należy użyć polecenia: `ls -1 /dev/video*`
   - Pokaże się wtedy lista kamer np. `/dev/video0, /dev/video1` co oznacza że nasze kamery w tym przykładzie mają indeksy 0 i 1.
   - Może się tak zdarzyć (jak w naszym przypadku), że jedna kamera generuje dwa urządzenia (jak dla nas 0, 1, 2, 3, 4, 5), wtedy dla każdej kamery wpisujemy tylko jej pierwszy indeks (0, 2, 4).
   - Indeksy kamer można zmienić podmieniając linijkę `CAM_INDEX = [0, 2, 4]` w pliku `record_image.py`
## 4. Uruchamianie skryptu
   - Skrypt uruchamiamy poleceniem: `API_URL="http://adres-do-serwera.com/api" python3 record_image.py`
   - Adres można również podmienić w samym pliku skryptu.

