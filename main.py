import cv2
import numpy as np
import datetime
import argparse

parser = argparse.ArgumentParser(description='Detector de objetos de color.')
parser.add_argument('--color', default='rojo', help='Color a detectar (por ejemplo, "rojo", "verde", "azul").')
args = parser.parse_args()


# Abre el vídeo
video_capture = cv2.VideoCapture('VideoConcept1.mp4')  # O cambia a 0 para usar una cámara en tiempo real

# Inicializa el fotograma anterior
prev_frame = None

while True:
    # Captura un fotograma del vídeo
    ret, frame = video_capture.read()

    if not ret:
        break

    color = args.color.lower()  # Convierte el color a minúsculas para que sea insensible a mayúsculas
    if color == 'rojo':
        lower_color = np.array([0, 0, 150])  # Umbral inferior para el color rojo en el espacio BGR
        upper_color = np.array([80, 80, 255])  # Umbral superior para el color rojo en el espacio BGR
    elif color == 'verde':
        lower_color = np.array([0, 100, 0])  # Umbral inferior para el color verde en el espacio BGR
        upper_color = np.array([80, 255, 80])  # Umbral superior para el color verde en el espacio BGR
    elif color == 'azul':
        lower_color = np.array([80, 0, 0])  # Umbral inferior para el color azul en el espacio BGR
        upper_color = np.array([255, 80, 80])  # Umbral superior para el color azul en el espacio BGR
    elif color == 'amarillo':
        lower_color = np.array([0, 150, 150])  # Umbral inferior para el color amarillo en el espacio BGR
        upper_color = np.array([80, 255, 255])  # Umbral superior para el color amarillo en el espacio BGR
    elif color == 'blanco':
        lower_color = np.array([220, 220, 220])  # Umbral inferior para el color blanco en el espacio BGR
        upper_color = np.array([255, 255, 255])  # Umbral superior para el color blanco en el espacio BGR
    elif color == 'negro':
        lower_color = np.array([0, 0, 0])  # Umbral inferior para el color negro en el espacio BGR
        upper_color = np.array([50, 50, 50])  # Umbral superior para el color negro en el espacio BGR
    elif color == 'gris':
        lower_color = np.array([100, 100, 100])  # Umbral inferior para el color gris en el espacio BGR
        upper_color = np.array([200, 200, 200])  # Umbral superior para el color gris en el espacio BGR
    else:
        print("Color no válido. Debes especificar 'rojo', 'verde', 'azul', 'amarillo', 'blanco', 'negro' o 'gris'")
        exit(1)

    color_mask = cv2.inRange(frame, lower_color, upper_color)

    # Detección de movimiento
    if prev_frame is not None:
        diff_frame = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
        _, motion_mask = cv2.threshold(gray_diff, 5, 255, cv2.THRESH_BINARY)

        # Encuentra contornos de objetos en movimiento del color especificado
        motion_color_mask = cv2.bitwise_and(color_mask, motion_mask)
        contours, _ = cv2.findContours(motion_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours = []  # Define contours como una lista vacía si no hay fotograma anterior

    for contour in contours:
        # Filtra contornos pequeños
        if cv2.contourArea(contour) > 1000:
            # Dibuja un rectángulo alrededor del objeto en movimiento del color especificado
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Guarda la captura de pantalla en la carpeta "eventos" con fecha y hora
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = f"Eventos/evento_{timestamp}.jpg"
            cv2.imwrite(file_path, frame)

    # Actualiza el fotograma anterior
    prev_frame = frame

    # Muestra el vídeo con objetos detectados
    cv2.imshow(f'Objetos de color {color.capitalize()}', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la captura de vídeo y cierra las ventanas
video_capture.release()
cv2.destroyAllWindows()
