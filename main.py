import cv2
from datetime import datetime
import numpy as np
import os

def save_face_image(face_image, filename):
    cv2.imwrite(filename, face_image)
    print(f"Foto guardada como {filename}")

def load_known_faces(person_name):
    known_faces = []
    face_dir = f"fotos/{person_name}"
    if os.path.exists(face_dir):
        for file_name in os.listdir(face_dir):
            if file_name.endswith(".jpg"):
                face_image = cv2.imread(os.path.join(face_dir, file_name), cv2.IMREAD_GRAYSCALE)
                known_faces.append(face_image)
    return known_faces

def compare_faces(known_faces, current_face):
    for known_face in known_faces:
        # Redimensionar las imágenes para que tengan el mismo tamaño
        known_face_resized = cv2.resize(known_face, (current_face.shape[1], current_face.shape[0]))
        # Calcular la diferencia absoluta entre las dos imágenes
        diff = cv2.absdiff(known_face_resized, current_face)
        # Calcular la suma de las diferencias
        result = np.sum(diff)
        if result < 10000:  # Ajusta este umbral según sea necesario
            return True
    return False

# Inicializar la captura de video desde la webcam
video = cv2.VideoCapture(0)

# Cargar el clasificador en cascada preentrenado para la detección de caras
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preguntar al usuario por el nombre de la persona
nombre = input('Nombre de la persona: ')

# Crear un directorio para la persona si no existe
os.makedirs(f"fotos/{nombre}", exist_ok=True)

# Cargar las caras conocidas
known_faces = load_known_faces(nombre)

# Capturar y almacenar nuevas caras si no hay caras conocidas
if not known_faces:
    salir_loop = False
    while not salir_loop:
        # Capturar un solo frame del video
        ret, frame = video.read()
        if not ret or frame is None:
            print("Error: No se pudo capturar el frame.")
            break

        # Convertir el frame a escala de grises
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar caras en el frame en escala de grises
        faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))

        # Dibujar rectángulos alrededor de las caras detectadas
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 51, 151), 2)
            cv2.putText(frame, nombre, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 51, 151), 2)
            # Extraer la imagen de la cara
            face_image = gray_frame[y:y+h, x:x+w]

        # Mostrar el frame con las caras detectadas
        cv2.imshow('Live Reaction', frame)

        # Salir del loop si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_face_image(face_image, f"fotos/{nombre}/face_{timestamp}.jpg")
            known_faces.append(face_image)
            salir_loop = True

salir_loop = False

while not salir_loop:
    # Capturar un solo frame del video
    ret, frame = video.read()
    if not ret or frame is None:
        print("Error: No se pudo capturar el frame.")
        break

    # Convertir el frame a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el frame en escala de grises
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))

    # Comparar la cara detectada con las imágenes conocidas
    for (x, y, w, h) in faces:
        current_face_image = gray_frame[y:y+h, x:x+w]
        if compare_faces(known_faces, current_face_image):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{nombre} - Coincidencia", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Desconocido", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Mostrar el frame con las caras detectadas
    cv2.imshow('Live Reaction', frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        salir_loop = True

# Liberar la captura de video y cerrar las ventanas
video.release()
cv2.destroyAllWindows()
