import cv2

# Inicializar la captura de video desde la webcam
video = cv2.VideoCapture(0)

# Cargar el clasificador en cascada preentrenado para la detección de caras
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

salir_loop = False

while not salir_loop:
    # Capturar un solo frame del video
    ret, frame = video.read()

    # Verificar si se ha capturado correctamente el frame
    if not ret or frame is None:
        print("Error: No se pudo capturar el frame.")
        break

    # Convertir el frame a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el frame en escala de grises
    faces = face_classifier.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(35, 35)
    )

    # Dibujar rectángulos alrededor de las caras detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 51, 151), 2)

    # Mostrar el frame con las caras detectadas
    cv2.imshow('Live Reaction', frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        salir_loop = True

# Liberar la captura de video y cerrar las ventanas
video.release()
cv2.destroyAllWindows()