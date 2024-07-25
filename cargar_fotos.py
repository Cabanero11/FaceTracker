import os
import face_recognition

# Directorio de imágenes de referencia
ruta_fotos = 'fotos/'

# Listar todas las imágenes en el directorio
imagenes_personas = [f for f in os.listdir(ruta_fotos) if os.path.isfile(os.path.join(ruta_fotos, f))]

# Diccionario para almacenar las codificaciones de las caras y los nombres
codificaciones_conocidas = []
nombres_conocidos = []

# Cargar y codificar las imágenes de referencia
for imagen_archivo in imagenes_personas:
    nombre_persona = os.path.splitext(imagen_archivo)[0]
    imagen_ruta = os.path.join(ruta_fotos, imagen_archivo)
    imagen = face_recognition.load_image_file(imagen_ruta)
    codificaciones = face_recognition.face_encodings(imagen)
    
    if codificaciones:
        codificaciones_conocidas.append(codificaciones[0])
        nombres_conocidos.append(nombre_persona)
