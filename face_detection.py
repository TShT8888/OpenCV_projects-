import cv2

# Загружаем каскад Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier('C:\HW\cv2\haarcascade_frontalface_default.xml')

# Открываем веб-камеру
webcam = cv2.VideoCapture(0)

while True:
    # Читаем изображение с камеры
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    
    # Рисуем прямоугольники вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Отображаем изображение
    cv2.imshow('Face detection', img)
    
    # Ждем нажатия клавиши "Esc" (код 27) для выхода
    key = cv2.waitKey(10)
    if key == 27:
        break

# Освобождаем ресурсы
webcam.release()
cv2.destroyAllWindows()
