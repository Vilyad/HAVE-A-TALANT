import cv2
import numpy as np
from ultralytics import YOLO
import time

class YOLOv11CameraDetector:
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.5, iou_threshold=0.5):
        """
        Инициализация детектора YOLO11
        
        Args:
            model_path: путь к модели YOLO11
            conf_threshold: порог уверенности для детекции
            iou_threshold: порог для подавления немаксимумов
        """
        # Загружаем модель
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Получаем имена классов
        self.class_names = self.model.names
        print(f"Загружена модель YOLO11 с {len(self.class_names)} классами")
        
        # Цвета для разных классов
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
    
    def process_frame(self, frame):
        """
        Обработка одного кадра
        """
        # Выполняем детекцию
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False  # Отключаем вывод в консоль для каждого кадра
        )
        
        # Обрабатываем результаты
        processed_frame = self.draw_detections(frame, results[0])
        
        return processed_frame, results[0]
    
    def draw_detections(self, frame, results):
        """
        Рисует bounding boxes и подписи на кадре
        """
        frame_copy = frame.copy()
        
        # Проверяем есть ли детекции
        if len(results.boxes) == 0:
            return frame_copy
        
        # Получаем информацию о детекциях
        boxes = results.boxes.xyxy.cpu().numpy()  # координаты bbox
        confidences = results.boxes.conf.cpu().numpy()  # уверенности
        class_ids = results.boxes.cls.cpu().numpy().astype(int)  # ID классов
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Получаем имя класса и цвет
            class_name = self.class_names[class_id]
            color = self.colors[class_id]
            
            # Рисуем bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Создаем подпись
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Рисуем подложку для текста
            cv2.rectangle(frame_copy, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Рисуем текст
            cv2.putText(frame_copy, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame_copy
    
    def run_camera_detection(self, camera_id=0, window_name="YOLO11 Detection"):
        """
        Запускает детекцию с веб-камеры
        
        Args:
            camera_id: ID камеры (0 для встроенной камеры)
            window_name: название окна для отображения
        """
        # Открываем камеру
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть камеру {camera_id}")
            return
        
        print("Камера успешно подключена")
        print("Нажмите 'q' для выхода")
        print("Нажмите 's' для сохранения кадра")
        
        fps_time = time.time()
        frame_count = 0
        
        try:
            while True:
                # Читаем кадр
                ret, frame = cap.read()
                
                if not ret:
                    print("Ошибка: Не удалось получить кадр с камеры")
                    break
                
                # Обрабатываем кадр
                processed_frame, results = self.process_frame(frame)
                
                # Рассчитываем FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - fps_time)
                    fps_time = current_time
                    
                    # Добавляем FPS на кадр
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Показываем информацию о детекциях
                if len(results.boxes) > 0:
                    detections_info = f"Detections: {len(results.boxes)}"
                    cv2.putText(processed_frame, detections_info, 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Отображаем результат
                cv2.imshow(window_name, processed_frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Сохраняем кадр
                    timestamp = int(time.time())
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Кадр сохранен как {filename}")
                
        except KeyboardInterrupt:
            print("Программа прервана пользователем")
        
        finally:
            # Освобождаем ресурсы
            cap.release()
            cv2.destroyAllWindows()
            print("Ресурсы освобождены")

# Дополнительные функции
def list_available_cameras(max_test=5):
    """
    Проверяет доступные камеры
    """
    print("Поиск доступных камер...")
    available_cameras = []
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    print(f"Доступные камеры: {available_cameras}")
    return available_cameras

def main():
    """
    Основная функция
    """
    # Проверяем доступные камеры
    cameras = list_available_cameras()
    
    if not cameras:
        print("Камеры не найдены!")
        return
    
    # Создаем детектор
    try:
        # Попробуем загрузить модель YOLO11
        detector = YOLOv11CameraDetector(
            model_path='yolov11n.pt',  # или 'yolo11n.pt'
            conf_threshold=0.5,
            iou_threshold=0.5
        )
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Пробуем загрузить стандартную модель YOLOv8...")
        try:
            detector = YOLOv11CameraDetector(model_path='yolov8n.pt')
        except Exception as e:
            print(f"Ошибка: {e}")
            return
    
    # Запускаем детекцию
    camera_id = cameras[0]  # Используем первую доступную камеру
    detector.run_camera_detection(camera_id=camera_id)

# Альтернативная версия для конкретной камеры
def simple_detection():
    """
    Упрощенная версия для быстрого запуска
    """
    # Загружаем модель
    model = YOLO('yolov11n.pt')  # или 'yolov8n.pt'
    
    # Открываем камеру
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детекция
        results = model.predict(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()  # Автоматическая аннотация
        
        # Показываем результат
        cv2.imshow('YOLO Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Запускаем основную версию
    main()
    
    # Или упрощенную версию
    # simple_detection()