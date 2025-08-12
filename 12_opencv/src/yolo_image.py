from ultralytics import YOLO

# YOLO 모델 설정
model = YOLO('yolo11n.pt')

results = model('http://ultralytics.com/images/bus.jpg')

results[0].show()