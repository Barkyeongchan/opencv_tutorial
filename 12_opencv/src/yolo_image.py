from ultralytics import YOLO

# YOLO 모델 설정
model = YOLO('yolo11n.pt')

# results = model('http://ultralytics.com/images/bus.jpg')

test_images = [
    'https://ultralytics.com/images/zidan.jpg'
    'https://ultralytics.com/images/bus.jpg'
]

for img in test_images:
    results = model(img)
    print(f'검출된 객체 수 : {len(results[0].boxes)}')

results[0].show()