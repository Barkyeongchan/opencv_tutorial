import cv2

def draw_eye_landmarks(frame, landmarks):
    # 눈 랜드마크 그리기
    # landmarks: 눈 좌표 리스트
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

def draw_alert_message(frame, alert_level):
    # 경고 메시지 표시
    colors = {
        'NORMAL': (0, 255, 0),
        'DROWSY': (0, 255, 255),
        'ALERT': (0, 165, 255),
        'DANGER': (0, 0, 255)
    }
    color = colors.get(alert_level, (255, 255, 255))
    cv2.putText(frame, f'Status: {alert_level}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)