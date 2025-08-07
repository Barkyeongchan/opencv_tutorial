# 전역 변수로 상태 관리 (간단하고 이해하기 쉬움)
consecutive_frames = 0      # 연속으로 눈 감은 프레임 수
ear_history = []           # EAR 값 이력 (그래프 그리기용)
alert_level = "정상"     # 현재 경고 레벨
last_alert_time = 0        # 마지막 경고 시간

# 설정값들
EAR_THRESHOLD = 0.25       # EAR 임계값
DROWSY_FRAMES = 20         # 졸음 판단 프레임 수
MAX_HISTORY = 50           # 이력 저장 최대 개수

def main():
    # 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1단계: 얼굴 검출
        faces = detect_faces(frame, detector)

        if len(faces) > 0:
            # 2단계: 랜드마크 추출
            landmarks = get_landmarks(frame, faces[0], predictor)

            # 3단계: EAR 계산
            ear_value = calculate_ear_from_landmarks(landmarks)
       
            # 4단계: 졸음 판단
            drowsy_state = check_drowsiness(ear_value)

            # 5단계: 결과 표시
            draw_results(frame, landmarks, ear_value, drowsy_state)

        
        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) == 27:  # ESC 키
            break
      
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()