def update_drowsiness_state(ear_value, threshold, consecutive_frames):
    # 졸음 상태 업데이트
    # ear_value (float): 현재 EAR 값
    # threshold (float): EAR 임계값
    # consecutive_frames (int): 연속으로 임계값 이하인 프레임 수
    if ear_value < threshold:
        consecutive_frames += 1
    else:
        consecutive_frames = 0
    return consecutive_frames

def get_drowsiness_level(consecutive_frames):
    # 졸음 레벨 반환 (정상/주의/경고/위험)
    # consecutive_frames (int): 연속으로 눈 감은 프레임 수    
    if consecutive_frames == 0:
        return 'NORMAL'
    elif consecutive_frames < 10:
        return 'DROWSY'
    elif consecutive_frames < 20:
        return 'ALERT'
    else:
        return 'DANGER'