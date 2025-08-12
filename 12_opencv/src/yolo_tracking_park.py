from ultralytics import YOLO
import yt_dlp

# 유튜브 영상 다운로드
url = 'https://www.youtube.com/shorts/mnPQH9RRZqc'
ydl_opts = {'outtmpl': 'video.mp4'}  # 저장 파일명 지정
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

# YOLO 모델 로드
model = YOLO('yolo11n.pt')

# 로컬로 저장된 영상 탐지
results = model('video.mp4', stream=True)
for r in results:
    r.show()  # 프레임별 결과 표시
