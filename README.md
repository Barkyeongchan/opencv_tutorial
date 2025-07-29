# OpenCV 컴퓨터 비전

> 언어 : Python 3.10
> 
> 라이브러리 : OpenCV, NumPy, Matplotlib
> 
> 환경 : VSCode + venv(파이썬 가상환경)


## 1. 파이썬 가상환경(venv)생성 - Virtual Environment

Visual Studio Code에서 원하는 디렉토리 열기

프로젝트를 저장할 폴더를 선택 후, 해당 폴더를 VSCode로 엽니다.

Git Bash 또는 VSCode 터미널 실행

VSCode에서 Ctrl + ~ 단축키 또는 상단 메뉴 → Terminal > New Terminal 클릭

터미널이 열린 경로가 현재 프로젝트 폴더인지 확인합니다.

가상환경 생성

bash
복사
편집
python -m venv [가상환경이름]
예시:

bash
복사
편집
python -m venv myvenv
가상환경 활성화

bash
복사
편집
source [가상환경이름]/Scripts/activate
예시:

bash
복사
편집
source myvenv/Scripts/activate
(myvenv) 와 같이 프롬프트에 접두사가 생기면 가상환경이 성공적으로 활성화된 것입니다.

가상환경 비활성화

bash
복사
편집
deactivate
언제든지 가상환경을 종료하고 기본 환경으로 돌아갈 수 있습니다.


## 2. OpenCV 라이브러리

OpenCV 설치 및 환경 설정

이미지 읽기, 변환, 처리, 저장

이미지 색공간 변환 (BGR, BGRA, GRAY SCALE, HSV 등)
