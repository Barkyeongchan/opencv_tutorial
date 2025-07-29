# OpenCV 컴퓨터 비전

> 언어 : Python 3.10
> 
> 라이브러리 : OpenCV, NumPy, Matplotlib
> 
> 환경 : VSCode + venv(파이썬 가상환경)


## 1. 파이썬 가상환경(venv)생성 - Virtual Environment

1. **Visual Studio Code에서 원하는 디렉토리 열기**  
   - 프로젝트를 저장할 폴더를 선택 후, 해당 폴더를 VSCode로 열기

2. **Git Bash 또는 VSCode 터미널 실행**  
   - VSCode에서 `Ctrl + ~` 단축키 또는 상단 메뉴 → `Terminal > New Terminal` 클릭  
   - 터미널이 열린 경로 확인 - 디렉토리 경로와 같아야 함

3. **가상환경 생성**
   ```bash
   python -m venv [가상환경이름]
   ```
   예시:
   ```bash
   python -m venv myvenv
   ```

4. **가상환경 활성화**
   ```bash
   source [가상환경이름]/Scripts/activate
   ```
   예시:
   ```bash
   source myvenv/Scripts/activate
   ```
   - 프롬포트에 `(myvenv)`가 생기면 가상환경이 활성화

5. **가상환경 비활성화**
   ```bash
   deactivate
   ```
   - 가상환경을 종료하고 기본 환경으로 돌아감


## 2. OpenCV 라이브러리

OpenCV 설치 및 환경 설정

이미지 읽기, 변환, 처리, 저장

이미지 색공간 변환 (BGR, BGRA, GRAY SCALE, HSV 등)
