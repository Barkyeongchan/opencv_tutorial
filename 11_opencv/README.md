# 학습 목표 : TensorFlow를 활용해 openCV 프로젝트를 진행한다.

# TensorFlow / 

## 목차

1. Tensorflow
   - TensorFlow란?
   - Tensorflow 설치

## 1. Tensorflow

<details>
<summary></summary>
<div markdown="1">

## **1-1. Tensorflow란?**

구글이 개발한 **오픈소스 머신러닝/딥러닝 프레임워크로, 딥러닝 신경망 모델을 만들고 학습**시키는데 많이 쓰인다.

## **1-2. Tensorflow 설치** 

_**학습시에는 vmware를 통해 우분투에서 실행함**_

**[1. vscode 설치]**

[vscode 메인 페이지](https://code.visualstudio.com/download)에서 .deb 다운로드

1. 다운로드 디렉토리로 이동

```terminla
cd Downloads/
```

2. 다운로드한 파일 실행

```terminal
sudo dpkg -i code_1.103.0-1754517494_amd64.deb
```

3. 다운로드 완

<br><br>

**[2. 윈도우 SSH키를 VMware Ubuntu로 복사하기]**

1. 윈도우에서 SSH 키 위치 확인

```
C:\Users\[사용자명]\.ssh\
```

2. 윈도우에서 복사한 뒤에 VMworkstation에 붙여넣기

```
.ssh\
```

3. 필수 권한 설정

```terminal
# 디렉토리 권한
chmod 700 ~/.ssh

# 개인키 권한 (Ed25519 키의 경우)
chmod 600 ~/.ssh/id_ed25519

chmod 644 ~/.ssh/id_ed25519.pub

# 기타 파일들
chmod 600 ~/.ssh/config          # 설정파일 (있다면)
chmod 600 ~/.ssh/known_hosts     # 호스트 정보

# 소유자 확인
chown -R $USER:$USER ~/.ssh
```

4. SSH 연결 테스트

```terminal
# GitHub 테스트
ssh -T git@github.com

# 성공 시 메시지:
# Hi [사용자명]! You've successfully authenticated, but GitHub does not provide shell access
```

<br><br>

**[3. 가상환경 생성]**

1. 텐서플로우 디렉토리 생성

```termianl
mkdir opencv_tf
cd opencv_tf
```

2. 파이썬 가상환경 설치

```terminal
sudo apt install python3.10-venv
```

3. 가상환경 생성

```terminal
# 가상환경 생성
python3 -m venv tfvenv

# 가상환경 진입
source tfvenv/bin/activate
```

4. 가상환경 진입 후 ros2 충돌시 pip list 해결방법

가상환경 종료` 후

```terminal
# 백업 파일 생성
cp ~/.bashrc ~/.bashrc.backup

# bashrc에서 ros2 자동 실행 명령어 주석처리
nano ~/.bashrc
# source /opt/ros/humble/setup.bash
```

5. ROS2 환경 변수 제거

```terminal
unset ROS_VERSION

unset ROS_PYTHON_VERSION  

unset ROS_LOCALHOST_ONLY

unset ROS_DISTRO

unset AMENT_PREFIX_PATH

unset PYTHONPATH
```

6. 가상환경 삭제 후 재설치

```terminal
# 가상환경 삭제
rm -rf tfvenv

# 가상환경 재설치
python3 -m venv tfvenv
```

7. 가상환경 진입 후 pip list 확인

```terminal
# 가상환경 실행
source tfvenv/bin/activate

# pip list 확인

# Package    Version
# ---------- -------
# pip        22.0.2
# setuptools 59.6.0
```

<br><br>

**[4. tensorflow 설치]**

```terminal
python3 -m pip install tensorflow
```



</div>
</details>
