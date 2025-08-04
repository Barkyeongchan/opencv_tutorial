import cv2, glob, numpy as np
import time

# 매칭 파라미터 설정
ratio = 0.7
MIN_MATCH = 10

# ORB 특징점 검출기 생성
detector = cv2.ORB_create(nfeatures=1000)

# FLANN 매칭기 설정
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 이미지 크기 축소 함수
def resize_image(img, max_width=400):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        img = cv2.resize(img, (max_width, new_h))
    return img

# 책 검색 함수
def search_and_draw(query_img):
    gray1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = detector.detectAndCompute(gray1, None)

    best_result = None
    best_accuracy = 0
    best_cover = None
    best_kp2 = None
    best_matches = None

    cover_paths = glob.glob('../img/books/*.*')

    for cover_path in cover_paths:
        cover = cv2.imread(cover_path)
        gray2 = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            continue

        matches = matcher.knnMatch(desc1, desc2, 2)
        good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * ratio]

        if len(good_matches) > MIN_MATCH:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                accuracy = float(mask.sum()) / mask.size

                if accuracy > best_accuracy:
                    best_result = cover_path
                    best_accuracy = accuracy
                    best_cover = cover
                    best_kp2 = kp2
                    best_matches = good_matches

    if best_result is not None:
        print(f"Best match: {best_result} - 정확도: {best_accuracy:.2%}")
        match_img = cv2.drawMatches(query_img, kp1, best_cover, best_kp2,
                                    best_matches, None,
                                    matchColor=(0, 255, 0),
                                    singlePointColor=None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        cv2.putText(match_img, f"Match: {best_accuracy*100:.2f}%", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Matching Result', match_img)
    else:
        print("No matched book cover found.")
        cv2.imshow('Matching Result', query_img)

# === 카메라에서 책 캡처 및 검색 ===

cap = cv2.VideoCapture(0)
qImg = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 가져올 수 없습니다.")
        break

    h, w = frame.shape[:2]
    left = w // 3
    right = (w // 3) * 2
    top = (h // 2) - (h // 3)
    bottom = (h // 2) + (h // 3)
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 3)

    flip = cv2.flip(frame, 1)
    cv2.imshow('Book Searcher', flip)

    key = cv2.waitKey(10)
    if key == ord(' '):
        qImg = frame[top:bottom, left:right]
        cv2.imshow('Query Image', qImg)
        break
    elif key == 27:
        break

cap.release()

if qImg is not None:
    start_time = time.time()
    search_and_draw(qImg)
    search_time = time.time() - start_time
    print(f"검색 시간: {search_time:.2f}초")

cv2.waitKey(0)
cv2.destroyAllWindows()
