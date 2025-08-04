import cv2
import glob
import numpy as np
import time

# --- 매칭 파라미터 ---
ratio = 0.7
MIN_MATCH = 10

# --- ORB 생성 ---
detector = cv2.ORB_create(nfeatures=1000)

# --- FLANN 매처 생성 ---
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# --- 트래커 리스트 ---
trackers = [cv2.legacy.TrackerBoosting_create,
            cv2.legacy.TrackerMIL_create,
            cv2.legacy.TrackerKCF_create,
            cv2.legacy.TrackerTLD_create,
            cv2.legacy.TrackerMedianFlow_create,
            cv2.legacy.TrackerCSRT_create,
            cv2.legacy.TrackerMOSSE_create]

trackerIdx = 0
tracker = None
is_tracking = False

# --- 책 검색 함수 ---
def search_book(query_img):
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
                    best_accuracy = accuracy
                    best_result = cover_path
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

# --- 카메라 열기 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

qImg = None
roi = None

win_name = 'Book Search and Tracking'

while True:
    if not is_tracking:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        h, w = frame.shape[:2]
        left = w // 3
        right = (w // 3) * 2
        top = (h // 2) - (h // 3)
        bottom = (h // 2) + (h // 3)

        flip = cv2.flip(frame, 1)
        cv2.imshow(win_name, flip)

        key = cv2.waitKey(10) & 0xFF

        if key == ord(' '):
        # ROI 직접 선택
            roi = cv2.selectROI(win_name, frame, False)
            if roi[2] > 0 and roi[3] > 0:
                x, y, w_roi, h_roi = [int(v) for v in roi]
                qImg = frame[y:y+h_roi, x:x+w_roi].copy()
                cv2.imshow('Query Image', qImg)
                print("Query image captured. Searching books...")
                search_book(qImg)
                print("Matching complete. Press Space to select ROI for tracking.")
        elif key == 27:
            break

    else:
        # 트래킹 모드
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        ok, bbox = tracker.update(frame)
        img_draw = frame.copy()

        if ok:
            x, y, w_box, h_box = [int(v) for v in bbox]
            cv2.rectangle(img_draw, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        else:
            cv2.putText(img_draw, "Tracking fail.", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        trackerName = tracker.__class__.__name__
        cv2.putText(img_draw, f"Tracker: {trackerIdx} - {trackerName}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(win_name, img_draw)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            # 트래킹 종료 후 다시 검색 모드로
            print("Tracking stopped. Returning to capture mode.")
            is_tracking = False
            tracker = None
            roi = None
        elif key in range(ord('0'), ord('7')):
            # 트래커 종류 변경
            trackerIdx = key - ord('0')
            if tracker is not None and roi is not None:
                tracker = trackers[trackerIdx]()
                tracker.init(frame, roi)
                print(f"Tracker changed to {tracker.__class__.__name__}")

        elif key == 27:
            break

    # ROI 선택 (트래킹 시작)
    if not is_tracking and qImg is not None:
        print("Select ROI for tracking and press ENTER or SPACE.")
        roi = cv2.selectROI(win_name, frame, False)
        if roi[2] > 0 and roi[3] > 0:
            tracker = trackers[trackerIdx]()
            tracker.init(frame, roi)
            is_tracking = True
            print(f"Tracking started with {tracker.__class__.__name__}")

cv2.destroyAllWindows()
cap.release()
