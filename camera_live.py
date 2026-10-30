"""
camera_live.py  -  standalone OpenCV webcam inspection (no browser).
Run:  python camera_live.py
Keys: SPACE = inspect current frame    A = toggle auto-inspect (every 2 s)    Q = quit
"""
import time
import cv2

from inference import inspect_bgr, model_ready

COLORS = {"GOOD": (0, 170, 0), "DEFECTIVE": (0, 0, 220), "NEEDS REVIEW": (0, 160, 255)}


def main():
    if not model_ready():
        raise SystemExit("No trained model found. Run train.py first.")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW is faster to open on Windows
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    last, auto, last_time = None, False, 0
    print("SPACE = inspect | A = auto mode | Q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if auto and now - last_time > 2:
            last, last_time = inspect_bgr(frame, save_overlay=False), now

        view = frame.copy()
        if last:
            color = COLORS.get(last["verdict"], (255, 255, 255))
            cv2.rectangle(view, (0, 0), (view.shape[1], 50), color, -1)
            text = f"{last['verdict']}  {last['confidence']}%"
            if last["defect_type"] not in ("-", ""):
                text += f"  ({last['defect_type']})"
            cv2.putText(view, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(view, "SPACE inspect | A auto | Q quit", (10, view.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Retail VQC - live", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            last = inspect_bgr(frame, save_overlay=False)
            print(last["verdict"], last["confidence"], last["reason"])
        if key == ord("a"):
            auto = not auto
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
