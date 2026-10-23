!pip install ultralytics
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# 角度計算関数
# ----------------------------
def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    angle = np.degrees(
        np.arccos(
            np.clip(cosine, -1.0, 1.0)
        )
    )

    return angle

# ----------------------------
# モデル読み込み
# ----------------------------
model = YOLO("yolov8n-pose.pt")


# ----------------------------
# メイン解析関数
# ----------------------------
def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_file = "knee_angle_output.mp4"

    writer = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    min_left_angle = 180
    min_right_angle = 180

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = model(frame, verbose=False)[0]

        if result.keypoints is not None:

            points = result.keypoints.xy.cpu().numpy()

            if len(points) > 0:

                kpts = points[0]



            # ----------------------------
            # 全関節番号表示
            # ----------------------------
            for i, point in enumerate(kpts):

                x = int(point[0])
                y = int(point[1])

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

            # ----------------------------
            # 下半身関節
            # ----------------------------
            left_hip = kpts[11]
            left_knee = kpts[13]
            left_ankle = kpts[15]

            right_hip = kpts[12]
            right_knee = kpts[14]
            right_ankle = kpts[16]

            # ----------------------------
            # 膝角度計算
            # ----------------------------
            left_angle = calculate_angle(
                left_hip,
                left_knee,
                left_ankle
            )

            right_angle = calculate_angle(
                right_hip,
                right_knee,
                right_ankle
            )

            # ----------------------------
            # 最小角度更新
            # ----------------------------
            min_left_angle = min(
                min_left_angle,
                left_angle
            )

            min_right_angle = min(
                min_right_angle,
                right_angle
            )

            # ----------------------------
            # 左右差
            # ----------------------------
            angle_diff = abs(
                left_angle -
                right_angle
            )

            # ----------------------------
            # 評価
            # ----------------------------
            avg_angle = (
                left_angle +
                right_angle
            ) / 2

            if avg_angle < 90:
                status = "Excellent"

            elif avg_angle < 120:
                status = "Good"

            else:
                status = "Shallow"

            # ----------------------------
            # スコア計算
            # ----------------------------
            current_min_angle = min(
                min_left_angle,
                min_right_angle
            )

            score = max(
                0,
                min(
                    100,
                    int((180 - current_min_angle) / 1.5)
                )
            )

            if score >= 80:
                comment = "Excellent"

            elif score >= 60:
                comment = "Good"

            else:
                comment = "Needs Improvement"



            # ----------------------------
            # 骨格線(左脚)
            # ----------------------------
            cv2.line(
                frame,
                (int(left_hip[0]), int(left_hip[1])),
                (int(left_knee[0]), int(left_knee[1])),
                (255, 255, 0),
                3
            )

            cv2.line(
                frame,
                (int(left_knee[0]), int(left_knee[1])),
                (int(left_ankle[0]), int(left_ankle[1])),
                (255, 255, 0),
                3
            )

            # ----------------------------
            # 骨格線(右脚)
            # ----------------------------
            cv2.line(
                frame,
                (int(right_hip[0]), int(right_hip[1])),
                (int(right_knee[0]), int(right_knee[1])),
                (255, 255, 0),
                3
            )

            cv2.line(
                frame,
                (int(right_knee[0]), int(right_knee[1])),
                (int(right_ankle[0]), int(right_ankle[1])),
                (255, 255, 0),
                3
            )

            # ----------------------------
            # 股関節 赤
            # ----------------------------
            cv2.circle(
                frame,
                (int(left_hip[0]), int(left_hip[1])),
                8,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                (int(right_hip[0]), int(right_hip[1])),
                8,
                (0, 0, 255),
                -1
            )

            # ----------------------------
            # 膝 緑
            # ----------------------------
            cv2.circle(
                frame,
                (int(left_knee[0]), int(left_knee[1])),
                8,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                frame,
                (int(right_knee[0]), int(right_knee[1])),
                8,
                (0, 255, 0),
                -1
            )

            # ----------------------------
            # 足首 青
            # ----------------------------
            cv2.circle(
                frame,
                (int(left_ankle[0]), int(left_ankle[1])),
                8,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                (int(right_ankle[0]), int(right_ankle[1])),
                8,
                (255, 0, 0),
                -1
            )

            # ----------------------------
            # リアルタイム角度
            # ----------------------------
            cv2.putText(
                frame,
                f"L:{left_angle:.1f}",
                (
                    int(left_knee[0]),
                    int(left_knee[1])
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"R:{right_angle:.1f}",
                (
                    int(right_knee[0]),
                    int(right_knee[1])
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # ----------------------------
            # 情報パネル
            # ----------------------------
            cv2.rectangle(
                frame,
                (10, 10),
                (430, 250),
                (192, 192, 192),
                -1
            )

            cv2.putText(
                frame,
                f"Min Left : {min_left_angle:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Min Right : {min_right_angle:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Diff : {angle_diff:.1f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Status : {status}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Score : {score}/100",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                status,
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        writer.write(frame)
    cap.release()
    writer.release()

    # ----------------------------
    # 最終評価
    # ----------------------------
    min_angle = min(
        min_left_angle,
        min_right_angle
    )

    diff = abs(
        min_left_angle -
        min_right_angle
    )

    score = max(
        0,
        min(
            100,
            int((180 - min_angle) / 1.5)
        )
    )

    if score >= 80:
        comment = "Excellent：十分に膝を曲げられています"

    elif score >= 60:
        comment = "Good：もう少し深く曲げましょう"

    else:
        comment = "Needs Improvement：浅い屈伸です"

    if diff < 5:
        balance = "左右バランス良好"

    elif diff < 15:
        balance = "左右差ややあり"

    else:
        balance = "要改善"

    result_text = f"""
　解析結果
　-------------------
  左膝最小角度 : {min_left_angle:.1f}°
  右膝最小角度 : {min_right_angle:.1f}°
  フォームスコア : {score}/100
  フォーム評価：{comment}
  左右差 : {diff:.1f}°
  バランス評価：{balance}
"""

    return output_file, result_text

def run_analysis(video):

    output_video, result_text = analyze_video(video)

    return output_video, result_text

app = gr.Interface(
    fn=run_analysis,

    inputs=gr.Video(
        label="ラジオ体操動画"
    ),

    outputs=[
        gr.Video(
            label="解析結果"
        ),

        gr.Textbox(
            label="評価結果"
        )
    ],

    title="AIラジオ体操フォーム解析システム",

    description="""
YOLO Poseを利用した
膝屈伸フォーム解析アプリ
"""
)
app.launch(share=True)
