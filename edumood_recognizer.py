import cv2
from collections import Counter
from datetime import datetime
import uuid

import av
from deepface import DeepFace
import pandas as pd


class EduMoodSessionStats:
    """
    تخزن سجلات المشاعر خلال جلسة واحدة.
    كل record عبارة عن:
    recorded_at + عدد الوجوه لكل شعور
    """

    def __init__(self):
        self.records = []

    def add_record(self, record: dict):
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "recorded_at",
                    "happy",
                    "sad",
                    "angry",
                    "surprise",
                    "neutral",
                    "disgusted",
                    "fearful",
                ]
            )
        return pd.DataFrame(self.records)


class EduMoodRecognizer:
    """
    يقرأ فريمات الفيديو من الكاميرا (عن طريق streamlit-webrtc),
    يقلب الصورة زي المراية,
    ويحلل المشاعر كل N فريم باستخدام DeepFace.
    """

    def __init__(self, session_stats: EduMoodSessionStats, analyze_every_n: int = 5):
        self.session_stats = session_stats
        self.analyze_every_n = analyze_every_n
        self.frame_count = 0
        self.last_annotated = None

        # كاشف الوجوه (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _analyze_frame(self, img_bgr):
        """
        يحلل فريم واحد:
        - يكتشف الوجوه
        - لكل وجه يستخرج dominant_emotion من DeepFace
        - يرسم المربعات والـ labels فوق الصورة
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        emotions = []

        for (x, y, w, h) in faces:
            face_roi = img_bgr[y : y + h, x : x + w]

            try:
                result = DeepFace.analyze(
                    face_roi, actions=["emotion"], enforce_detection=False
                )

                # DeepFace أحياناً يرجع list، أحياناً dict
                if isinstance(result, list):
                    dom = result[0].get("dominant_emotion", "").lower()
                else:
                    dom = result.get("dominant_emotion", "").lower()

                if dom:
                    emotions.append(dom)

                # نرسم المربع والـ label
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    img_bgr,
                    dom,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            except Exception:
                # نطنّش أي خطأ في الفريم هذا (مثلاً فشل DeepFace)
                continue

        return img_bgr, emotions

    def recognize(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        هذه الدالة تُستدعى من streamlit-webrtc على كل فريم.
        ترجع فريم معرّف (مرسوم فوقه المربعات والمشاعر).
        """
        self.frame_count += 1

        # نحول الفريم لـ numpy
        img = frame.to_ndarray(format="bgr24")

        # نقلب الصورة زي المراية
        img = cv2.flip(img, 1)

        # نحلل فقط كل N فريم لتخفيف الضغط
        if self.frame_count % self.analyze_every_n != 0:
            # لو ما عندنا annotated سابق، نرجع الصورة العادية
            if self.last_annotated is None:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            # نرجع آخر فريم تم تحليله
            return av.VideoFrame.from_ndarray(self.last_annotated, format="bgr24")

        # هنا الفريم اللي فعلاً بنحلله بـ DeepFace
        annotated, emotions = self._analyze_frame(img)
        self.last_annotated = annotated

        # نحدّث إحصائيات الجلسة لو فيه مشاعر
        if emotions:
            counts = Counter([e.lower() for e in emotions])

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            record = {
                "recorded_at": now_str,
                "happy": counts.get("happy", 0),
                "sad": counts.get("sad", 0),
                "angry": counts.get("angry", 0),
                "surprise": counts.get("surprise", 0)
                + counts.get("surprised", 0),
                "neutral": counts.get("neutral", 0),
                "disgusted": counts.get("disgust", 0)
                + counts.get("disgusted", 0),
                "fearful": counts.get("fear", 0) + counts.get("fearful", 0),
            }

            self.session_stats.add_record(record)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
