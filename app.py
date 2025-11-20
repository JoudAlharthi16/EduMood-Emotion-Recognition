import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import pandas as pd
import altair as alt

from edumood_recognizer import EduMoodRecognizer, EduMoodSessionStats

# إعداد صفحة Streamlit
st.set_page_config(layout="wide")
st.sidebar.title("EduMood – Students Emotion Recognition")

st.title("EduMood – Face Emotion Detection in Classrooms")
st.subheader("Press start to capture students in the classroom!")

# نجهز إحصائيات جلسة واحدة ونخزنها في session_state
if "session_stats" not in st.session_state:
    st.session_state["session_stats"] = EduMoodSessionStats()

session_stats: EduMoodSessionStats = st.session_state["session_stats"]


# Callback نهاية الفيديو (اختياري)
def endVideo():
    print("Video has ended!")


# نجهز الـ recognizer (هنا نتحكم كل كم فريم نحلّل)
recognizer = EduMoodRecognizer(session_stats, analyze_every_n=5)

webrtc_streamer(
    key="edumood-example",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True},
    video_frame_callback=recognizer.recognize,
    on_video_ended=endVideo,
)

# ================== جزء التقارير (مبني على فكرة Kevin) ==================

df = session_stats.to_dataframe()
st.title("Raw emotion records dataframe")
st.dataframe(df)

if not df.empty:
    # نحول recorded_at إلى datetime ونخليه index
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df_indexed = df.set_index("recorded_at")
    st.subheader("Dataframe indexed by time")
    st.dataframe(df_indexed)

    # نحدد أعمدة المشاعر فقط
    emotion_cols = ["happy", "sad", "angry", "surprise", "neutral", "disgusted", "fearful"]

    # مجموع عدد المرات التي ظهر فيها كل شعور خلال الجلسة
    st.title("Total students by emotion (sum)")
    df_grouped = df_indexed[emotion_cols].sum().sort_values(ascending=False)
    st.dataframe(df_grouped)

    # Metrics لكل شعور
    st.title("Emotion metrics")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Happy", int(df_grouped.get("happy", 0)))
    col2.metric("Sad", int(df_grouped.get("sad", 0)))
    col3.metric("Angry", int(df_grouped.get("angry", 0)))
    col4.metric("Surprised", int(df_grouped.get("surprise", 0)))
    col5.metric("Neutral", int(df_grouped.get("neutral", 0)))
    col6.metric("Disgusted", int(df_grouped.get("disgusted", 0)))
    col7.metric("Fearful", int(df_grouped.get("fearful", 0)))

    # الشعور الأكثر تكراراً في الجلسة
    st.title("Emotion with the highest total")
    st.write(df_grouped.idxmax())
    st.write(int(df_grouped.max()))

    # خط زمني لتغيّر المشاعر مع الوقت
    st.title("Total students by emotion over time")
    st.line_chart(df_indexed[emotion_cols])

    # -------------------------- Using Altair --------------------------
    st.title("Total students per emotion (bar chart)")

    df_for_chart = df_grouped.reset_index()
    df_for_chart.columns = ["emotion", "total_students"]
    st.dataframe(df_for_chart)

    source_dataframe = pd.DataFrame(df_for_chart)

    bar_chart = alt.Chart(source_dataframe).mark_bar().encode(
        x=alt.X("emotion", title="Emotion"),
        y=alt.Y("total_students", title="Total detections"),
        color=alt.Color("emotion", legend=None),
    )

    st.altair_chart(bar_chart, use_container_width=True)

else:
    st.info("No emotion data yet. Start the camera and wait for detections to appear.")
