import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, webrtc_streamer

st.set_page_config(page_title="Live Camera Mesh", layout="wide")
st.title("Live Camera Mesh")

# ---------------- Sidebar controls ----------------
mode = st.sidebar.radio("Mesh mode", ["Face mesh", "Hand mesh", "Scene mesh (Delaunay)"])
color_hex = st.sidebar.color_picker("Mesh color", "#00FF7F")
thickness = st.sidebar.slider("Line thickness", 1, 3, 1)
mirror = st.sidebar.checkbox("Mirror view", True)
black_bg = st.sidebar.checkbox("Mesh only (black background)", False)
n_points = st.sidebar.slider("Scene mesh: number of points", 30, 400, 150, disabled=mode != "Scene mesh (Delaunay)")

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


class MeshProcessor:
    def __init__(self):
        self.mode = "Face mesh"
        self.color = (127, 255, 0)
        self.thickness = 1
        self.mirror = True
        self.black_bg = False
        self.n_points = 150
        self.face = mp_face.FaceMesh(max_num_faces=2, refine_landmarks=True,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    def _scene_mesh(self, img, canvas):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=self.n_points,
                                          qualityLevel=0.01, minDistance=max(10, w // 40))
        pts = [] if corners is None else [tuple(map(float, p.ravel())) for p in corners]
        # frame anchors so the mesh covers the whole image
        for x in (0, w // 2, w - 1):
            for y in (0, h // 2, h - 1):
                pts.append((float(x), float(y)))
        subdiv = cv2.Subdiv2D((0, 0, w, h))
        for p in pts:
            try:
                subdiv.insert(p)
            except cv2.error:
                pass
        for t in subdiv.getTriangleList():
            tri = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
            if all(0 <= x < w and 0 <= y < h for x, y in tri):
                cv2.polylines(canvas, [np.array(tri)], True, self.color, self.thickness, cv2.LINE_AA)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.mirror:
            img = cv2.flip(img, 1)
        canvas = np.zeros_like(img) if self.black_bg else img.copy()
        spec = mp_draw.DrawingSpec(color=self.color, thickness=self.thickness, circle_radius=0)

        if self.mode == "Face mesh":
            res = self.face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for lm in (res.multi_face_landmarks or []):
                mp_draw.draw_landmarks(canvas, lm, mp_face.FACEMESH_TESSELATION, None, spec)
                mp_draw.draw_landmarks(canvas, lm, mp_face.FACEMESH_CONTOURS, None,
                                       mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=0))
        elif self.mode == "Hand mesh":
            res = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for lm in (res.multi_hand_landmarks or []):
                mp_draw.draw_landmarks(canvas, lm, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3), spec)
        else:
            self._scene_mesh(img, canvas)

        return av.VideoFrame.from_ndarray(canvas, format="bgr24")


ctx = webrtc_streamer(
    key="mesh",
    video_processor_factory=MeshProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": {"width": 960, "height": 540}, "audio": False},
    async_processing=True,
)

# push sidebar settings into the running video processor
if ctx.video_processor:
    p = ctx.video_processor
    p.mode, p.color, p.thickness = mode, hex_to_bgr(color_hex), thickness
    p.mirror, p.black_bg, p.n_points = mirror, black_bg, n_points

st.caption("Click START and allow camera access. Camera works on localhost or HTTPS only.")
