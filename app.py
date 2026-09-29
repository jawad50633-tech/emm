import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

st.set_page_config(page_title="Camera Mesh", layout="wide")
st.title("Camera Mesh")

mode = st.sidebar.radio("Mesh mode", ["Face mesh", "Hand mesh", "Scene mesh (Delaunay)"])
color_hex = st.sidebar.color_picker("Mesh color", "#00FF7F")
thickness = st.sidebar.slider("Line thickness", 1, 3, 1)
mirror = st.sidebar.checkbox("Mirror image", True)
black_bg = st.sidebar.checkbox("Mesh only (black background)", False)
n_points = st.sidebar.slider("Scene mesh: number of points", 30, 400, 150,
                             disabled=mode != "Scene mesh (Delaunay)")

mp_face, mp_hands, mp_draw = mp.solutions.face_mesh, mp.solutions.hands, mp.solutions.drawing_utils


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def scene_mesh(img, canvas, color, thick, n):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=n, qualityLevel=0.01, minDistance=max(10, w // 40))
    pts = [] if corners is None else [tuple(map(float, p.ravel())) for p in corners]
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
            cv2.polylines(canvas, [np.array(tri)], True, color, thick, cv2.LINE_AA)


def make_mesh(img):
    if mirror:
        img = cv2.flip(img, 1)
    canvas = np.zeros_like(img) if black_bg else img.copy()
    color = hex_to_bgr(color_hex)
    spec = mp_draw.DrawingSpec(color=color, thickness=thickness, circle_radius=0)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mode == "Face mesh":
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=4, refine_landmarks=True) as fm:
            res = fm.process(rgb)
        found = bool(res.multi_face_landmarks)
        for lm in res.multi_face_landmarks or []:
            mp_draw.draw_landmarks(canvas, lm, mp_face.FACEMESH_TESSELATION, None, spec)
            mp_draw.draw_landmarks(canvas, lm, mp_face.FACEMESH_CONTOURS, None,
                                   mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=0))
    elif mode == "Hand mesh":
        with mp_hands.Hands(static_image_mode=True, max_num_hands=4) as hd:
            res = hd.process(rgb)
        found = bool(res.multi_hand_landmarks)
        for lm in res.multi_hand_landmarks or []:
            mp_draw.draw_landmarks(canvas, lm, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3), spec)
    else:
        scene_mesh(img, canvas, color, thickness, n_points)
        found = True
    return canvas, found


shot = st.camera_input("Take a photo")

if shot is not None:
    img = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    out, found = make_mesh(img)
    st.image(out, channels="BGR", use_container_width=True)
    if not found:
        st.warning(f"No {mode.split()[0].lower()} detected. Try better lighting or move closer.")
    ok, png = cv2.imencode(".png", out)
    if ok:
        st.download_button("Download result", png.tobytes(), "mesh.png", "image/png")
