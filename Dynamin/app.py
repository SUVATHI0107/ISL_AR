"""
ocr_to_obj_viewer.py — Fixed Version (Model Upright)
----------------------------------------------------
Run: python ocr_to_obj_viewer.py
"""

import threading
import time
import sys
import os
import re
import warnings

import cv2
import numpy as np
import pytesseract
import pywavefront

# OpenGL imports
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# -------------------- Configuration --------------------
# ✅ Set correct Tesseract paths
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

CAMERA_INDEX = 0
OCR_INTERVAL = 0.6
MODEL_FOLDER = r"C:\Users\suvat\Downloads\arvr\models"
WINDOW_TITLE = "Hand-Model Viewer (OCR → 3D OBJ)"
WINDOW_SIZE = (800, 600)

# Suppress noisy loader warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Word → model map
WORD_TO_MODEL = {
    "zero":  os.path.join(MODEL_FOLDER, "palm_0.obj"),
    "one":   os.path.join(MODEL_FOLDER, "palm_1.obj"),
    "two":   os.path.join(MODEL_FOLDER, "palm_2.obj"),
    "three": os.path.join(MODEL_FOLDER, "palm_3.obj"),
    "four":  os.path.join(MODEL_FOLDER, "palm_4.obj"),
    "five":  os.path.join(MODEL_FOLDER, "palm_5.obj"),
    "six":   os.path.join(MODEL_FOLDER, "palm_6.obj"),
    "seven": os.path.join(MODEL_FOLDER, "palm_7.obj"),
    "eight": os.path.join(MODEL_FOLDER, "palm_8.obj"),
    "nine":  os.path.join(MODEL_FOLDER, "palm_9.obj"),
    **{str(i): os.path.join(MODEL_FOLDER, f"{i}.obj") for i in range(10)}
}

# -------------------- Shared State --------------------
state = {
    "current_word": None,
    "current_model_path": None,
    "model_changed": False,
    "quit": False
}
state_lock = threading.Lock()

gl_mesh = None
rotation_angle = 0.0
last_load_path = None


# -------------------- OCR Helper --------------------
def extract_number_word(ocr_text: str) -> str:
    if not ocr_text:
        return None
    s = ocr_text.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    for token in s.split():
        if token in WORD_TO_MODEL:
            return token
    return None


def ocr_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera index", CAMERA_INDEX)
        with state_lock:
            state["quit"] = True
        return

    print("Camera opened. Press 'q' in the camera window to exit.")
    last_word, last_time = None, 0.0

    while True:
        with state_lock:
            if state["quit"]:
                break

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        preview = frame.copy()
        cv2.putText(preview, "Press 'q' to exit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if last_word:
            cv2.putText(preview, f"Detected: {last_word}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("Camera Preview (OCR)", preview)

        if time.time() - last_time > OCR_INTERVAL:
            last_time = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            cx, cy = w // 2, h // 2
            box_w, box_h = int(w * 0.6), int(h * 0.6)
            roi = gray[cy - box_h//2:cy + box_h//2, cx - box_w//2:cx + box_w//2]
            roi = cv2.GaussianBlur(roi, (5, 5), 0)
            _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            try:
                ocr_result = pytesseract.image_to_string(roi, config="--psm 6")
            except Exception as e:
                print("Tesseract error:", e)
                ocr_result = ""

            word = extract_number_word(ocr_result)
            if word and word != last_word:
                print(f"OCR found: '{word}' (raw: {ocr_result.strip()!r})")
                mapped = WORD_TO_MODEL.get(word)
                with state_lock:
                    state["current_word"] = word
                    if mapped and os.path.isfile(mapped):
                        state["current_model_path"] = mapped
                        state["model_changed"] = True
                    else:
                        print(f"⚠️ Model for '{word}' not found: {mapped}")
                last_word = word

        if cv2.waitKey(1) & 0xFF == ord('q'):
            with state_lock:
                state["quit"] = True
            break

    cap.release()
    cv2.destroyAllWindows()
    print("OCR thread exiting.")


# -------------------- OpenGL Rendering --------------------
def load_obj_mesh(path):
    """Load .obj using pywavefront properly."""
    try:
        scene = pywavefront.Wavefront(path, parse=True, collect_faces=True)
        vertex_lists = []
        for name, mesh in scene.meshes.items():
            for face in mesh.faces:
                vertex_lists.append([tuple(scene.vertices[idx]) for idx in face])
        return vertex_lists, scene
    except Exception as e:
        print(f"Failed to load OBJ '{path}':", e)
        return None, None


def display():
    global rotation_angle, gl_mesh, last_load_path
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    with state_lock:
        model_path = state["current_model_path"]
        model_changed = state["model_changed"]
        if model_changed:
            state["model_changed"] = False

    if model_path != last_load_path and model_path:
        data, scene = load_obj_mesh(model_path)
        if data:
            gl_mesh = (data, scene)
            last_load_path = model_path
            print("✅ Loaded model:", model_path)
        else:
            gl_mesh = None
            last_load_path = None

    if gl_mesh is None:
        glPushMatrix()
        glRotatef(rotation_angle, 0, 1, 0)
        glutWireCube(1.0)
        glPopMatrix()
    else:
        vertex_lists, scene = gl_mesh
        glPushMatrix()

        # Smooth rotation
        glRotatef(rotation_angle, 0.0, 1.0, 0.0)

        # ✅ FIX: rotate 180° around X-axis to correct inverted hand model
        glRotatef(180.0, 1.0, 0.0, 0.0)

        try:
            verts = np.array(scene.vertices)
            minv, maxv = verts.min(axis=0), verts.max(axis=0)
            size = max(maxv - minv)
            scale = 1.2 / size if size > 0 else 1.0
            center = (minv + maxv) / 2.0
            glTranslatef(-center[0]*scale, -center[1]*scale, -center[2]*scale)
            glScalef(scale, scale, scale)
        except Exception:
            pass

        # ✅ Enable lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)

        # ✅ Set pink color material
        pink = [1.0, 0.75, 0.8, 1.0]       # RGBA
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, pink)
        glColor4f(1.0, 0.75, 0.8, 1.0)     # backup color for non-lit faces

        # Draw all triangles
        glBegin(GL_TRIANGLES)
        for tri in vertex_lists:
            try:
                v0, v1, v2 = map(np.array, tri)
                normal = np.cross(v1 - v0, v2 - v0)
                if np.linalg.norm(normal) > 0:
                    normal /= np.linalg.norm(normal)
                glNormal3f(*normal)
            except Exception:
                pass
            for v in tri:
                glVertex3f(*v)
        glEnd()

        glDisable(GL_LIGHTING)
        glPopMatrix()

    rotation_angle += 0.8
    glutSwapBuffers()


def reshape(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def keyboard(key, x, y):
    if key in [b'\x1b', b'q']:
        with state_lock:
            state["quit"] = True
        glutLeaveMainLoop()
        sys.exit(0)


def gl_mainloop():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(*WINDOW_SIZE)
    glutCreateWindow(WINDOW_TITLE.encode('utf-8'))

    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)
    glClearColor(0.9, 0.9, 0.9, 1.0)

    glLightfv(GL_LIGHT0, GL_POSITION, (4, 4, 4, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.7, 0.7, 0.7, 1.0))

    glutDisplayFunc(display)
    glutIdleFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)

    try:
        glutMainLoop()
    except SystemExit:
        pass


# -------------------- Entry Point --------------------
def main():
    if not os.path.isdir(MODEL_FOLDER):
        print(f"Creating model folder: {MODEL_FOLDER}")
        os.makedirs(MODEL_FOLDER, exist_ok=True)

    t = threading.Thread(target=ocr_loop, daemon=True)
    t.start()

    try:
        gl_mainloop()
    except KeyboardInterrupt:
        with state_lock:
            state["quit"] = True
        print("Interrupted, quitting.")

    t.join(timeout=1.0)
    print("Exiting program.")


if __name__ == "__main__":
    main()
