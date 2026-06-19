import os
import tempfile
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

# ---- constants ---------------------------------------------------------------
DPI = 300
MM = DPI / 25.4
CARD_W = round(85.60 * MM)
CARD_H = round(53.98 * MM)
A4_W   = round(210 * MM)
A4_H   = round(297 * MM)
TARGET_AR = 85.60 / 53.98
CORNER_R = round(3.18 * MM)
GUIDE = (200, 200, 200)


# ---- input loading -----------------------------------------------------------
def load_pages(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import fitz
        doc = fitz.open(path)
        out = []
        for p in doc:
            pix = p.get_pixmap(dpi=DPI)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
            out.append(img)
        return out
    else:
        data = np.fromfile(path, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return [img]


# ---- card detection ----------------------------------------------------------
def _order(p):
    r = np.zeros((4, 2), "float32"); s = p.sum(1); d = np.diff(p, 1)
    r[0] = p[np.argmin(s)]; r[2] = p[np.argmax(s)]
    r[1] = p[np.argmin(d)]; r[3] = p[np.argmax(d)]
    return r

def _edge_maps(gray):
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    return [
        cv2.Canny(g, 15, 60),
        cv2.Canny(g, 30, 120),
        cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY_INV, 51, 10),
        cv2.threshold(g, 230, 255, cv2.THRESH_BINARY_INV)[1],
    ]

def detect_quads(img):
    H, W = img.shape[:2]; A = H * W
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cand = []
    for e in _edge_maps(gray):
        e = cv2.dilate(e, np.ones((7, 7), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.02 * A:
                continue
            peri = cv2.arcLength(c, True)
            ap = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(ap) != 4:
                continue
            q = ap.reshape(4, 2)
            (w, h) = cv2.minAreaRect(q)[1]
            if w < 1 or h < 1:
                continue
            if abs(max(w, h) / min(w, h) - TARGET_AR) < 0.4:
                cand.append((q, area))
    out = []
    for q, a in sorted(cand, key=lambda x: -x[1]):
        cx, cy = q[:, 0].mean(), q[:, 1].mean()
        if all(abs(cx - o[:, 0].mean()) > 250 or abs(cy - o[:, 1].mean()) > 250 for o in out):
            out.append(q)
    out.sort(key=lambda q: (round(q[:, 1].mean() / 300), q[:, 0].mean()))
    return out

def warp_card(img, quad):
    r = _order(quad.astype("float32"))
    (tl, tr, br, bl) = r
    maxw = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    maxh = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if maxw >= maxh:
        ow, oh = CARD_W, CARD_H
    else:
        ow, oh = CARD_H, CARD_W
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], "float32")
    M = cv2.getPerspectiveTransform(r, dst)
    card = cv2.warpPerspective(img, M, (ow, oh))
    if oh > ow:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    if card.shape[1] != CARD_W or card.shape[0] != CARD_H:
        card = cv2.resize(card, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    return card

def cards_from_image(img):
    quads = detect_quads(img)
    if quads:
        return [warp_card(img, q) for q in quads]
    h, w = img.shape[:2]
    if w < h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return [cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)]


# ---- A4 composition ----------------------------------------------------------
def round_corners(card, r=CORNER_R, ss=4):
    h, w = card.shape[:2]
    R = r * ss
    big = np.zeros((h * ss, w * ss), np.uint8)
    cv2.rectangle(big, (R, 0), (w*ss - R, h*ss), 255, -1)
    cv2.rectangle(big, (0, R), (w*ss, h*ss - R), 255, -1)
    for cx, cy in [(R, R), (w*ss - R, R), (R, h*ss - R), (w*ss - R, h*ss - R)]:
        cv2.circle(big, (cx, cy), R, 255, -1)
    alpha = (cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)[..., None]
    white = np.full_like(card, 255, np.float32)
    return (card.astype(np.float32) * alpha + white * (1 - alpha)).astype(np.uint8)

def rounded_guide(canvas, x, y, w, h, r, color):
    cv2.line(canvas, (x+r, y), (x+w-r, y), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (x+r, y+h), (x+w-r, y+h), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y+r), (x, y+h-r), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (x+w, y+r), (x+w, y+h-r), color, 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x+r, y+r),     (r, r), 180, 0, 90, color, 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x+w-r, y+r),   (r, r), 270, 0, 90, color, 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x+w-r, y+h-r), (r, r),   0, 0, 90, color, 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x+r, y+h-r),   (r, r),  90, 0, 90, color, 1, cv2.LINE_AA)

def place(canvas, card, cx, cy):
    card = round_corners(card)
    h, w = card.shape[:2]
    x, y = cx - w // 2, cy - h // 2
    canvas[y:y+h, x:x+w] = card
    rounded_guide(canvas, x-2, y-2, w+3, h+3, CORNER_R, GUIDE)

def build_a4(front, back):
    canvas = np.full((A4_H, A4_W, 3), 255, np.uint8)
    cx = A4_W // 2
    gap = round(20 * MM)
    block = CARD_H * (2 if back is not None else 1) + (gap if back is not None else 0)
    top = (A4_H - block) // 2
    place(canvas, front, cx, top + CARD_H // 2)
    if back is not None:
        place(canvas, back, cx, top + CARD_H + gap + CARD_H // 2)
    return canvas

def process_files(paths):
    cards = []
    for path in paths:
        for page in load_pages(path):
            cards.extend(cards_from_image(page))
    if not cards:
        raise ValueError("No IC cards detected in the uploaded files.")
    pages = []
    for i in range(0, len(cards), 2):
        front = cards[i]
        back = cards[i + 1] if i + 1 < len(cards) else None
        pages.append(build_a4(front, back))
    pil = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in pages]
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    out.close()
    pil[0].save(out.name, "PDF", resolution=DPI, save_all=True, append_images=pil[1:])
    return out.name


# ---- FastAPI app -------------------------------------------------------------
app = FastAPI(title="IC → A4 Converter")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    tmp_inputs = []
    output_path = None
    try:
        for f in files:
            suffix = os.path.splitext(f.filename or "")[1].lower() or ".bin"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(await f.read())
            tmp.close()
            tmp_inputs.append(tmp.name)

        output_path = process_files(tmp_inputs)

    except ValueError as exc:
        for p in tmp_inputs:
            try: os.unlink(p)
            except OSError: pass
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        for p in tmp_inputs:
            try: os.unlink(p)
            except OSError: pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    def cleanup():
        for p in tmp_inputs:
            try: os.unlink(p)
            except OSError: pass
        if output_path:
            try: os.unlink(output_path)
            except OSError: pass

    background_tasks.add_task(cleanup)
    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename="ic_a4_output.pdf",
    )
