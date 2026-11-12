"""합성 과녁 데이터셋을 만듭니다. API 키나 라벨링이 필요 없습니다.

동심원 과녁(사격지, 양궁, 빨강·흰색 등)을 그려 배경 사진 위에 원근 기울기(±45°),
회전, 흐림, 조명 변화와 함께 붙이고, 가장 바깥 원을 감싸는 bbox를 라벨로 저장합니다.

사용법 (프로젝트 폴더에서):
    python -m tools.make_synthetic                    # train 1500장, valid 300장
    python -m tools.make_synthetic --train 3000

배경은 Ultralytics가 공개한 coco128 이미지(약 7MB)를 받아 쓰고, 단색/그라데이션
배경도 섞습니다.
"""
import argparse
import math
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

COCO128 = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
TEXTURE = 512  # 과녁 텍스처 한 변(px)


# ---------- 과녁 그리기 ----------

def rings(image, center, radius, colors, line=None):
    """바깥에서 안쪽으로 colors 순서대로 동심원을 채웁니다."""
    n = len(colors)
    for i, color in enumerate(colors):
        r = max(1, round(radius * (n - i) / n))
        cv2.circle(image, center, r, color, -1, cv2.LINE_AA)
        if line is not None:
            cv2.circle(image, center, r, line, max(1, radius // 150), cv2.LINE_AA)


def draw_target(rng):
    """BGRA 텍스처와 텍스처 안 과녁 원(중심, 반지름)을 반환합니다."""
    size = TEXTURE
    paper = rng.integers(215, 256, 3).tolist()
    image = np.zeros((size, size, 4), np.uint8)
    style = rng.choice(["shooting", "archery", "redwhite", "random"], p=[0.35, 0.25, 0.25, 0.15])
    square_paper = style == "shooting" or rng.random() < 0.6
    margin = rng.uniform(0.03, 0.18) if square_paper else 0.0
    if square_paper:
        image[:] = (*paper, 255)
    center = (size // 2, size // 2)
    radius = round(size / 2 * (1 - margin))
    alpha = np.zeros((size, size), np.uint8)
    cv2.circle(alpha, center, radius, 255, -1, cv2.LINE_AA)
    if not square_paper:
        image[..., 3] = alpha

    bgr = image[..., :3].copy()
    if style == "shooting":
        # 흰 바탕에 검은 선 링, 가운데 검은 원과 그 안의 흰 선 링
        bgr_paper = tuple(paper)
        cv2.circle(bgr, center, radius, bgr_paper, -1, cv2.LINE_AA)
        n = int(rng.integers(8, 11))
        black_from = int(rng.integers(3, 6))
        for i in range(n):
            r = round(radius * (n - i) / n)
            if i == n - black_from:
                cv2.circle(bgr, center, r, (20, 20, 20), -1, cv2.LINE_AA)
            inner = i >= n - black_from
            cv2.circle(bgr, center, r, bgr_paper if inner else (25, 25, 25), 2, cv2.LINE_AA)
            if rng.random() < 0.7 and i < n - 1:
                label = str(i + 1)
                y = center[1] - r + round(radius / n / 2) + 6
                cv2.putText(bgr, label, (center[0] - 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            bgr_paper if inner else (25, 25, 25), 1, cv2.LINE_AA)
    elif style == "archery":
        white, black, blue, red, gold = (240, 240, 240), (30, 30, 30), (200, 140, 20), (40, 40, 220), (40, 210, 250)
        colors = [white, white, black, black, blue, blue, red, red, gold, gold]
        rings(bgr, center, radius, colors, line=(90, 90, 90))
    elif style == "redwhite":
        red = tuple(int(v) for v in rng.integers([0, 0, 170], [60, 60, 256]))
        n = int(rng.integers(3, 7))
        rings(bgr, center, radius, [red if i % 2 == 0 else tuple(paper) for i in range(n)])
    else:
        a, b = rng.integers(0, 256, 3).tolist(), rng.integers(0, 256, 3).tolist()
        n = int(rng.integers(3, 9))
        rings(bgr, center, radius, [a if i % 2 == 0 else b for i in range(n)])
    if rng.random() < 0.5:  # 조준용 십자선
        cv2.line(bgr, (center[0], center[1] - radius // 12), (center[0], center[1] + radius // 12), (30, 30, 30), 1)
        cv2.line(bgr, (center[0] - radius // 12, center[1]), (center[0] + radius // 12, center[1]), (30, 30, 30), 1)
    if rng.random() < 0.3:  # 탄흔/구멍
        for _ in range(int(rng.integers(1, 8))):
            angle, dist = rng.uniform(0, 2 * math.pi), rng.uniform(0, radius * 0.8)
            point = (round(center[0] + dist * math.cos(angle)), round(center[1] + dist * math.sin(angle)))
            cv2.circle(bgr, point, int(rng.integers(3, 8)), (40, 40, 40), -1, cv2.LINE_AA)
    image[..., :3] = bgr
    if not square_paper:
        image[..., 3] = alpha
    return image, center, radius


# ---------- 원근 변환 ----------

def homography(rng, texture_size, diameter, position):
    """텍스처를 3D로 기울였다가 다시 투영하는 호모그래피. 기울기는 ±45° 이내."""
    tilt_x, tilt_y = np.radians(rng.uniform(-45, 45, 2))
    roll = rng.uniform(0, 2 * math.pi)
    c, s = np.cos(tilt_x), np.sin(tilt_x)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(tilt_y), np.sin(tilt_y)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c, s = np.cos(roll), np.sin(roll)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = Rx @ Ry @ Rz
    half = texture_size / 2
    corners = np.array([[0, 0], [texture_size, 0], [texture_size, texture_size], [0, texture_size]], np.float32)
    focal = texture_size * rng.uniform(1.5, 4.0)
    points = []
    for x, y in corners:
        X = R @ np.array([x - half, y - half, 0.0]) + np.array([0, 0, focal])
        points.append([X[0] / X[2] * focal, X[1] / X[2] * focal])
    points = np.array(points, np.float32)
    # 투영 후 크기를 원하는 지름에 맞추고 위치로 옮깁니다.
    scale = diameter / texture_size
    points = points * scale + np.array(position, np.float32)
    return cv2.getPerspectiveTransform(corners, points)


# ---------- 배경과 합성 ----------

def load_backgrounds(cache):
    folder = cache / "coco128"
    if not folder.is_dir():
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / "coco128.zip"
        print(f"배경 이미지 다운로드 → {archive}")
        urllib.request.urlretrieve(COCO128, archive)
        root = cache.resolve()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (root / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError(f"잘못된 압축 경로: {member.filename}")
            z.extractall(root)
        archive.unlink()
    return sorted(p for p in folder.rglob("*.jpg"))


def random_background(rng, backgrounds, width, height):
    if backgrounds and rng.random() < 0.8:
        image = cv2.imread(str(backgrounds[int(rng.integers(len(backgrounds)))]))
        image = cv2.resize(image, (width, height))
        if rng.random() < 0.5:
            image = cv2.flip(image, 1)
        return image
    # 벽이나 책상 같은 단순 배경
    top, bottom = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    t = np.linspace(0, 1, height)[:, None, None]
    image = (top * (1 - t) + bottom * t).astype(np.uint8)
    image = np.repeat(image, width, axis=1)
    noise = rng.normal(0, rng.uniform(2, 12), image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def paste(rng, background, texture, H):
    h, w = background.shape[:2]
    warped = cv2.warpPerspective(texture, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    alpha = warped[..., 3:4].astype(np.float32) / 255
    # 과녁마다 조명 밝기를 다르게
    shade = np.float32(rng.uniform(0.6, 1.15))
    rgb = np.clip(warped[..., :3].astype(np.float32) * shade, 0, 255)
    background[:] = (rgb * alpha + background.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def circle_box(H, center, radius, width, height):
    """투영된 원의 bbox(정규화 cx cy w h). 화면 밖으로 많이 잘리면 None."""
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    circle = np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], 1)
    points = cv2.perspectiveTransform(circle[None].astype(np.float32), H)[0]
    x1, y1 = points.min(0)
    x2, y2 = points.max(0)
    full = (x2 - x1) * (y2 - y1)
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < 0.6 * full:
        return None
    return ((cx1 + cx2) / 2 / width, (cy1 + cy2) / 2 / height, (cx2 - cx1) / width, (cy2 - cy1) / height)


def augment(rng, image):
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5, 7]))
        image = cv2.GaussianBlur(image, (k, k), 0)
    if rng.random() < 0.2:  # 손떨림/움직임 흐림
        k = int(rng.integers(5, 15))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1 / k
        center = (k / 2, k / 2)
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D(center, rng.uniform(0, 180), 1), (k, k))
        image = cv2.filter2D(image, -1, kernel / max(kernel.sum(), 1e-6))
    alpha, beta = rng.uniform(0.6, 1.4), rng.uniform(-40, 40)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if rng.random() < 0.5:
        image = np.clip(image + rng.normal(0, rng.uniform(2, 10), image.shape), 0, 255).astype(np.uint8)
    quality = int(rng.integers(40, 95))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def make_sample(rng, backgrounds, width, height):
    image = random_background(rng, backgrounds, width, height)
    labels = []
    count = 0 if rng.random() < 0.1 else int(rng.choice([1, 1, 1, 2, 3]))  # 10%는 배경만
    for _ in range(count):
        texture, center, radius = draw_target(rng)
        diameter = rng.uniform(0.08, 0.75) * min(width, height)
        position = (rng.uniform(-0.1, 1.1) * width, rng.uniform(-0.1, 1.1) * height)
        H = homography(rng, TEXTURE, diameter, position)
        box = circle_box(H, center, radius, width, height)
        if box is None:
            continue
        paste(rng, image, texture, H)
        labels.append(box)
    return augment(rng, image), labels


def parse_args():
    parser = argparse.ArgumentParser(description="합성 과녁 데이터셋 생성")
    parser.add_argument("--out", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--cache", type=Path, default=Path("datasets/raw"), help="배경 이미지 저장 위치")
    parser.add_argument("--train", type=int, default=1500)
    parser.add_argument("--valid", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-photos", action="store_true", help="coco128 없이 단순 배경만 사용")
    return parser.parse_args()


def main():
    args = parse_args()
    backgrounds = [] if args.no_photos else load_backgrounds(args.cache)
    print(f"배경 사진 {len(backgrounds)}장")
    if args.out.exists():
        shutil.rmtree(args.out)
    rng = np.random.default_rng(args.seed)
    for split, total in (("train", args.train), ("valid", args.valid)):
        (args.out / split / "images").mkdir(parents=True)
        (args.out / split / "labels").mkdir(parents=True)
        boxes = 0
        for i in range(total):
            image, labels = make_sample(rng, backgrounds, args.width, args.height)
            name = f"syn_{split}_{i:05d}"
            cv2.imwrite(str(args.out / split / "images" / f"{name}.jpg"), image)
            text = "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in labels)
            (args.out / split / "labels" / f"{name}.txt").write_text(text, encoding="utf-8")
            boxes += len(labels)
        print(f"{split}: 이미지 {total}장, 과녁 {boxes}개")
    data = dict(path=str(args.out.resolve()), train="train/images", val="valid/images",
                names={0: "bullseye"})
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print("완료:", args.out / "data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
