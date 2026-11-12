"""YOLO 라벨 변환. 여러 데이터셋의 과녁 클래스를 단일 클래스 0으로 합칩니다.

OpenCV/YOLO 없이도 테스트할 수 있도록 표준 라이브러리만 사용합니다.
"""
import re

KEEP = re.compile(r"bull|target|과녁", re.IGNORECASE)
# 탄흔, 화살 같은 클래스는 과녁 자체가 아니므로 제외합니다.
DROP = re.compile(r"arrow|hole|bullet|shot|impact|hit|화살|탄흔", re.IGNORECASE)


def target_class_ids(names, keep_names=None):
    """data.yaml의 names(list 또는 {id: name})에서 과녁 클래스 번호를 고릅니다.

    keep_names를 주면 이름이 정확히 일치하는 클래스만 사용합니다.
    """
    items = names.items() if isinstance(names, dict) else enumerate(names)
    items = [(int(i), str(n)) for i, n in items]
    if keep_names:
        return {i for i, n in items if n in keep_names}
    return {i for i, n in items if KEEP.search(n) and not DROP.search(n)}


def convert_line(line, keep_ids):
    """라벨 한 줄을 'class cx cy w h'(class=0)로 바꿉니다. 버릴 줄이면 None.

    Roboflow의 YOLO 내보내기에는 bbox(4개 값)와 폴리곤(x y 쌍 3개 이상)이
    섞여 있을 수 있어 폴리곤은 감싸는 bbox로 변환합니다.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if class_id not in keep_ids:
        return None
    if len(values) == 4:
        cx, cy, w, h = values
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def convert_label_text(text, keep_ids):
    lines = (convert_line(line, keep_ids) for line in text.splitlines())
    return [line for line in lines if line]
