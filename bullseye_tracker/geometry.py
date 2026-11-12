"""영상 좌표 계산. OpenCV/YOLO 없이도 테스트할 수 있습니다."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    name: str
    track_id: int | None = None

    def coordinates(self, width: int, height: int) -> dict[str, float]:
        # 원점은 영상 왼쪽 위입니다. x는 오른쪽, y는 아래쪽으로 증가합니다.
        x1, y1, x2, y2 = self.xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # 반올림은 그릴 때만 수행하여 출력 좌표의 정밀도를 유지합니다.
        return dict(cx=cx, cy=cy, error_x=cx - width / 2, error_y=cy - height / 2)


def pick_primary(detections, previous_track_id=None):
    """추적할 과녁 하나를 고릅니다.

    직전에 따라가던 track ID가 아직 보이면 그 과녁을 유지해 화면 중심 오차가
    과녁 사이를 오가지 않게 합니다. 없으면 신뢰도가 가장 높은 과녁을 고릅니다.
    """
    if not detections:
        return None
    if previous_track_id is not None:
        for detection in detections:
            if detection.track_id == previous_track_id:
                return detection
    return max(detections, key=lambda d: d.confidence)
