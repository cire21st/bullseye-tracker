"""병합한 과녁 데이터셋으로 YOLO를 fine-tune하고 models/bullseye.pt로 복사합니다.

사용법 (프로젝트 폴더에서):
    python -m tools.train                 # CPU, yolo11n, 10 epoch, 320px
    python -m tools.train --device 0      # CUDA GPU
"""
import argparse
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="과녁 단일 클래스 YOLO 학습")
    parser.add_argument("--data", default=None,
                        help="기본: datasets/bullseye(공개 데이터 병합)가 있으면 그것, 없으면 datasets/synthetic")
    parser.add_argument("--model", default="yolo11n.pt", help="시작 가중치 (COCO 사전학습)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5, help="검증 성능이 N epoch 동안 안 오르면 중단")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="cpu 또는 CUDA GPU 번호 0")
    parser.add_argument("--name", default="bullseye", help="runs/train/<name>에 결과 저장")
    parser.add_argument("--output", default="models/bullseye.pt", help="최종 모델 복사 위치")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data is None:
        candidates = ["datasets/bullseye/data.yaml", "datasets/synthetic/data.yaml"]
        args.data = next((c for c in candidates if Path(c).is_file()), candidates[-1])
    print(f"데이터셋: {args.data}")
    if not Path(args.data).is_file():
        print(f"{args.data}이 없습니다. 먼저 python -m tools.make_synthetic를 실행하세요.",
              file=sys.stderr)
        return 1
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
                batch=args.batch, workers=args.workers, device=args.device,
                project=str(Path("runs/train").resolve()), name=args.name, exist_ok=True,
                # 과녁은 원형 대칭이라 상하 반전도 유효한 증강입니다.
                fliplr=0.5, flipud=0.5, degrees=10.0)
    best = Path(model.trainer.best)
    if not best.is_file():
        print("학습 결과 best.pt를 찾지 못했습니다.", file=sys.stderr)
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    print(f"\n모델 저장: {output}  (원본: {best})")
    print("실행: python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
