# Bullseye detection model

`python -m tools.train`을 실행하면 이 폴더에 `bullseye.pt`가 생깁니다.
용량 때문에 `*.pt` 파일은 git에 올리지 않습니다.

- 구조: Ultralytics YOLO11n detection (`yolo11n.pt` COCO 사전학습에서 fine-tune)
- 클래스: `0 = bullseye`
- 학습 데이터: `tools/make_synthetic.py`로 만든 합성 과녁 이미지
  (또는 Roboflow 키가 있으면 `tools/download_datasets.py`로 받은 실사 데이터)
- 기본 학습 크기: 320px

합성 데이터로만 학습한 모델이라, 실제 카메라와 환경에서의 정확도는 따로 확인해야 합니다.
