# Bullseye Tracker

웹캠에서 과녁(bullseye)을 검출하고 ByteTrack으로 추적하는 테스트 프로젝트입니다.
[`uav-tracker-test`](../uav-tracker-test)의 구조(카메라 → 검출/추적 → 좌표 → 화면/콘솔)를 그대로 두고, 검출 대상을 UAV에서 과녁으로 바꿨습니다.

학습·테스트용이며 실제 운용이나 안전이 중요한 시스템용이 아닙니다.

## 현재 상태

| 항목 | 상태 |
| --- | --- |
| 코드 / 단위 테스트 | 완료, 테스트 17개 통과 |
| 합성 데이터셋 (`datasets/synthetic/`) | 생성됨: train 1500장(과녁 1296개), valid 300장(과녁 265개) |
| 학습된 모델 (`models/bullseye.pt`) | **아직 없음.** 첫 학습을 시작했다가 중단했습니다 |
| Roboflow 실사 데이터셋 | 받지 않음 (API 키 필요) |

다음 할 일: `python -m tools.train`을 실행해 `models/bullseye.pt`를 만든 뒤 `python main.py`로 확인합니다.

## UAV 버전에서 바뀐 점

- **대상과 모델**: 기본 모델이 UAV 모델에서 직접 학습하는 `models/bullseye.pt`(단일 클래스 `0 = bullseye`)로 바뀌었습니다. 모델이 없으면 `main.py`가 학습 방법을 안내하고 종료합니다.
- **주 과녁**: 여러 과녁이 보이면 하나를 빨간 상자로 표시하고 화면 중심과 선으로 잇습니다. 직전에 따라가던 track ID가 남아 있으면 그 과녁을 유지하고, 없으면 신뢰도가 가장 높은 과녁을 고릅니다. JSON 출력에는 `primary` 필드가 추가됐습니다.
- **추론 크기**: CPU 기준으로 기본 `--imgsz`를 640에서 320으로 줄여 학습 크기와 맞췄습니다.
- **데이터 준비·학습 도구**: `tools/` 폴더에 새로 추가했습니다.

## 폴더 구조

```text
bullseye-tracker/
├─ main.py                     실행 진입점
├─ bullseye_tracker/
│  ├─ app.py                   실행 옵션, 전체 흐름
│  ├─ camera.py                웹캠 입력 (OpenCV)
│  ├─ detector.py              YOLO 검출/추적 → Detection 목록
│  ├─ display.py               bbox, 중심점, 주 과녁 표시
│  ├─ geometry.py              중심 좌표·오차 계산, 주 과녁 선택
│  └─ labels.py                공개 데이터셋 라벨 → 단일 클래스 변환
├─ tools/
│  ├─ make_synthetic.py        합성 과녁 데이터셋 생성 (키 필요 없음)
│  ├─ download_datasets.py     Roboflow 데이터셋 다운로드·병합 (키 필요)
│  └─ train.py                 YOLO fine-tune → models/bullseye.pt
├─ tests/                      단위 테스트 (OpenCV/YOLO 없이 실행)
├─ datasets/                   생성·다운로드한 데이터 (git 제외)
├─ models/                     학습된 모델 (git 제외)
└─ runs/                       학습 기록·그래프 (git 제외)
```

## 설치

GPU가 없는 Windows PC와 Python 3.10에서 확인했습니다.

```powershell
cd C:\code\bullseye-tracker
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 1. 데이터셋 만들기 (키 필요 없음)

```powershell
.\.venv\Scripts\python.exe -m tools.make_synthetic
```

동심원 과녁을 프로그램으로 그려 배경 사진 위에 붙이고, 라벨(가장 바깥 원을 감싸는 bbox)도 자동으로 만듭니다.

- 과녁 종류: 사격지(검은 중심, 링 번호), 양궁(노랑·빨강·파랑·검정·흰색), 빨강·흰색 교차, 무작위 색. 사각 종이 또는 원형으로 그리고, 일부에는 십자선과 탄흔을 넣습니다.
- 변형: 원근 기울기 ±45°, 회전, 크기(화면 짧은 변의 8~75%), 화면 가장자리 잘림, 밝기·대비, 가우시안 흐림, 움직임 흐림, 노이즈, JPEG 압축을 무작위로 줍니다.
- 배경: Ultralytics가 공개한 [coco128](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip) 128장(약 7MB, `datasets/raw/`에 자동 다운로드)과 단색·그라데이션 배경을 씁니다.
- 기본값은 train 1500장, valid 300장이며, 10%는 과녁 없는 배경 이미지입니다. 장수는 `--train 3000 --valid 500`처럼 바꿀 수 있고, `--seed`가 같으면 같은 데이터가 나옵니다.

### 공개 실사 데이터셋 조사 결과

- **Roboflow Universe**: 과녁 데이터셋은 있지만 공개 데이터셋도 API 키가 있어야 받을 수 있습니다.
- **Hugging Face**: 키 없이 받을 수 있는 과녁 데이터셋은 [`mohamedAli4722/SHOOTING_TARGET_DATASET`](https://huggingface.co/datasets/mohamedAli4722/SHOOTING_TARGET_DATASET)(MIT) 정도였습니다. 그런데 커밋 `c782006` 기준으로 이미지와 라벨 파일 이름이 하나도 맞지 않아 제외했습니다. 완전히 받은 test 분할(이미지 362장, 라벨 362개)에서도 일치하는 쌍이 0개였습니다.
- **공개 학습 모델**: 과녁을 검출하는 공개 YOLO `.pt` 모델은 찾지 못했습니다. COCO 기본 모델에는 과녁 클래스가 없습니다.

### (선택) Roboflow 실사 데이터셋

Roboflow 무료 계정을 만든 뒤 Settings → API Keys에서 키를 복사하세요. 키를 주면 아래 데이터셋을 받아 `datasets/bullseye/`에 단일 클래스로 합칩니다.

- [BULLSEYE](https://universe.roboflow.com/aerokle-f3ksd/bullseye-srxic), [Bullseye (raph)](https://universe.roboflow.com/raph-es6jg/bullseye-xfdum), [target-detection](https://universe.roboflow.com/yolo-eqchw/target-detection-sfcjg), [Target and Arrow detection](https://universe.roboflow.com/archery-zrbei/target-and-arrow-detection) (과녁 클래스만 사용)
- 로봇 표지판용(`mdp`), 탄흔 근접 촬영, 다트판 데이터셋은 분포가 달라 넣지 않았습니다.

```powershell
$env:ROBOFLOW_API_KEY = "여기에_키"
.\.venv\Scripts\python.exe -m tools.download_datasets
```

- 과녁 클래스는 이름으로 자동 선택합니다. `bull`, `target`이 들어간 이름을 고르고, `arrow`, `hole`, `bullet` 등은 제외합니다. 자동 선택이 틀리면 `--classes bullseye-xfdum=Bullseye,board`처럼 직접 지정하세요.
- 폴리곤 라벨은 bbox로 바꿉니다. valid 분할이 없는 데이터셋은 train에서 15%를 떼어 검증용으로 씁니다.
- 다운로드가 중간에 끊겼으면 `--offline`으로 이미 받은 것만 병합할 수 있습니다. 라벨이 없는 이미지는 건너뜁니다.
- 출처는 `datasets/bullseye/sources.json`에 기록됩니다. 각 데이터셋의 라이선스는 Universe 페이지에서 확인하세요.
- `datasets/bullseye/`가 있으면 학습할 때 합성 데이터 대신 이 폴더를 사용합니다. 두 데이터를 섞지는 않습니다.

## 2. 학습

```powershell
.\.venv\Scripts\python.exe -m tools.train              # CPU
.\.venv\Scripts\python.exe -m tools.train --device 0   # NVIDIA GPU
```

- `yolo11n.pt`(COCO 사전학습, 자동 다운로드)에서 시작해 fine-tune하고, 가장 좋은 가중치를 `models/bullseye.pt`로 복사합니다.
- GPU 없는 PC 기준으로 기본값을 작게 잡았습니다: `10 epoch`, `320px`, `batch 16`. 검증 성능이 5 epoch 동안 오르지 않으면 일찍 멈춥니다.
- 과녁은 원형 대칭이라 좌우·상하 반전과 ±10° 회전 증강을 씁니다.
- 학습 기록과 그래프는 `runs/train/bullseye/`에 저장됩니다.
- 멀리 있는 작은 과녁까지 잡아야 하면 `--imgsz 480 --epochs 20`으로 학습하고, 실행할 때도 `--imgsz 480`을 주세요.

`models/bullseye.pt`만 있으면 실행할 수 있습니다. 다른 PC에서는 이 파일만 복사하면 다시 학습할 필요가 없습니다.

## 3. 실행

```powershell
.\.venv\Scripts\python.exe main.py
```

기본 설정:

- 모델: `models/bullseye.pt`
- 추적기: ByteTrack
- 추론 크기: 320
- 카메라: 장치 0 (DirectShow)
- 종료: `Q` 또는 `Esc`

```powershell
.\.venv\Scripts\python.exe main.py --camera 1     # 다른 카메라
.\.venv\Scripts\python.exe main.py --conf 0.15    # 검출 임계값 낮추기
.\.venv\Scripts\python.exe main.py --mode detect  # 추적 없이 검출만
.\.venv\Scripts\python.exe main.py --help         # 전체 옵션
```

콘솔 출력(기본 10프레임마다, `--print-every`로 변경):

```json
{"frame": 10, "width": 1280, "height": 720, "objects": [
  {"class_id": 0, "name": "bullseye", "confidence": 0.87, "track_id": 1, "primary": true,
   "cx": 702.5, "cy": 341.0, "error_x": 62.5, "error_y": -19.0}]}
```

- 좌표의 원점은 영상 왼쪽 위입니다. `error_x`, `error_y`는 화면 중심에서 과녁 중심까지의 픽셀 오차입니다(오른쪽·아래가 +).
- `objects`가 비어 있으면 그 프레임에서 과녁을 찾지 못한 것입니다. 이전 좌표를 재사용하지 않습니다.

## 시연 팁

- 모델은 합성 데이터로만 학습되므로, 학습에 쓴 것과 비슷한 **선명한 동심원 과녁을 A3 정도로 출력**해서 쓰면 가장 잘 잡힙니다.
- 과녁은 카메라 정면 ±45° 안에서 움직이세요. 각도가 크면 원이 납작한 타원이 되어 bbox가 불안정해집니다.
- 실제 환경에서 정확도가 부족하면, 시연 장소에서 웹캠으로 찍은 사진 200~300장을 라벨링해 섞으세요. 공개 데이터 1000장보다 효과가 큽니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

좌표 계산, 주 과녁 선택, 라벨 변환(클래스 선택, 폴리곤 → bbox, 범위 자르기)을 확인합니다.
