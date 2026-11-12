"""공개 과녁 데이터셋을 받아 단일 클래스(bullseye) YOLO 데이터셋으로 합칩니다.

사용법 (프로젝트 폴더에서):
    python -m tools.download_datasets

기본 데이터셋은 Hugging Face에 있어 API 키 없이 받습니다.
Roboflow API 키(ROBOFLOW_API_KEY 환경 변수 또는 --roboflow-key)를 주면
Roboflow Universe 과녁 데이터셋도 함께 받아 합칩니다.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import yaml

from bullseye_tracker.labels import convert_label_text, target_class_ids

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 키 없이 받는 Hugging Face 데이터셋. revision을 고정해 다시 받아도 같은 데이터가 나옵니다.
# 형식: dict(repo="owner/name", revision="커밋 SHA", license="...")
# 참고: mohamedAli4722/SHOOTING_TARGET_DATASET(c782006)은 이미지와 라벨 파일 이름이 하나도
# 맞지 않아(업로드 오류로 보임) 제외했습니다. 키가 없으면 tools.make_synthetic을 쓰세요.
HUGGINGFACE = []

# Roboflow 키가 있을 때만 받습니다. (workspace, project)
# 로봇 표지판(mdp), 탄흔 근접 촬영, 다트판 데이터셋은 분포가 달라 제외했습니다.
ROBOFLOW = [
    ("aerokle-f3ksd", "bullseye-srxic"),
    ("raph-es6jg", "bullseye-xfdum"),
    ("yolo-eqchw", "target-detection-sfcjg"),
    ("archery-zrbei", "target-and-arrow-detection"),  # 과녁 클래스만 사용, 화살 제외
]
ROBOFLOW_API = "https://api.roboflow.com"


# ---------- Hugging Face ----------

def download_huggingface(repo, revision, raw_dir):
    from huggingface_hub import snapshot_download

    destination = raw_dir / repo.replace("/", "__")
    print(f"  다운로드 중 → {destination} (이미 받은 파일은 건너뜀)")
    snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                      local_dir=destination, max_workers=8)
    return destination


# ---------- Roboflow ----------

def get_json(url, params):
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "bullseye-tracker"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from None


def roboflow_latest_version(workspace, project, api_key):
    info = get_json(f"{ROBOFLOW_API}/{workspace}/{project}", {"api_key": api_key})
    numbers = [int(str(v["id"]).rsplit("/", 1)[-1]) for v in info.get("versions") or [] if "id" in v]
    if not numbers:
        raise RuntimeError("공개된 데이터셋 버전이 없습니다.")
    return max(numbers)


def roboflow_export_link(workspace, project, version, api_key):
    """Roboflow가 내보내기 파일을 만들 때까지 기다렸다가 다운로드 링크를 반환합니다."""
    url = f"{ROBOFLOW_API}/{workspace}/{project}/{version}/yolov8"
    for _ in range(60):
        link = (get_json(url, {"api_key": api_key}).get("export") or {}).get("link")
        if link:
            return link
        time.sleep(5)
    raise RuntimeError("내보내기 파일 생성이 5분 안에 끝나지 않았습니다.")


def safe_extract(zip_path, destination):
    """압축 안의 경로가 destination 밖을 가리키면 거부합니다."""
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"잘못된 압축 경로: {member.filename}")
        archive.extractall(root)


def download_roboflow(workspace, project, api_key, raw_dir):
    version = roboflow_latest_version(workspace, project, api_key)
    destination = raw_dir / f"{project}-v{version}"
    if (destination / "data.yaml").is_file():
        print(f"  이미 받음: {destination}")
        return destination, version
    link = roboflow_export_link(workspace, project, version, api_key)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{project}-v{version}.zip"
    print(f"  버전 {version} 다운로드 중 → {zip_path}")
    urllib.request.urlretrieve(link, zip_path)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    safe_extract(zip_path, destination)
    zip_path.unlink()
    return destination, version


# ---------- 병합 ----------

def is_validation(name, ratio):
    """파일 이름 해시로 나누므로 다시 실행해도 같은 분할이 나옵니다."""
    digest = hashlib.md5(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < ratio


def merge(source, prefix, output, keep_names, val_ratio):
    """source의 train/valid/test를 output으로 복사하며 과녁 라벨만 클래스 0으로 남깁니다."""
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    names = config.get("names") or []
    keep_ids = target_class_ids(names, keep_names)
    print(f"  클래스: {names} → 사용: {[names[i] for i in sorted(keep_ids)] or '없음'}")
    if not keep_ids:
        print("  ! 과녁 클래스를 찾지 못해 건너뜁니다. --classes 옵션으로 이름을 지정하세요.")
        return {}
    has_valid = (source / "valid" / "labels").is_dir() and any((source / "valid" / "labels").iterdir())
    counts, skipped = {}, 0
    for split in ("train", "valid", "test"):
        image_dir = source / split / "images"
        if not image_dir.is_dir():
            continue
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = source / split / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                # Roboflow 내보내기는 배경 이미지도 빈 라벨 파일이 있으므로, 라벨이 없으면
                # 다운로드가 중간에 끊긴 것입니다. 배경으로 잘못 넣지 않게 건너뜁니다.
                skipped += 1
                continue
            text = label.read_text(encoding="utf-8")
            lines = convert_label_text(text, keep_ids)
            target_split = split
            if split == "train" and not has_valid and is_validation(image.name, val_ratio):
                target_split = "valid"
            stem = f"{prefix}__{image.stem}"
            for kind in ("images", "labels"):
                (output / target_split / kind).mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, output / target_split / "images" / f"{stem}{image.suffix}")
            # 과녁이 없는 이미지는 빈 라벨(배경 이미지)로 남겨 오검출을 줄입니다.
            (output / target_split / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            key = (target_split, bool(lines))
            counts[key] = counts.get(key, 0) + 1
    if skipped:
        print(f"  라벨 파일이 없는 이미지 {skipped}장은 건너뜀 (다운로드 미완료)")
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="공개 과녁 데이터셋 다운로드 및 단일 클래스로 병합")
    parser.add_argument("--roboflow-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                        help="있으면 Roboflow 데이터셋도 사용 (기본: 환경 변수 ROBOFLOW_API_KEY)")
    parser.add_argument("--raw", type=Path, default=Path("datasets/raw"), help="원본 데이터 위치")
    parser.add_argument("--out", type=Path, default=Path("datasets/bullseye"), help="병합 데이터셋 위치")
    parser.add_argument("--classes", action="append", default=[], metavar="이름=클래스1,클래스2",
                        help="자동 선택 대신 사용할 과녁 클래스 이름. 이름은 HF repo 또는 Roboflow project")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="valid 분할이 없는 데이터셋에서 검증용으로 뗄 비율")
    parser.add_argument("--offline", action="store_true",
                        help="다운로드 없이 --raw에 이미 받은 파일만 병합 (중간에 멈춘 경우)")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    for item in args.classes:
        name, _, classes = item.partition("=")
        overrides[name] = {c.strip() for c in classes.split(",") if c.strip()}

    if args.out.exists():
        shutil.rmtree(args.out)  # 이전 병합 결과와 섞이지 않게 매번 새로 만듭니다.
    sources, totals = [], {}

    def add(source, name, prefix, info):
        counts = merge(source, prefix, args.out, overrides.get(name), args.val_ratio)
        if counts:
            info["images"] = {f"{s}/{'target' if t else 'background'}": n
                              for (s, t), n in sorted(counts.items())}
            sources.append(info)
            for key, n in counts.items():
                totals[key] = totals.get(key, 0) + n

    for item in HUGGINGFACE:
        print(f"\n[huggingface: {item['repo']}]")
        try:
            if args.offline:
                source = args.raw / item["repo"].replace("/", "__")
                if not (source / "data.yaml").is_file():
                    print("  받은 파일 없음, 건너뜀")
                    continue
            else:
                source = download_huggingface(item["repo"], item["revision"], args.raw)
        except Exception as exc:
            print(f"  ! 실패, 건너뜀: {exc}")
            continue
        add(source, item["repo"], item["repo"].split("/")[-1].lower(),
            dict(url=f"https://huggingface.co/datasets/{item['repo']}", **item))

    if args.roboflow_key and not args.offline:
        for workspace, project in ROBOFLOW:
            print(f"\n[roboflow: {workspace}/{project}]")
            try:
                source, version = download_roboflow(workspace, project, args.roboflow_key, args.raw)
            except Exception as exc:
                print(f"  ! 실패, 건너뜀: {exc}")
                continue
            add(source, project, project,
                dict(url=f"https://universe.roboflow.com/{workspace}/{project}", version=version))
    else:
        print("\nRoboflow 키가 없어 Roboflow 데이터셋은 건너뜁니다 (선택 사항).")

    def count(split):
        return totals.get((split, True), 0) + totals.get((split, False), 0)

    if not count("train") or not count("valid"):
        print("\n병합된 train/valid 이미지가 없습니다. 위 메시지를 확인하세요.", file=sys.stderr)
        return 1
    data = dict(path=str(args.out.resolve()), train="train/images", val="valid/images",
                names={0: "bullseye"})
    if count("test"):
        data["test"] = "test/images"
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                        encoding="utf-8")
    (args.out / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\n병합 완료:", args.out / "data.yaml")
    for split in ("train", "valid", "test"):
        print(f"  {split}: 과녁 {totals.get((split, True), 0)}장, 배경 {totals.get((split, False), 0)}장")
    print("출처와 라이선스는 sources.json에 기록했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
