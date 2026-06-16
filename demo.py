"""
Демонстрация работы детектора.

Запуск:
    python demo.py             # сгенерировать тесты и прогнать детектор
    python demo.py --camera 0  # читать с веб-камеры (живой режим, 'q' для выхода)
"""

import argparse
import os
import sys
from typing import List

import cv2

from detector import DangerZone, DangerZoneDetector
from generator import TestImageGenerator


def _print_zones(title: str, zones: List[DangerZone]) -> None:
    print(f"\n{title}: найдено зон — {len(zones)}")
    for i, z in enumerate(zones, 1):
        x, y, w, h = z.bbox
        print(
            f"  [{i}] {z.color:5s}  area={int(z.area):5d}  "
            f"center=({z.center[0]},{z.center[1]})  "
            f"bbox=({x},{y},{w},{h})  prox={z.proximity:.2f}"
        )


def run_synthetic_tests(out_root: str) -> None:
    inputs_dir = os.path.join(out_root, "inputs")
    detected_dir = os.path.join(out_root, "detected")
    os.makedirs(detected_dir, exist_ok=True)

    # Генератор сам создаёт каталог и складывает туда картинки
    gen = TestImageGenerator(seed=42, output_dir=inputs_dir)
    detector = DangerZoneDetector(min_area=400)

    # Основной набор: вид с робота (с перспективой)
    paths = gen.generate_dataset(count=8, perspective=True)
    # Плюс пара контрольных без перспективы
    paths += gen.generate_dataset(count=2, perspective=False)

    print(f"Сгенерировано {len(paths)} изображений в '{inputs_dir}'")

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  Не удалось прочитать {path}", file=sys.stderr)
            continue
        zones = detector.detect(img)
        vis = detector.visualize(img, zones)

        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(detected_dir, f"{name}_detected.png"), vis)
        _print_zones(name, zones)

    print(f"\nГотово. Результаты детекции сохранены в '{detected_dir}'")


def run_camera(camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру с индексом {camera_index}", file=sys.stderr)
        sys.exit(1)
    detector = DangerZoneDetector(min_area=500)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            zones = detector.detect(frame)
            vis = detector.visualize(frame, zones)
            if zones:
                nearest = zones[0]
                cv2.putText(
                    vis,
                    f"NEAREST: {nearest.color} prox={nearest.proximity:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Danger zones", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help="Индекс камеры; если не задан — синтетические тесты.")
    parser.add_argument("--out", default="output",
                        help="Корневой каталог для inputs/ и detected/.")
    args = parser.parse_args()

    if args.camera is not None:
        run_camera(args.camera)
    else:
        run_synthetic_tests(args.out)


if __name__ == "__main__":
    main()
