"""
Детектор опасных зон по цвету (белый / красный).

Используется HSV-пространство:
- Белый: низкая насыщенность, высокая яркость.
- Красный "оборачивается" вокруг H=0, поэтому используется два диапазона.

Камера установлена спереди робота и смотрит вниз, поэтому объекты
в нижней части кадра ближе к роботу. Для каждой зоны вычисляется
proximity ∈ [0, 1] — приближённая мера близости (1 = у самого робота).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np


@dataclass
class DangerZone:
    color: str                            # 'white' или 'red'
    contour: np.ndarray                   # контур зоны
    bbox: Tuple[int, int, int, int]       # x, y, w, h
    area: float                           # площадь в пикселях
    center: Tuple[int, int]               # центроид
    proximity: float                      # 0..1, чем больше — тем ближе к роботу


class DangerZoneDetector:
    def __init__(
        self,
        min_area: int = 500,
        # Параметры белого
        white_s_max: int = 40,
        white_v_min: int = 200,
        # Параметры красного (две зоны H, т. к. красный оборачивается вокруг 0)
        red_s_min: int = 110,
        red_v_min: int = 80,
        red_h1_max: int = 10,
        red_h2_min: int = 170,
        # Морфология
        morph_kernel: int = 5,
    ):
        self.min_area = min_area
        self.white_s_max = white_s_max
        self.white_v_min = white_v_min
        self.red_s_min = red_s_min
        self.red_v_min = red_v_min
        self.red_h1_max = red_h1_max
        self.red_h2_min = red_h2_min
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )

    # ---------- маски ----------

    def _white_mask(self, hsv: np.ndarray) -> np.ndarray:
        lower = np.array([0, 0, self.white_v_min], dtype=np.uint8)
        upper = np.array([179, self.white_s_max, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def _red_mask(self, hsv: np.ndarray) -> np.ndarray:
        lower1 = np.array([0, self.red_s_min, self.red_v_min], dtype=np.uint8)
        upper1 = np.array([self.red_h1_max, 255, 255], dtype=np.uint8)
        lower2 = np.array([self.red_h2_min, self.red_s_min, self.red_v_min], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(m1, m2)

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        # Open убирает мелкий шум, close заполняет дыры внутри зоны
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    # ---------- извлечение зон ----------

    def _zones_from_mask(
        self, mask: np.ndarray, color: str, img_h: int
    ) -> List[DangerZone]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        zones: List[DangerZone] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Берём низ bbox: это самая ближняя к роботу часть зоны.
            proximity = float((y + h) / img_h)
            proximity = max(0.0, min(1.0, proximity))
            zones.append(
                DangerZone(
                    color=color,
                    contour=cnt,
                    bbox=(x, y, w, h),
                    area=float(area),
                    center=(cx, cy),
                    proximity=proximity,
                )
            )
        return zones

    # ---------- публичный API ----------

    def detect(self, image_bgr: np.ndarray) -> List[DangerZone]:
        """Вернуть список зон, отсортированный по убыванию близости к роботу."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        white = self._clean(self._white_mask(hsv))
        red = self._clean(self._red_mask(hsv))

        h = image_bgr.shape[0]
        zones = self._zones_from_mask(white, "white", h) + self._zones_from_mask(
            red, "red", h
        )
        zones.sort(key=lambda z: z.proximity, reverse=True)
        return zones

    def detect_with_masks(
        self, image_bgr: np.ndarray
    ) -> Tuple[List[DangerZone], Dict[str, np.ndarray]]:
        """То же, что detect(), плюс возвращает сырые маски (полезно для отладки)."""
        blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        white = self._clean(self._white_mask(hsv))
        red = self._clean(self._red_mask(hsv))
        h = image_bgr.shape[0]
        zones = self._zones_from_mask(white, "white", h) + self._zones_from_mask(
            red, "red", h
        )
        zones.sort(key=lambda z: z.proximity, reverse=True)
        return zones, {"white": white, "red": red}

    # ---------- визуализация ----------

    @staticmethod
    def visualize(image_bgr: np.ndarray, zones: List[DangerZone]) -> np.ndarray:
        vis = image_bgr.copy()
        for z in zones:
            outline = (255, 255, 255) if z.color == "white" else (0, 0, 255)
            cv2.drawContours(vis, [z.contour], -1, outline, 2)
            x, y, w, h = z.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), outline, 1)
            cv2.circle(vis, z.center, 4, outline, -1)
            label = f"{z.color} prox={z.proximity:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 4, y - 4)
            cv2.rectangle(vis, (x, ty - th - 3), (x + tw + 4, ty + 2), (0, 0, 0), -1)
            cv2.putText(
                vis, label, (x + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline, 1, cv2.LINE_AA,
            )
        return vis
