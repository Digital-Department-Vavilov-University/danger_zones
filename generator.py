"""
Генератор синтетических тестовых изображений.

Камера установлена спереди робота и смотрит вниз, поэтому опасные зоны
выглядят как разметка на дороге: длинные полосы — сплошные, прерывистые
или в форме угла (L). Полосы рисуются на «ровном» виде сверху, а затем
(опционально) применяется перспективное преобразование, имитирующее
наклон камеры — дальний край пола сужается к верху кадра.

Сам генератор отвечает и за сохранение результатов: метод
`generate_dataset` создаёт указанное число тестовых картинок и
складывает их в `output_dir`.
"""

import math
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np


# BGR-цвета для рисования зон
_WHITE = (245, 245, 245)
_RED = (35, 35, 220)


class TestImageGenerator:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        seed: Optional[int] = None,
        output_dir: str = "generated_images",
    ):
        self.width = width
        self.height = height
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.output_dir = output_dir

    # ---------- фон ----------

    def _make_floor(self) -> np.ndarray:
        # Базовый серый фон с пиксельным шумом
        base = self._np_rng.integers(
            85, 130, size=(self.height, self.width, 3), dtype=np.uint8
        )
        # Низкочастотные пятна освещения
        small = self._np_rng.integers(
            0, 35, size=(self.height // 16, self.width // 16, 3), dtype=np.uint8
        )
        big = cv2.resize(small, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return cv2.add(base, big)

    # ---------- геометрия полос ----------

    def _random_stripe_endpoints(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Возвращает два конца полосы, заведомо пересекающей рабочую область.
        Стиль выбирается случайно:
          lateral      — почти горизонтальная (поперёк хода робота),
          longitudinal — почти вертикальная (вдоль хода),
          diagonal     — по диагонали из угла в угол.
        """
        style = self._rng.choices(
            ["lateral", "longitudinal", "diagonal"], weights=[2, 2, 1]
        )[0]
        w, h = self.width, self.height

        if style == "lateral":
            y = self._rng.randint(80, h - 80)
            wobble = self._rng.randint(-25, 25)
            x1 = self._rng.randint(-30, w // 4)
            x2 = self._rng.randint(3 * w // 4, w + 30)
            return (x1, y), (x2, y + wobble)

        if style == "longitudinal":
            x = self._rng.randint(80, w - 80)
            wobble = self._rng.randint(-25, 25)
            y1 = self._rng.randint(-30, h // 4)
            y2 = self._rng.randint(3 * h // 4, h + 30)
            return (x + wobble, y1), (x, y2)

        # diagonal — между двумя противоположными углами
        if self._rng.random() < 0.5:
            x1 = self._rng.randint(-30, w // 3)
            y1 = self._rng.randint(-30, h // 3)
            x2 = self._rng.randint(2 * w // 3, w + 30)
            y2 = self._rng.randint(2 * h // 3, h + 30)
        else:
            x1 = self._rng.randint(2 * w // 3, w + 30)
            y1 = self._rng.randint(-30, h // 3)
            x2 = self._rng.randint(-30, w // 3)
            y2 = self._rng.randint(2 * h // 3, h + 30)
        return (x1, y1), (x2, y2)

    # ---------- виды разметки ----------

    def _draw_solid_stripe(self, img: np.ndarray, color_bgr: Tuple[int, int, int]) -> None:
        p1, p2 = self._random_stripe_endpoints()
        thickness = self._rng.randint(18, 38)
        cv2.line(img, p1, p2, color_bgr, thickness, cv2.LINE_AA)

    def _draw_dashed_stripe(self, img: np.ndarray, color_bgr: Tuple[int, int, int]) -> None:
        p1, p2 = self._random_stripe_endpoints()
        thickness = self._rng.randint(18, 32)
        dash_len = self._rng.randint(35, 65)
        gap_len = self._rng.randint(20, 45)
        self._draw_dashed_line(img, p1, p2, color_bgr, thickness, dash_len, gap_len)

    @staticmethod
    def _draw_dashed_line(
        img: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color_bgr: Tuple[int, int, int],
        thickness: int,
        dash_len: int,
        gap_len: int,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        step = dash_len + gap_len
        d = 0.0
        while d < length:
            seg_end = min(d + dash_len, length)
            sx, sy = int(x1 + ux * d), int(y1 + uy * d)
            ex, ey = int(x1 + ux * seg_end), int(y1 + uy * seg_end)
            cv2.line(img, (sx, sy), (ex, ey), color_bgr, thickness, cv2.LINE_AA)
            d += step

    def _draw_corner(self, img: np.ndarray, color_bgr: Tuple[int, int, int]) -> None:
        """L-образный маркер: две перпендикулярные полосы из общей точки."""
        cx = self._rng.randint(120, self.width - 120)
        cy = self._rng.randint(120, self.height - 120)
        arm1 = self._rng.randint(90, 180)
        arm2 = self._rng.randint(90, 180)
        thickness = self._rng.randint(18, 30)
        base_angle = self._rng.uniform(0, 2 * math.pi)
        perp_angle = base_angle + math.pi / 2
        end1 = (
            int(cx + arm1 * math.cos(base_angle)),
            int(cy + arm1 * math.sin(base_angle)),
        )
        end2 = (
            int(cx + arm2 * math.cos(perp_angle)),
            int(cy + arm2 * math.sin(perp_angle)),
        )
        cv2.line(img, (cx, cy), end1, color_bgr, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), end2, color_bgr, thickness, cv2.LINE_AA)

    def _draw_marking(self, img: np.ndarray, color_bgr: Tuple[int, int, int]) -> None:
        kind = self._rng.choices(
            ["solid", "dashed", "corner"], weights=[4, 2, 1]
        )[0]
        if kind == "solid":
            self._draw_solid_stripe(img, color_bgr)
        elif kind == "dashed":
            self._draw_dashed_stripe(img, color_bgr)
        else:
            self._draw_corner(img, color_bgr)

    # ---------- основные генераторы ----------

    def generate_topdown(self, n_white: int = 1, n_red: int = 1) -> np.ndarray:
        """Вид строго сверху (без перспективы)."""
        img = self._make_floor()
        for _ in range(n_white):
            self._draw_marking(img, _WHITE)
        for _ in range(n_red):
            self._draw_marking(img, _RED)
        # Лёгкий блюр + шум, чтобы не было идеальных границ
        img = cv2.GaussianBlur(img, (3, 3), 0)
        noise = self._np_rng.integers(-8, 9, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def generate_with_perspective(
        self, n_white: int = 1, n_red: int = 1, tilt: float = 0.30
    ) -> np.ndarray:
        """
        Имитация камеры спереди робота, направленной вниз.
        tilt ∈ (0, 0.5) — насколько сужается дальний край.
        """
        flat = self.generate_topdown(n_white, n_red)
        margin = int(self.width * tilt)
        src = np.float32(
            [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]
        )
        dst = np.float32(
            [
                [margin, 0],
                [self.width - margin, 0],
                [self.width, self.height],
                [0, self.height],
            ]
        )
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            flat, M, (self.width, self.height), borderValue=(55, 65, 70)
        )

    # ---------- сохранение ----------

    def generate_dataset(
        self,
        count: int = 10,
        *,
        output_dir: Optional[str] = None,
        perspective: bool = True,
        max_white: int = 2,
        max_red: int = 2,
    ) -> List[str]:
        """
        Сгенерировать `count` изображений и сохранить их в `output_dir`
        (по умолчанию — `self.output_dir`). Возвращает список путей к
        созданным файлам. Имя файла кодирует количество зон каждого
        цвета и режим съёмки, например: sample_003_w1_r2_p.png
        (1 белая, 2 красные, режим с перспективой).
        """
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        paths: List[str] = []
        for i in range(count):
            n_white = self._rng.randint(0, max_white)
            n_red = self._rng.randint(0, max_red)
            # гарантируем хотя бы одну зону, иначе тест бессмыслен
            if n_white + n_red == 0:
                if self._rng.random() < 0.5:
                    n_white = 1
                else:
                    n_red = 1

            if perspective:
                img = self.generate_with_perspective(n_white, n_red)
                tag = "p"
            else:
                img = self.generate_topdown(n_white, n_red)
                tag = "t"

            name = f"sample_{i:03d}_w{n_white}_r{n_red}_{tag}.png"
            path = os.path.join(out, name)
            cv2.imwrite(path, img)
            paths.append(path)
        return paths
