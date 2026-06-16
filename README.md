# Детектор опасных зон по цветовой разметке

Программный комплекс для обнаружения опасных участков рабочей области робота, размеченных белыми или красными полосами на полу. Камера установлена спереди робота и направлена вниз.

---

## Структура проекта

```
danger_zones/
├── detector.py   — класс DangerZoneDetector (обработка изображения)
├── generator.py  — класс TestImageGenerator (генерация тестового набора)
└── demo.py       — скрипт запуска (тесты или режим камеры)
```

---

## Зависимости

```bash
pip install opencv-python numpy
```

---

## Быстрый старт

### Синтетические тесты

```bash
python demo.py
```

Генератор создаёт 10 тестовых изображений в `output/inputs/`, детектор обрабатывает их и сохраняет визуализацию в `output/detected/`.

Пример вывода:
```
Сгенерировано 10 изображений в 'output/inputs'

sample_002_w0_r1_p: найдено зон — 1
  [1] red    area= 4949  center=(328,155)  bbox=(197,145,265,23)  prox=0.35
```

### Режим реальной камеры

```bash
python demo.py --camera 0
```

Захват с камеры с индексом `0`, вывод в окно. Нажмите `q` для выхода.

Можно указать свой каталог для результатов:
```bash
python demo.py --out my_results
```

---

## Использование в коде

### Детектор

```python
import cv2
from detector import DangerZoneDetector

detector = DangerZoneDetector(min_area=500)

frame = cv2.imread("frame.png")
zones = detector.detect(frame)

for z in zones:
    print(f"{z.color}  proximity={z.proximity:.2f}  center={z.center}")

# Отладочная визуализация
vis = detector.visualize(frame, zones)
cv2.imwrite("result.png", vis)
```

Структура `DangerZone`:

| Поле | Тип | Описание |
|------|-----|----------|
| `color` | `str` | `"white"` или `"red"` |
| `contour` | `np.ndarray` | Контур (cv2) |
| `bbox` | `tuple` | `(x, y, w, h)` |
| `area` | `float` | Площадь в пикселях² |
| `center` | `tuple` | `(cx, cy)` по моментам |
| `proximity` | `float` | Близость к роботу, 0…1 (1 = вплотную) |

Список отсортирован по `proximity` по убыванию — первый элемент всегда ближайшая зона.

### Генератор

```python
from generator import TestImageGenerator

gen = TestImageGenerator(seed=42, output_dir="my_dataset")

# Сохранить 20 изображений (с перспективой)
paths = gen.generate_dataset(count=20, perspective=True, max_white=2, max_red=2)

# Или получить numpy-массив без сохранения
img = gen.generate_with_perspective(n_white=1, n_red=1)
img_flat = gen.generate_topdown(n_white=0, n_red=2)
```

Имена сохранённых файлов кодируют параметры сцены:
```
sample_004_w1_r2_p.png
         │   │  └── p = perspective, t = topdown
         │   └────── r2 = 2 красные зоны
         └────────── w1 = 1 белая зона
```

---

## Алгоритм детектирования

```
BGR-кадр
  → GaussianBlur 5×5
  → BGR → HSV
  → маска белого (S ≤ 40, V ≥ 200)      маска красного (H ∈ [0,10] ∪ [170,179], S ≥ 110, V ≥ 80)
  → morph_open + morph_close (ядро 5×5, эллипс)
  → findContours (RETR_EXTERNAL)
  → фильтр по площади (min_area)
  → вычисление bbox, центроида, proximity = (y + h) / H
  → сортировка по proximity ↓
  → List[DangerZone]
```

Параметры класса `DangerZoneDetector`:

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `min_area` | `500` | Минимальная площадь контура (пикс²) |
| `white_s_max` | `40` | Максимальная насыщенность для белого |
| `white_v_min` | `200` | Минимальная яркость для белого |
| `red_s_min` | `110` | Минимальная насыщенность для красного |
| `red_v_min` | `80` | Минимальная яркость для красного |

---

## Виды генерируемой разметки

| Тип | Вес | Описание |
|-----|-----|----------|
| `solid` | 4 | Сплошная линия, толщина 18–38 пкс |
| `dashed` | 2 | Прерывистая: штрих 35–65 пкс, зазор 20–45 пкс |
| `corner` | 1 | L-угол из двух перпендикулярных полос |

Ориентации: `lateral` (поперёк хода), `longitudinal` (вдоль хода), `diagonal`.
