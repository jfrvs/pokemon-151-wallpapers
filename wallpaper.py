"""Sprite colorization, color clustering, and space-filling composition."""

import colorsys
import io
import math
import zipfile
from dataclasses import dataclass
from PIL import Image

from extractor import CANVAS_SIZE, RGBColor, SpriteGrid

RGBAColor = tuple[int, int, int, int]

ASPECT_RATIOS: dict[str, dict[str, tuple[int, int]]] = {
    "16:9 (Widescreen)": {
        "1080p Full HD (1920x1080)": (1920, 1080),
        "1440p QHD (2560x1440)": (2560, 1440),
        "4K UHD (3840x2160)": (3840, 2160),
    },
    "16:10 (Laptop / Display)": {
        "WUXGA (1920x1200)": (1920, 1200),
        "WQXGA (2560x1600)": (2560, 1600),
    },
    "21:9 (Ultrawide)": {
        "UW-FHD (2560x1080)": (2560, 1080),
        "UW-QHD (3440x1440)": (3440, 1440),
    },
    "4:3 (Classic / Tablet)": {
        "UXGA (1600x1200)": (1600, 1200),
        "QXGA (2048x1536)": (2048, 1536),
    },
    "9:16 (Mobile Portrait)": {
        "FHD+ Mobile (1080x1920)": (1080, 1920),
        "QHD+ Mobile (1440x2560)": (1440, 2560),
    },
    "1:1 (Square)": {
        "1080x1080 Square": (1080, 1080),
        "2048x2048 Square": (2048, 2048),
    },
}

CHANNEL_SWAPS: dict[str, tuple[int, int, int]] = {
    "RGB (Default)": (0, 1, 2),
    "BGR (Blue <-> Red)": (2, 1, 0),
    "RBG (Green <-> Blue)": (0, 2, 1),
    "GRB (Red <-> Green)": (1, 0, 2),
    "BRG (Cycle Right)": (2, 0, 1),
    "GBR (Cycle Left)": (1, 2, 0),
}

COLOR_CATEGORIES: tuple[str, ...] = (
    "Red",
    "Orange / Brown",
    "Yellow",
    "Green",
    "Cyan / Teal",
    "Blue",
    "Purple / Pink",
    "Neutral",
)

ROW_ALIGN_MODES: tuple[str, ...] = (
    "Evenly Distribute Across Canvas",
    "Justify Per Row (Full Horizontal Span)",
    "Uniform Grid Pitch (Centered)",
)

REMAINDER_MODES: tuple[str, ...] = (
    "Centered (Contiguous)",
    "Top Heavy (Contiguous)",
    "Bottom Heavy (Contiguous)",
    "Alternate (Centered)",
    "Alternate (Top Heavy)",
    "Alternate (Bottom Heavy)",
)


@dataclass(frozen=True)
class RowArrangement:
    """Represents a row-driven, space-filling layout configuration."""

    rows: int
    min_cols: int
    max_cols: int
    auto_scale: int
    fill_pct: float
    score: float

    @property
    def label(self) -> str:
        """Human-readable label for the row arrangement selector."""
        if self.min_cols == self.max_cols:
            col_desc = f"{self.max_cols} per row"
        else:
            col_desc = f"{self.min_cols} to {self.max_cols} per row"
        return (
            f"{self.rows} Rows ({col_desc}) — "
            f"{self.auto_scale}x Scale ({self.fill_pct:.0f}% area fill)"
        )


def hex_to_rgb(hex_str: str) -> RGBColor:
    """Converts a #RRGGBB hex string into an (R, G, B) tuple."""
    clean = hex_str.lstrip("#")
    return (
        int(clean[0:2], 16),
        int(clean[2:4], 16),
        int(clean[4:6], 16),
    )


def build_interpolated_palette(
    bg_hex: str,
    fg_hex: str,
) -> list[RGBColor]:
    """Generates a 4-color gradient from background (0) to foreground (3)."""
    bg = hex_to_rgb(bg_hex)
    fg = hex_to_rgb(fg_hex)
    palette: list[RGBColor] = []

    for step in range(4):
        ratio = step / 3.0
        r = int(round(bg[0] + (fg[0] - bg[0]) * ratio))
        g = int(round(bg[1] + (fg[1] - bg[1]) * ratio))
        b = int(round(bg[2] + (fg[2] - bg[2]) * ratio))
        palette.append((r, g, b))

    return palette


def transform_color(
    color: RGBColor,
    channel_order: tuple[int, int, int] = (0, 1, 2),
    hue_shift: int = 0,
    invert: bool = False,
) -> RGBColor:
    """Applies channel swapping, hue rotation, and inversion to an RGB tuple."""
    r, g, b = (
        color[channel_order[0]],
        color[channel_order[1]],
        color[channel_order[2]],
    )

    if hue_shift != 0:
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        h = (h + (hue_shift / 360.0)) % 1.0
        rf, gf, bf = colorsys.hsv_to_rgb(h, s, v)
        r = int(round(rf * 255))
        g = int(round(gf * 255))
        b = int(round(bf * 255))

    if invert:
        r, g, b = 255 - r, 255 - g, 255 - b

    return (r, g, b)


def transform_palette_map(
    palettes: dict[int, list[RGBColor]],
    channel_order: tuple[int, int, int] = (0, 1, 2),
    hue_shift: int = 0,
    invert: bool = False,
    transform_bg_slot: bool = False,
) -> dict[int, list[RGBColor]]:
    """Applies global color transformations across all 151 Pokemon palettes."""
    out: dict[int, list[RGBColor]] = {}
    for dex_num, pal in palettes.items():
        new_pal: list[RGBColor] = []
        for idx, color in enumerate(pal):
            if idx == 0 and not transform_bg_slot:
                new_pal.append(color)
            else:
                new_pal.append(
                    transform_color(color, channel_order, hue_shift, invert)
                )
        out[dex_num] = new_pal
    return out


def classify_rgb_hue(r: float, g: float, b: float) -> str:
    """Maps normalized (0..255) RGB values into a dominant color family."""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if s < 0.18 or v < 0.15:
        return "Neutral"

    deg = h * 360.0
    if deg < 18.0 or deg >= 340.0:
        return "Red"
    if deg < 42.0:
        return "Orange / Brown"
    if deg < 72.0:
        return "Yellow"
    if deg < 160.0:
        return "Green"
    if deg < 200.0:
        return "Cyan / Teal"
    if deg < 265.0:
        return "Blue"
    return "Purple / Pink"


def classify_sprite_color(
    grid: SpriteGrid,
    palette: list[RGBColor],
) -> str:
    """Determines the dominant foreground color category of a sprite."""
    counts = [0, 0, 0, 0]
    for row in grid:
        for idx in row:
            if idx != 0:
                counts[idx] += 1

    weights = (0.0, 1.0, 1.0, 0.25)
    total_w = 0.0
    r_sum = 0.0
    g_sum = 0.0
    b_sum = 0.0

    for idx in (1, 2, 3):
        _, s, v = colorsys.rgb_to_hsv(
            palette[idx][0] / 255.0,
            palette[idx][1] / 255.0,
            palette[idx][2] / 255.0,
        )
        chroma_boost = 1.0 + (s * 2.0) if v > 0.15 else 0.1
        w = counts[idx] * weights[idx] * chroma_boost
        total_w += w
        r_sum += palette[idx][0] * w
        g_sum += palette[idx][1] * w
        b_sum += palette[idx][2] * w

    if total_w == 0.0:
        return "Neutral"

    return classify_rgb_hue(
        r_sum / total_w,
        g_sum / total_w,
        b_sum / total_w,
    )


def categorize_all_sprites(
    sprites: dict[int, SpriteGrid],
    palettes: dict[int, list[RGBColor]],
) -> dict[str, list[int]]:
    """Groups Pokedex numbers by dominant foreground color category."""
    groups: dict[str, list[int]] = {cat: [] for cat in COLOR_CATEGORIES}
    for dex_num, grid in sprites.items():
        cat = classify_sprite_color(grid, palettes[dex_num])
        groups[cat].append(dex_num)
    return groups


def distribute_row_counts(
    total: int,
    rows: int,
    placement: str,
) -> list[int]:
    """Splits total items across rows using contiguous or alternating modes."""
    base = total // rows
    rem = total % rows
    if rem == 0:
        return [base] * rows

    counts = [base] * rows

    if placement == "Top Heavy (Contiguous)":
        for idx in range(rem):
            counts[idx] += 1
        return counts

    if placement == "Bottom Heavy (Contiguous)":
        for idx in range(rows - rem, rows):
            counts[idx] += 1
        return counts

    if placement == "Centered (Contiguous)":
        start = (rows - rem) // 2
        for idx in range(start, start + rem):
            counts[idx] += 1
        return counts

    if placement == "Alternate (Top Heavy)":
        order = [*range(0, rows, 2), *range(1, rows, 2)]
        for idx in order[:rem]:
            counts[idx] += 1
        return counts

    if placement == "Alternate (Bottom Heavy)":
        order = [*range(rows - 1, -1, -2), *range(rows - 2, -1, -2)]
        for idx in order[:rem]:
            counts[idx] += 1
        return counts

    # Alternate (Centered): interleave wider rows within the middle span
    span = min(rows, 2 * rem - 1)
    start = (rows - span) // 2
    sub_indices = list(range(start, start + span))
    alt_sub = [*sub_indices[0::2], *sub_indices[1::2]]
    for idx in alt_sub[:rem]:
        counts[idx] += 1
    return counts


def compute_row_arrangements(
    count: int,
    resolution: tuple[int, int],
    margin_pct: int,
) -> list[RowArrangement]:
    """Generates row-based arrangements ranked by screen space utilization."""
    if count <= 0:
        return []

    width, height = resolution
    pad_x = int(width * (margin_pct / 100.0))
    pad_y = int(height * (margin_pct / 100.0))
    avail_w = max(1, width - 2 * pad_x)
    avail_h = max(1, height - 2 * pad_y)
    target_ratio = avail_w / avail_h

    arrangements: list[RowArrangement] = []

    for rows in range(1, count + 1):
        max_cols = math.ceil(count / rows)
        min_cols = count // rows

        scale_x = avail_w // (max_cols * CANVAS_SIZE)
        scale_y = avail_h // (rows * CANVAS_SIZE)
        auto_scale = int(min(scale_x, scale_y))
        if auto_scale < 1:
            continue

        cell_area = (CANVAS_SIZE * auto_scale) ** 2
        fill_pct = min(100.0, (count * cell_area) / (avail_w * avail_h) * 100)
        grid_ratio = max_cols / rows
        ratio_diff = abs(math.log(grid_ratio / target_ratio))
        score = ratio_diff - (fill_pct / 200.0)

        arrangements.append(
            RowArrangement(
                rows=rows,
                min_cols=min_cols,
                max_cols=max_cols,
                auto_scale=auto_scale,
                fill_pct=fill_pct,
                score=score,
            )
        )

    arrangements.sort(key=lambda item: item.score)
    return arrangements


def compute_axis_positions(
    count: int,
    max_count: int,
    cell_size: int,
    avail_span: int,
    start_offset: int,
    mode: str,
) -> list[int]:
    """Calculates 1D coordinates for items along an axis to fill space."""
    if count <= 0:
        return []
    if count == 1:
        return [start_offset + (avail_span - cell_size) // 2]

    leftover = max(0, avail_span - count * cell_size)

    if mode == "Justify Per Row (Full Horizontal Span)":
        step = (avail_span - cell_size) / (count - 1)
        return [int(round(start_offset + i * step)) for i in range(count)]

    if mode == "Uniform Grid Pitch (Centered)":
        max_leftover = max(0, avail_span - max_count * cell_size)
        gap = max_leftover / (max_count + 1)
        row_span = count * cell_size + (count - 1) * gap
        row_start = start_offset + (avail_span - row_span) / 2.0
        return [
            int(round(row_start + i * (cell_size + gap)))
            for i in range(count)
        ]

    gap = leftover / (count + 1)
    return [
        int(round(start_offset + gap + i * (cell_size + gap)))
        for i in range(count)
    ]


def render_sprite(
    grid: SpriteGrid,
    palette: list[RGBColor],
    scale: int = 1,
    transparent_bg: bool = True,
    bg_override: RGBColor | None = None,
) -> Image.Image:
    """Renders a 2-bit index grid to an RGBA PIL Image with pixel scaling."""
    img = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE))
    flat_pixels: list[RGBAColor] = []

    bg_rgb = bg_override if bg_override is not None else palette[0]
    c0: RGBAColor = (
        (bg_rgb[0], bg_rgb[1], bg_rgb[2], 0)
        if transparent_bg
        else (*bg_rgb, 255)
    )
    c1: RGBAColor = (*palette[1], 255)
    c2: RGBAColor = (*palette[2], 255)
    c3: RGBAColor = (*palette[3], 255)
    color_map = (c0, c1, c2, c3)

    for row in grid:
        for idx in row:
            flat_pixels.append(color_map[idx])

    img.putdata(flat_pixels)

    if scale > 1:
        target_size = (CANVAS_SIZE * scale, CANVAS_SIZE * scale)
        img = img.resize(target_size, Image.Resampling.NEAREST)

    return img


def generate_wallpaper(
    sprites: dict[int, SpriteGrid],
    palettes: dict[int, list[RGBColor]],
    selected_dex: list[int],
    resolution: tuple[int, int],
    bg_rgb: RGBColor,
    rows: int,
    scale: int,
    margin_pct: int,
    row_align_mode: str,
    remainder_mode: str,
) -> Image.Image:
    """Composes sprites to fill the entire canvas across balanced rows."""
    width, height = resolution
    wallpaper = Image.new("RGBA", (width, height), (*bg_rgb, 255))

    count = len(selected_dex)
    if count == 0:
        return wallpaper.convert("RGB")

    active_rows = max(1, min(rows, count))
    row_counts = distribute_row_counts(count, active_rows, remainder_mode)
    max_cols = max(row_counts)

    pad_x = int(width * (margin_pct / 100.0))
    pad_y = int(height * (margin_pct / 100.0))
    avail_w = max(1, width - 2 * pad_x)
    avail_h = max(1, height - 2 * pad_y)

    safe_scale = max(
        1,
        min(
            avail_w // (max_cols * CANVAS_SIZE),
            avail_h // (active_rows * CANVAS_SIZE),
        ),
    )
    active_scale = max(1, min(scale, int(safe_scale)))
    cell_size = CANVAS_SIZE * active_scale

    y_positions = compute_axis_positions(
        count=active_rows,
        max_count=active_rows,
        cell_size=cell_size,
        avail_span=avail_h,
        start_offset=pad_y,
        mode="Evenly Distribute Across Canvas",
    )

    cursor = 0
    for row_idx, r_count in enumerate(row_counts):
        row_items = selected_dex[cursor : cursor + r_count]
        cursor += r_count
        y = y_positions[row_idx]

        x_positions = compute_axis_positions(
            count=r_count,
            max_count=max_cols,
            cell_size=cell_size,
            avail_span=avail_w,
            start_offset=pad_x,
            mode=row_align_mode,
        )

        for col_idx, dex_num in enumerate(row_items):
            x = x_positions[col_idx]
            sprite_img = render_sprite(
                sprites[dex_num],
                palette=palettes[dex_num],
                scale=active_scale,
                transparent_bg=True,
            )
            wallpaper.alpha_composite(sprite_img, dest=(x, y))

    return wallpaper.convert("RGB")


def image_to_png_bytes(img: Image.Image) -> bytes:
    """Serializes a PIL Image into a PNG byte buffer."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_sprites_zip(
    sprites: dict[int, SpriteGrid],
    palettes: dict[int, list[RGBColor]],
    selected_dex: list[int],
    scale: int = 1,
    transparent_bg: bool = True,
    bg_override: RGBColor | None = None,
) -> bytes:
    """Packages selected rendered sprites into an in-memory ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for dex_num in selected_dex:
            img = render_sprite(
                sprites[dex_num],
                palettes[dex_num],
                scale,
                transparent_bg,
                bg_override=bg_override,
            )
            png_bytes = image_to_png_bytes(img)
            zf.writestr(f"{dex_num:03d}.png", png_bytes)
    return buf.getvalue()
