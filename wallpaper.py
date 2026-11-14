"""Sprite colorization, ZIP archiving, and wallpaper composition."""

import io
import math
import zipfile

from PIL import Image

from extractor import CANVAS_SIZE, SpriteGrid

RGBColor = tuple[int, int, int]
RGBAColor = tuple[int, int, int, int]

PRESET_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1080p Full HD (1920x1080)": (1920, 1080),
    "1440p QHD (2560x1440)": (2560, 1440),
    "4K UHD (3840x2160)": (3840, 2160),
    "UltrawideUWQHD (3440x1440)": (3440, 1440),
    "Mobile Portrait (1080x2400)": (1080, 2400),
}


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


def render_sprite(
    grid: SpriteGrid,
    palette: list[RGBColor],
    scale: int = 1,
    transparent_bg: bool = True,
) -> Image.Image:
    """Renders a 2-bit index grid to an RGBA PIL Image with pixel scaling."""
    img = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE))
    flat_pixels: list[RGBAColor] = []

    c0: RGBAColor = (
        (palette[0][0], palette[0][1], palette[0][2], 0)
        if transparent_bg
        else (*palette[0], 255)
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
    selected_dex: list[int],
    resolution: tuple[int, int],
    bg_hex: str,
    palette: list[RGBColor],
    columns: int,
    scale: int,
    spacing: int,
) -> Image.Image:
    """Composes selected Pokemon sprites onto a centered grid wallpaper."""
    width, height = resolution
    bg_rgb = hex_to_rgb(bg_hex)
    wallpaper = Image.new("RGBA", (width, height), (*bg_rgb, 255))

    count = len(selected_dex)
    if count == 0:
        return wallpaper.convert("RGB")

    cols = max(1, min(columns, count))
    rows = math.ceil(count / cols)
    cell_size = CANVAS_SIZE * scale

    grid_w = cols * cell_size + (cols - 1) * spacing
    grid_h = rows * cell_size + (rows - 1) * spacing

    start_x = (width - grid_w) // 2
    start_y = (height - grid_h) // 2

    for idx, dex_num in enumerate(selected_dex):
        row_idx = idx // cols
        col_idx = idx % cols

        x = start_x + col_idx * (cell_size + spacing)
        y = start_y + row_idx * (cell_size + spacing)

        sprite_img = render_sprite(
            sprites[dex_num],
            palette=palette,
            scale=scale,
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
    palette: list[RGBColor],
    scale: int = 1,
    transparent_bg: bool = True,
) -> bytes:
    """Packages all 151 rendered sprites into an in-memory ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for dex_num, grid in sprites.items():
            img = render_sprite(grid, palette, scale, transparent_bg)
            png_bytes = image_to_png_bytes(img)
            zf.writestr(f"{dex_num:03d}.png", png_bytes)
    return buf.getvalue()
