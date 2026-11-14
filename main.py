"""Streamlit interface for extracting sprites and building wallpapers."""

import streamlit as st

from extractor import SpriteGrid, extract_all_indices
from wallpaper import (
    PRESET_RESOLUTIONS,
    RGBColor,
    build_interpolated_palette,
    create_sprites_zip,
    generate_wallpaper,
    hex_to_rgb,
    image_to_png_bytes,
    render_sprite,
)


@st.cache_data(show_spinner="Decompressing 151 sprites from ROM...")
def cached_extract(
    rom_bytes: bytes,
    back_sprites: bool,
) -> dict[int, SpriteGrid]:
    """Caches raw 2-bit sprite grids in memory per uploaded ROM."""
    return extract_all_indices(rom_bytes, back_sprites=back_sprites)


def render_sidebar() -> dict[str, object]:
    """Renders sidebar configuration controls and returns user settings."""
    with st.sidebar:
        st.header("1. ROM Input")
        rom_file = st.file_uploader(
            "Upload Pokemon Yellow (.gbc / .gb)",
            type=["gb", "gbc"],
        )
        back_sprites = st.toggle("Extract Back Sprites", value=False)

        st.header("2. Resolution & Grid")
        res_choice = st.selectbox(
            "Resolution Preset",
            options=[*list(PRESET_RESOLUTIONS.keys()), "Custom"],
        )
        if res_choice == "Custom":
            col_w, col_h = st.columns(2)
            width = col_w.number_input("Width", 320, 7680, 1920, 10)
            height = col_h.number_input("Height", 240, 4320, 1080, 10)
            resolution = (int(width), int(height))
        else:
            resolution = PRESET_RESOLUTIONS[str(res_choice)]

        columns = st.slider("Grid Columns", 1, 25, 16)
        scale = st.slider("Sprite Scale (Nearest Neighbor)", 1, 6, 2)
        spacing = st.slider("Grid Spacing (px)", 0, 64, 8)

        st.header("3. Color Palette")
        color_mode = st.radio(
            "Palette Mode",
            options=["2-Color Auto Gradient", "4-Color Custom GBC"],
            horizontal=True,
        )

        if color_mode == "2-Color Auto Gradient":
            c_bg, c_fg = st.columns(2)
            bg_hex = c_bg.color_picker("Background", "#0F380F")
            fg_hex = c_fg.color_picker("Foreground", "#9BBC0F")
            palette = build_interpolated_palette(bg_hex, fg_hex)
        else:
            c0, c1, c2, c3 = st.columns(4)
            bg_hex = c0.color_picker("BG (0)", "#FFFFFF")
            tone1 = c1.color_picker("Light (1)", "#F8A858")
            tone2 = c2.color_picker("Dark (2)", "#D84020")
            tone3 = c3.color_picker("Ink (3)", "#000000")
            palette = [
                hex_to_rgb(bg_hex),
                hex_to_rgb(tone1),
                hex_to_rgb(tone2),
                hex_to_rgb(tone3),
            ]

    return {
        "rom_file": rom_file,
        "back_sprites": back_sprites,
        "resolution": resolution,
        "columns": columns,
        "scale": scale,
        "spacing": spacing,
        "bg_hex": bg_hex,
        "palette": palette,
    }


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="PokeWall Generator",
        page_icon=":material/wallpaper:",
        layout="wide",
    )
    st.title("Pokemon Yellow Sprite & Wallpaper Studio")
    st.caption(
        "Upload a legally acquired Pokemon Yellow ROM to decompress all "
        "151 2BPP bitplane sprites in-memory and build custom wallpapers."
    )

    cfg = render_sidebar()
    rom_file = cfg["rom_file"]

    if rom_file is None:
        st.info("Upload your Pokemon Yellow ROM in the sidebar to begin.")
        return

    try:
        sprites = cached_extract(
            rom_file.getvalue(),
            bool(cfg["back_sprites"]),
        )
    except ValueError as err:
        st.error(str(err))
        return

    tab_wall, tab_dex = st.tabs(
        [":material/image: Wallpaper Studio", ":material/grid_view: Sprites"]
    )

    palette: list[RGBColor] = cfg["palette"]  # type: ignore[assignment]
    res: tuple[int, int] = cfg["resolution"]  # type: ignore[assignment]
    scale = int(cfg["scale"])  # type: ignore[arg-type]

    with tab_wall:
        dex_range = st.slider(
            "Pokedex Range to Include",
            min_value=1,
            max_value=151,
            value=(1, 151),
        )
        selected_dex = list(range(dex_range[0], dex_range[1] + 1))

        wall_img = generate_wallpaper(
            sprites=sprites,
            selected_dex=selected_dex,
            resolution=res,
            bg_hex=str(cfg["bg_hex"]),
            palette=palette,
            columns=int(cfg["columns"]),  # type: ignore[arg-type]
            scale=scale,
            spacing=int(cfg["spacing"]),  # type: ignore[arg-type]
        )

        st.image(
            wall_img,
            caption=f"Preview ({res[0]}x{res[1]} px, {len(selected_dex)} "
            "Pokemon)",
            width='stretch',
        )

        png_data = image_to_png_bytes(wall_img)
        st.download_button(
            label="Download Wallpaper PNG",
            data=png_data,
            file_name=f"pokewall_{res[0]}x{res[1]}.png",
            mime="image/png",
            type="primary",
        )

    with tab_dex:
        col_dl, col_trans = st.columns([1, 2])
        transparent_zip = col_trans.checkbox(
            "Transparent background in ZIP export",
            value=True,
        )
        zip_bytes = create_sprites_zip(
            sprites=sprites,
            palette=palette,
            scale=scale,
            transparent_bg=transparent_zip,
        )
        col_dl.download_button(
            label="Download All 151 Sprites (.zip)",
            data=zip_bytes,
            file_name="pokemon_yellow_sprites.zip",
            mime="application/zip",
        )

        grid_ui = st.grid(columns=8)
        for dex_num, s_grid in sprites.items():
            sprite_preview = render_sprite(
                s_grid,
                palette=palette,
                scale=2,
                transparent_bg=False,
            )
            grid_ui.image(sprite_preview, caption=f"#{dex_num:03d}")


if __name__ == "__main__":
    main()
