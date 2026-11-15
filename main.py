"""Streamlit interface with gated Step 3 preview and Step 4 layout studio."""

import streamlit as st

from extractor import RGBColor, RomData, SpriteGrid, extract_rom_data
from wallpaper import (
    ASPECT_RATIOS,
    CHANNEL_SWAPS,
    COLOR_CATEGORIES,
    REMAINDER_MODES,
    ROW_ALIGN_MODES,
    build_interpolated_palette,
    categorize_all_sprites,
    compute_row_arrangements,
    create_sprites_zip,
    generate_wallpaper,
    hex_to_rgb,
    image_to_png_bytes,
    render_sprite,
    transform_color,
    transform_palette_map,
)


@st.cache_data(show_spinner="Extracting sprites and palettes from ROM...")
def cached_extract(
    rom_bytes: bytes,
    back_sprites: bool,
) -> RomData:
    """Caches raw 2-bit sprite grids and GBC palettes in memory."""
    return extract_rom_data(rom_bytes, back_sprites=back_sprites)


def go_to_step(step_num: int) -> None:
    """Updates the active workflow step in Streamlit session state."""
    st.session_state["step"] = step_num


def render_color_sidebar() -> dict[str, object]:
    """Renders Steps 1 and 2 (ROM upload and color styling) in the sidebar."""
    with st.sidebar:
        st.header("Step 1: ROM Input")
        rom_file = st.file_uploader(
            "Upload Pokemon Yellow (.gbc / .gb)",
            type=["gb", "gbc"],
        )
        back_sprites = st.toggle("Extract Back Sprites", value=False)

        st.header("Step 2: Palettes & Transforms")
        color_mode = st.radio(
            "Base Palette Mode",
            options=[
                "Original GBC (Per-Pokemon)",
                "2-Color Auto Gradient",
                "4-Color Custom GBC",
            ],
        )

        uniform_palette: list[RGBColor] | None = None
        if color_mode == "Original GBC (Per-Pokemon)":
            bg_hex = st.color_picker("Wallpaper Background", "#1E1E24")
        elif color_mode == "2-Color Auto Gradient":
            c_bg, c_fg = st.columns(2)
            bg_hex = c_bg.color_picker("Background", "#0F380F")
            fg_hex = c_fg.color_picker("Foreground", "#9BBC0F")
            uniform_palette = build_interpolated_palette(bg_hex, fg_hex)
        else:
            c0, c1, c2, c3 = st.columns(4)
            bg_hex = c0.color_picker("BG (0)", "#1E1E24")
            tone1 = c1.color_picker("Light (1)", "#F8A858")
            tone2 = c2.color_picker("Dark (2)", "#D84020")
            tone3 = c3.color_picker("Ink (3)", "#000000")
            uniform_palette = [
                hex_to_rgb(bg_hex),
                hex_to_rgb(tone1),
                hex_to_rgb(tone2),
                hex_to_rgb(tone3),
            ]

        st.subheader("Global Color Operations")
        swap_label = st.selectbox(
            "RGB Channel Swap",
            options=list(CHANNEL_SWAPS.keys()),
        )
        hue_shift = st.slider("Hue Shift (Degrees)", -180, 180, 0, 5)
        invert_colors = st.toggle("Invert Sprite Palettes", value=False)
        transform_bg = st.toggle(
            "Apply Operations to Background",
            value=False,
        )

    return {
        "rom_file": rom_file,
        "back_sprites": back_sprites,
        "bg_hex": bg_hex,
        "uniform_palette": uniform_palette,
        "channel_order": CHANNEL_SWAPS[str(swap_label)],
        "hue_shift": int(hue_shift),
        "invert_colors": bool(invert_colors),
        "transform_bg": bool(transform_bg),
    }


def build_active_palettes(
    cfg: dict[str, object],
    rom_palettes: dict[int, list[RGBColor]],
) -> tuple[dict[int, list[RGBColor]], RGBColor]:
    """Resolves the transformed per-Pokemon palettes and background color."""
    uniform_pal = cfg["uniform_palette"]
    if uniform_pal is None:
        base_map = rom_palettes
    else:
        pal_list: list[RGBColor] = uniform_pal  # type: ignore[assignment]
        base_map = {dex: list(pal_list) for dex in range(1, 152)}

    chan_order: tuple[int, int, int] = cfg[
        "channel_order"
    ]  # type: ignore[assignment]
    hue_shift = int(cfg["hue_shift"])  # type: ignore[arg-type]
    invert = bool(cfg["invert_colors"])
    transform_bg = bool(cfg["transform_bg"])

    transformed_map = transform_palette_map(
        palettes=base_map,
        channel_order=chan_order,
        hue_shift=hue_shift,
        invert=invert,
        transform_bg_slot=transform_bg,
    )

    bg_rgb = hex_to_rgb(str(cfg["bg_hex"]))
    if transform_bg:
        bg_rgb = transform_color(bg_rgb, chan_order, hue_shift, invert)

    return transformed_map, bg_rgb


def render_step3_subset(
    sprites: dict[int, SpriteGrid],
    active_palettes: dict[int, list[RGBColor]],
    bg_rgb: RGBColor,
) -> None:
    """Renders Step 3: Dominant color subset selection and final-look tiles."""
    st.subheader("Step 3: Select Subset & Preview Final Sprite Colors")
    st.caption(
        "Each sprite below is categorized by its post-transform foreground "
        "color and rendered directly onto your chosen wallpaper background."
    )

    color_groups = categorize_all_sprites(sprites, active_palettes)
    non_empty_cats = [cat for cat in COLOR_CATEGORIES if color_groups[cat]]

    col_mode, col_cats = st.columns([1, 2])
    subset_mode = col_mode.radio(
        "Selection Scope",
        options=["All 151 Pokemon", "Filter by Dominant Color"],
    )

    if subset_mode == "All 151 Pokemon":
        selected_cats = non_empty_cats
    else:
        cat_labels = {
            cat: f"{cat} ({len(color_groups[cat])})" for cat in non_empty_cats
        }
        chosen_labels = col_cats.multiselect(
            "Dominant Foreground Colors (Excludes Background)",
            options=list(cat_labels.values()),
            default=list(cat_labels.values())[:1],
        )
        inv_map = {v: k for k, v in cat_labels.items()}
        selected_cats = [inv_map[lbl] for lbl in chosen_labels]

    selected_dex: list[int] = []
    for cat in selected_cats:
        selected_dex.extend(color_groups[cat])
    selected_dex.sort()

    st.session_state["selected_dex"] = selected_dex

    with st.expander(
        f"Preview Individual Sprites on Wallpaper Background "
        f"({len(selected_dex)} Pokemon)",
        expanded=True,
    ):
        if not selected_dex:
            st.warning("Select at least one color category above.")
        else:
            preview_grid = st.grid(columns=10)
            for dex_num in selected_dex:
                img = render_sprite(
                    sprites[dex_num],
                    palette=active_palettes[dex_num],
                    scale=2,
                    transparent_bg=False,
                    bg_override=bg_rgb,
                )
                preview_grid.image(img, caption=f"#{dex_num:03d}")

    if selected_dex:
        st.button(
            f"Lock Selection ({len(selected_dex)} Pokemon) & "
            "Proceed to Step 4 \u2192",
            type="primary",
            use_container_width=True,
            on_click=go_to_step,
            args=(4,),
        )


def render_step4_wallpaper(
    sprites: dict[int, SpriteGrid],
    active_palettes: dict[int, list[RGBColor]],
    bg_rgb: RGBColor,
    selected_dex: list[int],
) -> None:
    """Renders Step 4: Full-canvas row arrangement and wallpaper generation."""
    col_back, col_title = st.columns([1, 3])
    col_back.button(
        "\u2190 Back to Step 3 (Subset & Preview)",
        on_click=go_to_step,
        args=(3,),
        use_container_width=True,
    )
    col_title.subheader(
        f"Step 4: Space-Filling Layout ({len(selected_dex)} Pokemon)"
    )

    c_ar, c_res, c_margin = st.columns(3)
    ar_choice = c_ar.selectbox(
        "Aspect Ratio",
        options=[*list(ASPECT_RATIOS.keys()), "Custom"],
    )

    if ar_choice == "Custom":
        cw, ch = c_res.columns(2)
        width = int(cw.number_input("Width (px)", 320, 7680, 1920, 10))
        height = int(ch.number_input("Height (px)", 240, 4320, 1080, 10))
        resolution = (width, height)
    else:
        presets = ASPECT_RATIOS[str(ar_choice)]
        res_label = c_res.selectbox(
            "Target Resolution",
            options=list(presets.keys()),
        )
        resolution = presets[str(res_label)]

    margin_pct = int(
        c_margin.slider("Outer Canvas Margin (%)", 0, 15, 2)
    )

    arrangements = compute_row_arrangements(
        count=len(selected_dex),
        resolution=resolution,
        margin_pct=margin_pct,
    )

    if not arrangements:
        st.error(
            "Resolution is too small to fit all selected sprites without "
            "clipping. Increase resolution or reduce outer margin."
        )
        return

    c_rows, c_align, c_rem, c_scale = st.columns([2, 2, 1.5, 1])
    arr_map = {arr.label: arr for arr in arrangements}
    chosen_label = c_rows.selectbox(
        "Row Configuration (Ranked by Screen Fill & Aspect Ratio)",
        options=list(arr_map.keys()),
    )
    chosen_arr = arr_map[str(chosen_label)]

    row_align = str(
        c_align.selectbox(
            "Horizontal Space Distribution",
            options=list(ROW_ALIGN_MODES),
        )
    )

    has_uneven_rows = chosen_arr.min_cols != chosen_arr.max_cols
    remainder_mode = str(
        c_rem.selectbox(
            "Uneven Row Placement",
            options=list(REMAINDER_MODES),
            disabled=not has_uneven_rows,
        )
    )

    if chosen_arr.auto_scale == 1:
        c_scale.metric("Sprite Scale", "1x (Max)")
        scale = 1
    else:
        scale = int(
            c_scale.slider(
                "Sprite Scale",
                min_value=1,
                max_value=chosen_arr.auto_scale,
                value=chosen_arr.auto_scale,
            )
        )

    wall_img = generate_wallpaper(
        sprites=sprites,
        palettes=active_palettes,
        selected_dex=selected_dex,
        resolution=resolution,
        bg_rgb=bg_rgb,
        rows=chosen_arr.rows,
        scale=scale,
        margin_pct=margin_pct,
        row_align_mode=row_align,
        remainder_mode=remainder_mode,
    )

    st.image(
        wall_img,
        caption=(
            f"{resolution[0]}x{resolution[1]} px | "
            f"{len(selected_dex)} Pokemon across {chosen_arr.rows} Rows | "
            f"{scale}x Scale | {row_align}"
        ),
        use_container_width=True,
    )

    col_dl_wall, col_dl_zip = st.columns(2)
    png_data = image_to_png_bytes(wall_img)
    col_dl_wall.download_button(
        label="Download Wallpaper PNG",
        data=png_data,
        file_name=f"pokewall_{resolution[0]}x{resolution[1]}.png",
        mime="image/png",
        type="primary",
        use_container_width=True,
    )

    zip_bytes = create_sprites_zip(
        sprites=sprites,
        palettes=active_palettes,
        selected_dex=selected_dex,
        scale=scale,
        transparent_bg=False,
        bg_override=bg_rgb,
    )
    col_dl_zip.download_button(
        label=f"Download Sprites with Background ({len(selected_dex)} .zip)",
        data=zip_bytes,
        file_name="pokemon_subset_sprites.zip",
        mime="application/zip",
        use_container_width=True,
    )


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="PokeWall Studio",
        page_icon=":material/wallpaper:",
        layout="wide",
    )
    st.title("Pokemon Yellow Sprite & Wallpaper Studio")

    if "step" not in st.session_state:
        st.session_state["step"] = 3

    cfg = render_color_sidebar()
    rom_file = cfg["rom_file"]

    if rom_file is None:
        st.session_state["step"] = 3
        st.info("Step 1: Upload your Pokemon Yellow ROM in the sidebar.")
        return

    try:
        sprites, rom_palettes = cached_extract(
            rom_file.getvalue(),
            bool(cfg["back_sprites"]),
        )
    except ValueError as err:
        st.error(str(err))
        return

    active_palettes, bg_rgb = build_active_palettes(cfg, rom_palettes)

    if st.session_state["step"] == 3:
        render_step3_subset(sprites, active_palettes, bg_rgb)
    else:
        selected_dex = st.session_state.get(
            "selected_dex",
            list(range(1, 152)),
        )
        if not selected_dex:
            st.session_state["step"] = 3
            st.rerun()
        render_step4_wallpaper(
            sprites=sprites,
            active_palettes=active_palettes,
            bg_rgb=bg_rgb,
            selected_dex=selected_dex,
        )


if __name__ == "__main__":
    main()
