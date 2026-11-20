"""Single-screen 4-step Streamlit wizard for Pokemon Yellow wallpapers."""

from typing import Any
import uuid

import streamlit as st

from extractor import RGBColor, RomData, SpriteGrid, extract_rom_data
from filters import (
    FILTER_LABELS,
    RESAMPLERS,
    default_filter_params,
)
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
    render_comparison_pair,
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


def init_session_state() -> None:
    """Initializes default session state keys for the 4-step wizard."""
    defaults: dict[str, object] = {
        "step": 1,
        "rom_bytes": None,
        "rom_name": "",
        "back_sprites": False,
        "color_mode": "Original GBC (Per-Pokemon)",
        "bg_hex": "#1E1E24",
        "grad_bg_hex": "#0F380F",
        "grad_fg_hex": "#9BBC0F",
        "cust_c0": "#1E1E24",
        "cust_c1": "#F8A858",
        "cust_c2": "#D84020",
        "cust_c3": "#000000",
        "swap_label": "RGB (Default)",
        "hue_shift": 0,
        "invert_colors": False,
        "transform_bg": False,
        "subset_mode": "All 151 Pokemon",
        "selected_dex": list(range(1, 152)),
        "ar_choice": "16:9 (Widescreen)",
        "resolution": (1920, 1080),
        "margin_pct": 2,
        "rows": 9,
        "scale": 2,
        "row_align": ROW_ALIGN_MODES[0],
        "remainder_mode": REMAINDER_MODES[0],
        "filter_pipeline": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def go_to_step(step_num: int) -> None:
    """Transitions the active wizard step."""
    st.session_state["step"] = step_num


def add_filter_step(filter_type: str) -> None:
    """Appends a new filter configuration to the ordered pipeline."""
    pipeline: list[dict[str, Any]] = st.session_state["filter_pipeline"]
    pipeline.append(
        {
            "id": uuid.uuid4().hex[:8],
            "type": filter_type,
            "enabled": True,
            "params": default_filter_params(filter_type),
        }
    )


def move_filter_step(index: int, delta: int) -> None:
    """Moves a pipeline step up (-1) or down (+1) in execution order."""
    pipeline: list[dict[str, Any]] = st.session_state["filter_pipeline"]
    target = index + delta
    if 0 <= target < len(pipeline):
        pipeline[index], pipeline[target] = pipeline[target], pipeline[index]


def remove_filter_step(index: int) -> None:
    """Removes a filter step from the pipeline by index."""
    pipeline: list[dict[str, Any]] = st.session_state["filter_pipeline"]
    if 0 <= index < len(pipeline):
        pipeline.pop(index)


def clear_filter_pipeline() -> None:
    """Clears all post-processing filters from the pipeline."""
    st.session_state["filter_pipeline"] = []


def build_active_palettes(
    rom_palettes: dict[int, list[RGBColor]],
) -> tuple[dict[int, list[RGBColor]], RGBColor]:
    """Computes transformed per-Pokemon palettes and background from state."""
    mode = str(st.session_state["color_mode"])
    if mode == "Original GBC (Per-Pokemon)":
        base_map = rom_palettes
        raw_bg_hex = str(st.session_state["bg_hex"])
    elif mode == "2-Color Auto Gradient":
        raw_bg_hex = str(st.session_state["grad_bg_hex"])
        fg_hex = str(st.session_state["grad_fg_hex"])
        uni = build_interpolated_palette(raw_bg_hex, fg_hex)
        base_map = {dex: list(uni) for dex in range(1, 152)}
    else:
        raw_bg_hex = str(st.session_state["cust_c0"])
        uni = [
            hex_to_rgb(raw_bg_hex),
            hex_to_rgb(str(st.session_state["cust_c1"])),
            hex_to_rgb(str(st.session_state["cust_c2"])),
            hex_to_rgb(str(st.session_state["cust_c3"])),
        ]
        base_map = {dex: list(uni) for dex in range(1, 152)}

    swap_key = str(st.session_state["swap_label"])
    chan_order = CHANNEL_SWAPS[swap_key]
    hue_shift = int(st.session_state["hue_shift"])
    invert = bool(st.session_state["invert_colors"])
    transform_bg = bool(st.session_state["transform_bg"])

    transformed_map = transform_palette_map(
        palettes=base_map,
        channel_order=chan_order,
        hue_shift=hue_shift,
        invert=invert,
        transform_bg_slot=transform_bg,
    )

    bg_rgb = hex_to_rgb(raw_bg_hex)
    if transform_bg:
        bg_rgb = transform_color(bg_rgb, chan_order, hue_shift, invert)

    return transformed_map, bg_rgb


def render_step1_import() -> None:
    """Renders Step 1: Main-screen ROM import and sprite extraction."""
    st.subheader("Step 1: Import Pokemon Yellow ROM")
    st.write(
        "Upload your legally acquired Pokemon Yellow (`.gbc` or `.gb`) ROM "
        "to extract all 151 sprites and original Game Boy Color palettes."
    )

    rom_file = st.file_uploader(
        "Select ROM File",
        type=["gb", "gbc"],
    )
    if rom_file is not None:
        st.session_state["rom_bytes"] = rom_file.getvalue()
        st.session_state["rom_name"] = rom_file.name

    st.session_state["back_sprites"] = st.toggle(
        "Extract Battle Back Sprites Instead of Front Sprites",
        value=bool(st.session_state["back_sprites"]),
    )

    rom_bytes: bytes | None = st.session_state["rom_bytes"]
    if rom_bytes is None:
        st.info("Waiting for ROM upload to unlock Step 2.")
        return

    try:
        cached_extract(rom_bytes, bool(st.session_state["back_sprites"]))
    except ValueError as err:
        st.error(str(err))
        return

    st.success(f"Loaded ROM: **{st.session_state['rom_name']}**")
    st.button(
        "Proceed to Step 2: Colors & Pokemon Subset \u2192",
        type="primary",
        use_container_width=True,
        on_click=go_to_step,
        args=(2,),
    )


def render_step2_colors_and_subset(
    sprites: dict[int, SpriteGrid],
    rom_palettes: dict[int, list[RGBColor]],
) -> None:
    """Renders Step 2: Palettes, global transforms, and subset selection."""
    col_back, col_title = st.columns([1, 3])
    col_back.button(
        "\u2190 Back to Step 1 (ROM Import)",
        on_click=go_to_step,
        args=(1,),
        use_container_width=True,
    )
    col_title.subheader("Step 2: Color Palettes, Transformations & Subset")

    col_pal, col_ops = st.columns(2)

    with col_pal:
        st.markdown("**Base Palette & Wallpaper Background**")
        modes = [
            "Original GBC (Per-Pokemon)",
            "2-Color Auto Gradient",
            "4-Color Custom GBC",
        ]
        st.session_state["color_mode"] = st.radio(
            "Palette Mode",
            options=modes,
            index=modes.index(str(st.session_state["color_mode"])),
        )

        active_mode = st.session_state["color_mode"]
        if active_mode == "Original GBC (Per-Pokemon)":
            st.session_state["bg_hex"] = st.color_picker(
                "Wallpaper Background Color",
                value=str(st.session_state["bg_hex"]),
            )
        elif active_mode == "2-Color Auto Gradient":
            c_bg, c_fg = st.columns(2)
            st.session_state["grad_bg_hex"] = c_bg.color_picker(
                "Background (Slot 0)",
                value=str(st.session_state["grad_bg_hex"]),
            )
            st.session_state["grad_fg_hex"] = c_fg.color_picker(
                "Foreground (Slot 3)",
                value=str(st.session_state["grad_fg_hex"]),
            )
        else:
            c0, c1, c2, c3 = st.columns(4)
            st.session_state["cust_c0"] = c0.color_picker(
                "BG (0)",
                value=str(st.session_state["cust_c0"]),
            )
            st.session_state["cust_c1"] = c1.color_picker(
                "Light (1)",
                value=str(st.session_state["cust_c1"]),
            )
            st.session_state["cust_c2"] = c2.color_picker(
                "Dark (2)",
                value=str(st.session_state["cust_c2"]),
            )
            st.session_state["cust_c3"] = c3.color_picker(
                "Ink (3)",
                value=str(st.session_state["cust_c3"]),
            )

    with col_ops:
        st.markdown("**Global Color Transformations**")
        swap_keys = list(CHANNEL_SWAPS.keys())
        st.session_state["swap_label"] = st.selectbox(
            "RGB Channel Swap",
            options=swap_keys,
            index=swap_keys.index(str(st.session_state["swap_label"])),
        )
        st.session_state["hue_shift"] = st.slider(
            "Hue Shift (Degrees)",
            min_value=-180,
            max_value=180,
            value=int(st.session_state["hue_shift"]),
            step=5,
        )
        t1, t2 = st.columns(2)
        st.session_state["invert_colors"] = t1.toggle(
            "Invert Sprite Palettes",
            value=bool(st.session_state["invert_colors"]),
        )
        st.session_state["transform_bg"] = t2.toggle(
            "Transform Background Too",
            value=bool(st.session_state["transform_bg"]),
        )

    active_palettes, bg_rgb = build_active_palettes(rom_palettes)

    st.divider()
    st.markdown("**Select Pokemon Subset by Post-Transform Dominant Color**")
    color_groups = categorize_all_sprites(sprites, active_palettes)
    non_empty_cats = [cat for cat in COLOR_CATEGORIES if color_groups[cat]]

    col_scope, col_cats = st.columns([1, 2])
    scope_opts = ["All 151 Pokemon", "Filter by Dominant Color"]
    st.session_state["subset_mode"] = col_scope.radio(
        "Selection Scope",
        options=scope_opts,
        index=scope_opts.index(str(st.session_state["subset_mode"])),
    )

    if st.session_state["subset_mode"] == "All 151 Pokemon":
        selected_cats = non_empty_cats
    else:
        cat_labels = {
            cat: f"{cat} ({len(color_groups[cat])})" for cat in non_empty_cats
        }
        chosen_labels = col_cats.multiselect(
            "Dominant Foreground Colors (Ignores Background)",
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
        f"Preview Individual Sprites with Final Background "
        f"({len(selected_dex)} Pokemon)",
        expanded=True,
    ):
        if not selected_dex:
            st.warning("Select at least one dominant color category above.")
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
            f"Lock {len(selected_dex)} Pokemon & Proceed to Step 3 "
            "(Wallpaper Layout) \u2192",
            type="primary",
            use_container_width=True,
            on_click=go_to_step,
            args=(3,),
        )


def render_step3_layout(
    sprites: dict[int, SpriteGrid],
    rom_palettes: dict[int, list[RGBColor]],
) -> None:
    """Renders Step 3: Aspect ratio and 6 uneven row placement options."""
    active_palettes, bg_rgb = build_active_palettes(rom_palettes)
    selected_dex: list[int] = st.session_state["selected_dex"]

    col_back, col_title = st.columns([1, 3])
    col_back.button(
        "\u2190 Back to Step 2 (Colors & Subset)",
        on_click=go_to_step,
        args=(2,),
        use_container_width=True,
    )
    col_title.subheader(
        f"Step 3: Wallpaper Arrangement ({len(selected_dex)} Pokemon)"
    )

    c_ar, c_res, c_margin = st.columns(3)
    ar_opts = [*list(ASPECT_RATIOS.keys()), "Custom"]
    ar_choice = str(
        c_ar.selectbox(
            "Aspect Ratio",
            options=ar_opts,
            index=ar_opts.index(str(st.session_state["ar_choice"])),
        )
    )
    st.session_state["ar_choice"] = ar_choice

    if ar_choice == "Custom":
        cw, ch = c_res.columns(2)
        cur_w, cur_h = st.session_state["resolution"]
        width = int(cw.number_input("Width (px)", 320, 7680, int(cur_w), 10))
        height = int(ch.number_input("Height (px)", 240, 4320, int(cur_h), 10))
        resolution = (width, height)
    else:
        presets = ASPECT_RATIOS[ar_choice]
        res_label = c_res.selectbox(
            "Target Resolution",
            options=list(presets.keys()),
        )
        resolution = presets[str(res_label)]

    margin_pct = int(
        c_margin.slider(
            "Outer Canvas Margin (%)",
            min_value=0,
            max_value=15,
            value=int(st.session_state["margin_pct"]),
        )
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

    align_opts = list(ROW_ALIGN_MODES)
    row_align = str(
        c_align.selectbox(
            "Horizontal Space Distribution",
            options=align_opts,
            index=align_opts.index(str(st.session_state["row_align"])),
        )
    )

    has_uneven_rows = chosen_arr.min_cols != chosen_arr.max_cols
    rem_opts = list(REMAINDER_MODES)
    remainder_mode = str(
        c_rem.selectbox(
            "Uneven Row Placement",
            options=rem_opts,
            index=rem_opts.index(str(st.session_state["remainder_mode"])),
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

    st.session_state["resolution"] = resolution
    st.session_state["margin_pct"] = margin_pct
    st.session_state["rows"] = chosen_arr.rows
    st.session_state["scale"] = scale
    st.session_state["row_align"] = row_align
    st.session_state["remainder_mode"] = remainder_mode

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
            f"Layout Preview: {resolution[0]}x{resolution[1]} px | "
            f"{chosen_arr.rows} Rows | {scale}x Scale | {remainder_mode}"
        ),
        use_container_width=True,
    )

    st.button(
        "Lock Layout & Proceed to Step 4 (Filter Pipeline & Export) \u2192",
        type="primary",
        use_container_width=True,
        on_click=go_to_step,
        args=(4,),
    )


def render_filter_controls(step: dict[str, Any]) -> None:
    """Renders inline parameter widgets for a single pipeline filter step."""
    uid = str(step["id"])
    f_type = str(step["type"])
    params: dict[str, Any] = step["params"]

    if f_type == "scale2x":
        c1, c2 = st.columns(2)
        params["passes"] = c1.radio(
            "Passes (2x or 4x)",
            options=[1, 2],
            index=[1, 2].index(int(params.get("passes", 1))),
            horizontal=True,
            key=f"s2x_p_{uid}",
        )
        params["tolerance"] = c2.slider(
            "Color Match Tolerance",
            0,
            60,
            int(params.get("tolerance", 15)),
            5,
            key=f"s2x_t_{uid}",
        )
    elif f_type == "hq2x":
        c1, c2 = st.columns(2)
        params["passes"] = c1.radio(
            "Passes (2x or 4x)",
            options=[1, 2],
            index=[1, 2].index(int(params.get("passes", 1))),
            horizontal=True,
            key=f"hq_p_{uid}",
        )
        params["strength"] = c2.slider(
            "Corner Blend Strength",
            0.1,
            1.0,
            float(params.get("strength", 0.75)),
            0.05,
            key=f"hq_s_{uid}",
        )
    elif f_type == "xbr2x":
        c1, c2 = st.columns(2)
        params["passes"] = c1.radio(
            "Passes (2x or 4x)",
            options=[1, 2],
            index=[1, 2].index(int(params.get("passes", 1))),
            horizontal=True,
            key=f"xbr_p_{uid}",
        )
        params["blend"] = c2.slider(
            "Directional Edge Blend",
            0.1,
            1.0,
            float(params.get("blend", 0.65)),
            0.05,
            key=f"xbr_b_{uid}",
        )
    elif f_type == "morphological":
        c1, c2, c3 = st.columns(3)
        m_opts = ["ModeFilter", "MedianFilter"]
        params["mode"] = c1.selectbox(
            "Filter Type",
            options=m_opts,
            index=m_opts.index(str(params.get("mode", "ModeFilter"))),
            key=f"mor_m_{uid}",
        )
        k_opts = [3, 5]
        params["kernel"] = c2.radio(
            "Kernel Size",
            options=k_opts,
            index=k_opts.index(int(params.get("kernel", 3))),
            horizontal=True,
            key=f"mor_k_{uid}",
        )
        u_opts = [1, 2, 4]
        params["upscale"] = c3.radio(
            "Pre-Upscale",
            options=u_opts,
            index=u_opts.index(int(params.get("upscale", 2))),
            horizontal=True,
            key=f"mor_u_{uid}",
        )
    elif f_type == "ssaa":
        c1, c2 = st.columns(2)
        f_opts = [2, 4]
        params["factor"] = c1.radio(
            "Supersample Factor",
            options=f_opts,
            index=f_opts.index(int(params.get("factor", 2))),
            horizontal=True,
            key=f"ssaa_f_{uid}",
        )
        r_opts = list(RESAMPLERS.keys())
        params["resampler"] = c2.selectbox(
            "Downscale Kernel",
            options=r_opts,
            index=r_opts.index(str(params.get("resampler", "Bilinear"))),
            key=f"ssaa_r_{uid}",
        )
    elif f_type == "gaussian_blur":
        c1, c2 = st.columns(2)
        params["radius"] = c1.slider(
            "Blur Radius (px)",
            0.1,
            3.0,
            float(params.get("radius", 0.6)),
            0.1,
            key=f"gb_r_{uid}",
        )
        params["strength"] = c2.slider(
            "Blend Alpha",
            0.05,
            1.0,
            float(params.get("strength", 0.30)),
            0.05,
            key=f"gb_s_{uid}",
        )


def render_step4_postprocess(
    sprites: dict[int, SpriteGrid],
    rom_palettes: dict[int, list[RGBColor]],
) -> None:
    """Renders Step 4: Reorderable filter pipeline, inspector, and export."""
    active_palettes, bg_rgb = build_active_palettes(rom_palettes)
    selected_dex: list[int] = st.session_state["selected_dex"]
    resolution: tuple[int, int] = st.session_state["resolution"]
    rows = int(st.session_state["rows"])
    scale = int(st.session_state["scale"])
    margin_pct = int(st.session_state["margin_pct"])
    row_align = str(st.session_state["row_align"])
    remainder_mode = str(st.session_state["remainder_mode"])

    col_back, col_title = st.columns([1, 3])
    col_back.button(
        "\u2190 Back to Step 3 (Wallpaper Layout)",
        on_click=go_to_step,
        args=(3,),
        use_container_width=True,
    )
    col_title.subheader("Step 4: Reorderable Post-Processing Pipeline")

    c_pick, c_add, c_clr = st.columns([3, 1.5, 1])
    filter_keys = list(FILTER_LABELS.keys())
    new_filter = str(
        c_pick.selectbox(
            "Add Filter to Pipeline",
            options=filter_keys,
            format_func=lambda k: FILTER_LABELS[str(k)],
        )
    )
    c_add.write("")
    c_add.button(
        "+ Add Filter",
        type="primary",
        use_container_width=True,
        on_click=add_filter_step,
        args=(new_filter,),
    )
    c_clr.write("")
    c_clr.button(
        "Clear All",
        use_container_width=True,
        on_click=clear_filter_pipeline,
    )

    pipeline: list[dict[str, Any]] = st.session_state["filter_pipeline"]

    col_queue, col_inspect = st.columns([1.4, 1])

    with col_queue:
        st.markdown("**Active Pipeline (Applied Top to Bottom)**")
        if not pipeline:
            st.info(
                "No filters added yet. Add one or more filters above to "
                "test and reorder transformations."
            )
        for idx, step in enumerate(pipeline):
            uid = str(step["id"])
            title = FILTER_LABELS[str(step["type"])]
            with st.container(border=True):
                h_lbl, h_en, h_up, h_dn, h_rm = st.columns(
                    [3.5, 1.2, 0.7, 0.7, 0.7]
                )
                h_lbl.markdown(f"**{idx + 1}. {title}**")
                step["enabled"] = h_en.toggle(
                    "On",
                    value=bool(step["enabled"]),
                    key=f"en_{uid}",
                )
                h_up.button(
                    "\u2191",
                    key=f"up_{uid}",
                    disabled=idx == 0,
                    on_click=move_filter_step,
                    args=(idx, -1),
                )
                h_dn.button(
                    "\u2193",
                    key=f"dn_{uid}",
                    disabled=idx == len(pipeline) - 1,
                    on_click=move_filter_step,
                    args=(idx, 1),
                )
                h_rm.button(
                    "\u2715",
                    key=f"rm_{uid}",
                    on_click=remove_filter_step,
                    args=(idx,),
                )
                render_filter_controls(step)

    with col_inspect:
        st.markdown("**Zoomed Edge Inspector (Before vs. After)**")
        default_dex = 6 if 6 in selected_dex else selected_dex[0]
        inspect_dex = int(
            st.selectbox(
                "Sample Pokemon",
                options=selected_dex,
                index=selected_dex.index(default_dex),
                format_func=lambda d: f"#{d:03d}",
            )
        )
        raw_zoom, flt_zoom = render_comparison_pair(
            grid=sprites[inspect_dex],
            palette=active_palettes[inspect_dex],
            bg_rgb=bg_rgb,
            scale=scale,
            filter_pipeline=pipeline,
        )
        ci1, ci2 = st.columns(2)
        ci1.image(raw_zoom, caption="Original (Nearest)", width="stretch")
        ci2.image(flt_zoom, caption="After Pipeline", width="stretch")

    st.divider()
    wall_img = generate_wallpaper(
        sprites=sprites,
        palettes=active_palettes,
        selected_dex=selected_dex,
        resolution=resolution,
        bg_rgb=bg_rgb,
        rows=rows,
        scale=scale,
        margin_pct=margin_pct,
        row_align_mode=row_align,
        remainder_mode=remainder_mode,
        filter_pipeline=pipeline,
    )

    active_names = [
        FILTER_LABELS[str(s["type"])].split(" (")[0]
        for s in pipeline
        if s.get("enabled", True)
    ]
    chain_desc = " -> ".join(active_names) if active_names else "None (Raw)"

    st.image(
        wall_img,
        caption=(
            f"Final Output: {resolution[0]}x{resolution[1]} px | "
            f"Pipeline: {chain_desc}"
        ),
        use_container_width=True,
    )

    col_dl_wall, col_dl_zip = st.columns(2)
    png_data = image_to_png_bytes(wall_img)
    col_dl_wall.download_button(
        label="Download Final Wallpaper PNG",
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
        filter_pipeline=pipeline,
    )
    col_dl_zip.download_button(
        label=f"Download Post-Processed Sprites ({len(selected_dex)} .zip)",
        data=zip_bytes,
        file_name="pokemon_smoothed_sprites.zip",
        mime="application/zip",
        use_container_width=True,
    )


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="PokeWall Studio",
        page_icon=":material/wallpaper:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()

    st.title("Pokemon Yellow Sprite & Wallpaper Studio")

    step = int(st.session_state["step"])
    rom_bytes: bytes | None = st.session_state["rom_bytes"]

    if step == 1 or rom_bytes is None:
        st.session_state["step"] = 1
        render_step1_import()
        return

    sprites, rom_palettes = cached_extract(
        rom_bytes,
        bool(st.session_state["back_sprites"]),
    )

    if step == 2:
        render_step2_colors_and_subset(sprites, rom_palettes)
    elif step in (3, 4):
        if not st.session_state["selected_dex"]:
            st.session_state["step"] = 2
            st.rerun()
        if step == 3:
            render_step3_layout(sprites, rom_palettes)
        else:
            render_step4_postprocess(sprites, rom_palettes)


if __name__ == "__main__":
    main()
