"""ROM bitstream decompression and palette extraction for Pokemon Yellow."""

import struct

HEADER_SIZE = 28
CANVAS_SIZE = 56

RGBColor = tuple[int, int, int]
SpriteGrid = list[list[int]]
RomData = tuple[dict[int, SpriteGrid], dict[int, list[RGBColor]]]


class BitstreamReader:
    """Reads variable-length MSB-first bitstreams from a ROM buffer."""

    def __init__(self, data: bytes, offset: int) -> None:
        self.data = data
        self.byte_pos = offset
        self.bit_pos = 7

    def read_bit(self) -> int:
        bit = (self.data[self.byte_pos] >> self.bit_pos) & 1
        self.bit_pos -= 1
        if self.bit_pos < 0:
            self.bit_pos = 7
            self.byte_pos += 1
        return bit

    def read_bits(self, count: int) -> int:
        val = 0
        for _ in range(count):
            val = (val << 1) | self.read_bit()
        return val


def find_base_stats_offset(rom: bytes) -> int:
    """Locates the BaseStats table via Bulbasaur's stat signature."""
    bulbasaur_sig = bytes([1, 45, 49, 49, 45, 65, 22, 3, 45, 64])
    offset = rom.find(bulbasaur_sig)
    if offset == -1:
        msg = "Invalid ROM: BaseStats table signature not found."
        raise ValueError(msg)
    return offset


def get_internal_index_map(rom: bytes) -> dict[int, int]:
    """Maps Pokedex # (1-151) to Internal ID (1-190) via ROM signature."""
    sig = bytes([112, 115, 32, 35, 21, 100, 34, 80, 2, 103])
    table_offset = rom.find(sig)
    if table_offset == -1:
        msg = "Invalid ROM: PokedexOrder table signature not found."
        raise ValueError(msg)

    dex_to_internal: dict[int, int] = {}
    for internal_id in range(1, 191):
        dex_num = rom[table_offset + internal_id - 1]
        if 1 <= dex_num <= 151 and dex_num not in dex_to_internal:
            dex_to_internal[dex_num] = internal_id
    return dex_to_internal


def get_sprite_bank(internal_id: int) -> int:
    """Maps an internal Pokemon ID to its ROM bank in Pokemon Yellow."""
    if internal_id < 0x1F:
        return 0x09
    if internal_id < 0x4A:
        return 0x0A
    if internal_id < 0x74:
        return 0x0B
    if internal_id < 0x99:
        return 0x0C
    return 0x0D


def decode_bgr555_palette(data: bytes) -> list[RGBColor]:
    """Decodes an 8-byte GBC/SGB BGR555 palette into 4 RGB888 tuples."""
    words = struct.unpack("<4H", data[:8])
    palette: list[RGBColor] = []
    for word in words:
        r = ((word & 0x1F) * 255) // 31
        g = (((word >> 5) & 0x1F) * 255) // 31
        b = (((word >> 10) & 0x1F) * 255) // 31
        palette.append((r, g, b))
    return palette


def extract_pokemon_palettes(rom: bytes) -> dict[int, list[RGBColor]]:
    """Extracts each Pokemon's original GBC palette from the ROM."""
    dex_1_to_15_sig = bytes(
        [
            0x16,
            0x16,
            0x16,
            0x12,
            0x12,
            0x12,
            0x13,
            0x13,
            0x13,
            0x16,
            0x16,
            0x13,
            0x18,
            0x18,
            0x18,
        ]
    )
    sig_off = rom.find(dex_1_to_15_sig)
    if sig_off == -1:
        msg = "Invalid ROM: MonsterPalettes table not found."
        raise ValueError(msg)

    mon_pal_off = sig_off - 1
    super_pal_off = mon_pal_off + 152

    palettes: dict[int, list[RGBColor]] = {}
    for dex_num in range(1, 152):
        pal_id = rom[mon_pal_off + dex_num]
        entry_off = super_pal_off + (pal_id * 8)
        palettes[dex_num] = decode_bgr555_palette(
            rom[entry_off : entry_off + 8]
        )
    return palettes


def unpack_bitplane(
    reader: BitstreamReader,
    width_tiles: int,
    height_tiles: int,
) -> SpriteGrid:
    """Unpacks a single 1BPP bitplane using RLE and literal alternation."""
    width_px = width_tiles * 8
    height_px = height_tiles * 8
    grid: SpriteGrid = [[0] * width_px for _ in range(height_px)]

    total_pairs = (width_px // 2) * height_px
    pairs: list[int] = []
    packet_type = reader.read_bit()

    while len(pairs) < total_pairs:
        if packet_type == 0:
            prefix_ones = 0
            while reader.read_bit() == 1:
                prefix_ones += 1
            val_bits = reader.read_bits(prefix_ones + 1)
            run_len = (1 << (prefix_ones + 1)) - 1 + val_bits
            pairs.extend([0] * run_len)
            packet_type = 1
        else:
            while len(pairs) < total_pairs:
                pair = reader.read_bits(2)
                if pair == 0:
                    break
                pairs.append(pair)
            packet_type = 0

    idx = 0
    for col_pair in range(width_px // 2):
        for row in range(height_px):
            pair = pairs[idx] if idx < len(pairs) else 0
            grid[row][col_pair * 2] = (pair >> 1) & 1
            grid[row][col_pair * 2 + 1] = pair & 1
            idx += 1

    return grid


def delta_decode(grid: SpriteGrid) -> None:
    """Applies horizontal cumulative XOR across each row in-place."""
    for row in grid:
        state = 0
        for x in range(len(row)):
            state ^= row[x]
            row[x] = state


def xor_grids(dest: SpriteGrid, src: SpriteGrid) -> None:
    """XORs dest bitplane with src bitplane in-place."""
    for y in range(len(dest)):
        for x in range(len(dest[y])):
            dest[y][x] ^= src[y][x]


def decompress_sprite_indices(rom: bytes, offset: int) -> SpriteGrid:
    """Decompresses a sprite into a 56x56 grid of 2-bit color indices."""
    reader = BitstreamReader(rom, offset)
    width_tiles = reader.read_bits(4)
    height_tiles = reader.read_bits(4)

    if not (1 <= width_tiles <= 7 and 1 <= height_tiles <= 7):
        msg = (
            f"Invalid sprite size ({width_tiles}x{height_tiles} tiles) "
            f"at offset 0x{offset:05X}."
        )
        raise ValueError(msg)

    primary_plane = reader.read_bit()
    first_grid = unpack_bitplane(reader, width_tiles, height_tiles)

    if reader.read_bit() == 0:
        mode = 0
    else:
        mode = 1 if reader.read_bit() == 0 else 2

    second_grid = unpack_bitplane(reader, width_tiles, height_tiles)

    if primary_plane == 0:
        plane0, plane1 = first_grid, second_grid
    else:
        plane1, plane0 = first_grid, second_grid

    if mode == 0:
        delta_decode(plane0)
        delta_decode(plane1)
    elif mode == 1:
        delta_decode(first_grid)
        xor_grids(second_grid, first_grid)
    elif mode == 2:
        delta_decode(first_grid)
        delta_decode(second_grid)
        xor_grids(second_grid, first_grid)

    canvas: SpriteGrid = [[0] * CANVAS_SIZE for _ in range(CANVAS_SIZE)]
    sprite_w = width_tiles * 8
    sprite_h = height_tiles * 8
    pad_x = (CANVAS_SIZE - sprite_w) // 2
    pad_y = CANVAS_SIZE - sprite_h

    for y in range(sprite_h):
        for x in range(sprite_w):
            color_idx = (plane1[y][x] << 1) | plane0[y][x]
            canvas[pad_y + y][pad_x + x] = color_idx

    return canvas


def extract_rom_data(
    rom: bytes,
    back_sprites: bool = False,
) -> RomData:
    """Extracts 2-bit index grids and GBC palettes for all 151 Pokemon."""
    base_stats_offset = find_base_stats_offset(rom)
    dex_to_internal = get_internal_index_map(rom)
    palettes = extract_pokemon_palettes(rom)
    sprites: dict[int, SpriteGrid] = {}
    ptr_slice = slice(0x0D, 0x0F) if back_sprites else slice(0x0B, 0x0D)

    for dex_num in range(1, 152):
        h_off = base_stats_offset + (dex_num - 1) * HEADER_SIZE
        header = rom[h_off : h_off + HEADER_SIZE]
        ptr = struct.unpack("<H", header[ptr_slice])[0]

        internal_id = dex_to_internal[dex_num]
        bank = get_sprite_bank(internal_id)
        rom_offset = (bank * 0x4000) + (ptr - 0x4000)

        sprites[dex_num] = decompress_sprite_indices(rom, rom_offset)

    return sprites, palettes
