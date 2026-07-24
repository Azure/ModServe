## Analytical vision-token counts per image, used to weight image-node routing.

# InternVL2.5: a 448x448 tile -> 1024 ViT patches -> 256 after pixel-unshuffle (/4).
INTERNVL_TILE_SIZE = 448
INTERNVL_TOKENS_PER_TILE = 256
INTERNVL_MIN_TILES = 1
INTERNVL_MAX_TILES = 12

MLLAMA_TOKENS_PER_TILE = 1601
MLLAMA_NUM_TILES = 4


def find_closest_aspect_ratio(
    aspect_ratio, target_ratios, *, width, height, image_size
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def get_internvl_target_ratios(min_num: int, max_num: int):
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def calculate_internvl_targets(
    *, orig_width, orig_height, target_ratios, image_size, use_thumbnail
):
    aspect_ratio = orig_width / orig_height
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # An extra thumbnail tile (whole image) is added whenever the image splits into >1 tile.
    if use_thumbnail and blocks != 1:
        blocks += 1
    return blocks, target_width, target_height


def internvl_num_tiles(width: int, height: int) -> int:
    ratios = get_internvl_target_ratios(INTERNVL_MIN_TILES, INTERNVL_MAX_TILES)
    blocks, _, _ = calculate_internvl_targets(
        orig_width=width,
        orig_height=height,
        target_ratios=ratios,
        image_size=INTERNVL_TILE_SIZE,
        use_thumbnail=True,
    )
    return blocks


def vision_tokens(model_arch: str, width: int, height: int) -> int:
    if model_arch == "MLLAMA":
        return MLLAMA_NUM_TILES * MLLAMA_TOKENS_PER_TILE
    if model_arch == "INTERNVL":
        return internvl_num_tiles(width, height) * INTERNVL_TOKENS_PER_TILE
    raise ValueError(f"Unknown model_arch={model_arch!r} (expected INTERNVL or MLLAMA)")
