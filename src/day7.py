from bisect import bisect_left
from io import BytesIO

from PIL import Image, ImageFont, ImageDraw
from einops import rearrange
import numpy as np
from sty import fg
import requests


def generate_font_luminances(font, characters, invert=True):
    # Load font
    text_length = int(font.getlength(characters))
    font_width = text_length // len(characters)
    # Render to image
    font_draw = Image.new("L", (text_length, font.size), (255))
    draw_obj = ImageDraw.Draw(font_draw)
    draw_obj.text((0, 0), characters, font=font)
    # Extract luminances
    font_array = np.array(font_draw)
    if invert:
        font_array = 255 - font_array
    char_patches = rearrange(font_array, "h (b1 w) -> b1 h w", b1=len(characters))
    char_avgs = {
        char: float(np.mean(patch)) for char, patch in zip(characters, char_patches)
    }
    sorted_chars, sorted_lums = [
        *zip(*sorted([*char_avgs.items()], key=lambda x: x[1]))
    ]
    return font_width, sorted_chars, sorted_lums


def round_nearest(x, n=5):
    return int(n * round(float(x) / n))


URL = "https://foto.wuestenigel.com/wp-content/uploads/api/potted-plant-of-ferocactus-in-the-white-pot.jpeg"

FONT_PATH = "/Users/jamesashford/Library/Fonts/JetBrainsMonoNerdFont-Bold.ttf"
CHARACTERS = " .:coPO?@"
FONT_HEIGHT = 10
PATCH_SCALE = 2.5

# Generate font information
font = ImageFont.truetype(FONT_PATH, FONT_HEIGHT)
font_width, sorted_chars, sorted_lums = generate_font_luminances(font, CHARACTERS)
font_aspect = FONT_HEIGHT / font_width

patch_width = int(font_width * PATCH_SCALE)
patch_height = int(FONT_HEIGHT * PATCH_SCALE * font_aspect)

# Load image from URL
img = Image.open(BytesIO(requests.get(URL).content))
img = img.resize(
    size=(
        round_nearest(img.size[0], patch_width),
        round_nearest(img.size[1], patch_height),
    )
)
img_array_lum = np.array(img.convert("L"))

# Generate image patches
patch_dim = {"p1": patch_height, "p2": patch_width}
colour_patches = rearrange(
    np.array(img), "(h p1) (w p2) c -> (h w) p1 p2 c", **patch_dim
)
lum_patches = rearrange(img_array_lum, "(h p1) (w p2) -> (h w) p1 p2", **patch_dim)

# Find and colour best matching letters
image = ""
patch_colours = np.mean(colour_patches, axis=(1, 2)).astype(int)
for ind, (lum_patch, colour) in enumerate(zip(lum_patches, patch_colours)):
    if ind % (img.size[0] // patch_width) == 0 and ind != 0:
        image += "\n"
    nearest_letter = sorted_chars[bisect_left(sorted_lums, np.mean(lum_patch)) - 1]
    image += fg(*colour) + nearest_letter + fg.rs
print(image)


# Draw to image
out_font = ImageFont.truetype(FONT_PATH, FONT_HEIGHT * PATCH_SCALE)

image = ""
for ind, lum_patch in enumerate(lum_patches):
    if ind % (img.size[0] // patch_width) == 0 and ind != 0:
        image += "\n"
    image += sorted_chars[bisect_left(sorted_lums, np.mean(lum_patch)) - 1]

final_draw = Image.new("RGB", img.size, 0)
draw_obj = ImageDraw.Draw(final_draw)
draw_obj.multiline_text((0, 0), image, font=out_font)
final_draw.save("out/day7.png", "PNG")
