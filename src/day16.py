"""Generative Colour Palette"""

from einops import rearrange, repeat
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift
from skimage import color
import matplotlib.pylab as plt
from PIL import Image
import numpy as np
import requests

URL = "https://c4.wallpaperflare.com/wallpaper/764/431/702/river-trees-forest-clouds-wallpaper-preview.jpg"


def create_pattern(rng_obj):
    img = Image.open(requests.get(URL, stream=True).raw)
    img = np.array(img)
    h, w, c = img.shape
    mid = np.array([h // 2, w // 2])

    # Convert the pixels into (y, x, r, g, b) form
    indexes = np.indices(img.shape[:2])
    coords = rearrange(indexes, "c h w -> (h w) c")
    coords = (coords / mid) * 255
    colours = rearrange(img, "h w c -> (h w) c")
    colours = color.rgb2lab(colours)

    pixel_data = np.hstack([coords * 0, colours])

    # Embed in low-dim space with PCA
    pca_obj = PCA(n_components=3)
    low_dim = pca_obj.fit_transform(pixel_data)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(*low_dim.T, s=0.01, c=colours / 255)
    # plt.show()

    # Cluster the low-dim space
    samples = low_dim[np.random.choice(len(low_dim), replace=False, size=1000), :]
    clusterer = MeanShift(bandwidth=15)
    clusterer = clusterer.fit(samples)
    labels = clusterer.predict(low_dim)
    centres = clusterer.cluster_centers_

    # Match each pixel to it's nearest centroid
    centre_colours = pca_obj.inverse_transform(centres)[:, 2:]
    centre_colours = color.lab2rgb(centre_colours)
    palette = repeat(centre_colours, "h c -> h w c", w=1)
    pixel_colours = centre_colours[labels]

    # Put back into image form
    new_img = rearrange(pixel_colours, "(h w) c -> h w c", h=h, w=w)
    return new_img, palette


rng = np.random.default_rng(3)
image, palette = create_pattern(rng)
aspect = image.shape[0] / image.shape[1]
figsize = (10, int(10 * aspect))

fig, axs = plt.subplots(1, 2, figsize=figsize, squeeze=True, width_ratios=[8, 1])

axs[0].imshow(image)
axs[0].set_xticks([])
axs[0].set_yticks([])

axs[1].imshow(palette)
axs[1].set_xticks([])
axs[1].set_yticks([])

fig.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig("out/day16.png", bbox_inches="tight", transparent=True)
plt.show()
