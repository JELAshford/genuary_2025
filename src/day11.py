"""Impossible!"""

from transformers import AutoProcessor, AutoTokenizer, FlaxCLIPModel
from einops import rearrange, repeat
import matplotlib.pylab as plt
from tqdm import trange
from flax import nnx
import optax
import jax


class CPPN(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        dim_in=2,
        dim_hidden=64,
        dim_out=3,
        n_layers=3,
    ):
        self.linear_in = nnx.Linear(dim_in, dim_hidden, rngs=rngs, use_bias=True)
        self.linear_mids = [
            nnx.Linear(dim_hidden, dim_hidden, rngs=rngs, use_bias=True)
            for _ in range(n_layers)
        ]
        self.linear_out = nnx.Linear(dim_hidden, dim_out, rngs=rngs, use_bias=False)

    def __call__(self, x):
        x = self.linear_in(x)
        x = jax.nn.tanh(x)
        for layer in self.linear_mids:
            x = layer(x)
            x = jax.nn.tanh(x)
        x = self.linear_out(x)
        x = jax.nn.hard_sigmoid(x)
        return x


def show_image(model, indices):
    colours = model(indices)
    image_out = rearrange(colours, "(h w) c -> h w c", h=SIZE, w=SIZE)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
    for ax in axs.flatten():
        ax.imshow(image_out, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("out/day11.png", bbox_inches="tight")
    plt.show()


@nnx.jit
def train_step(cppn, optim, x, y):
    def loss_fn(cppn):
        y_pred = cppn(x)
        y_pred = repeat(y_pred, "(h w) c -> 1 c h w", h=SIZE, w=SIZE)
        y_pred = clip_model.get_image_features(pixel_values=y_pred)
        return ((y_pred - y) ** 2).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(cppn)
    optim.update(grads)
    return loss


SIZE = 224
EPOCHS = 10_000
LEARNING_RATE = 3e-4

# Load CLIP models
clip_model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Create CPPN model
cppn_model = CPPN(rngs=nnx.Rngs(1))

# Visualise defualt reponse
x = jax.numpy.linspace(-5, 5, SIZE)
y = jax.numpy.linspace(-5, 5, SIZE)
grid_in = jax.numpy.meshgrid(x, y)
coords_ind = rearrange(grid_in, "d h w -> (h w) d")
show_image(cppn_model, coords_ind)


# Setup embedding of desired text
inputs = clip_tokenizer(["a picture of a sailboat"], padding=True, return_tensors="jax")
text_features = clip_model.get_text_features(**inputs)

# Iteratively update cppn to improve similarity between output and clip embeddding
loss_history = []
optimizer = nnx.Optimizer(cppn_model, optax.adam(LEARNING_RATE))  # reference sharing
for epoch in (pbar := trange(EPOCHS)):
    loss = train_step(cppn_model, optimizer, coords_ind, text_features)
    pbar.set_description(f"{epoch=} {loss=}")
    loss_history.append(loss)
    if epoch % 100 == 0:
        show_image(cppn_model, coords_ind)
plt.scatter(jax.numpy.arange(len(loss_history)), loss_history)
plt.show()

# Save final
show_image(cppn_model, coords_ind)
