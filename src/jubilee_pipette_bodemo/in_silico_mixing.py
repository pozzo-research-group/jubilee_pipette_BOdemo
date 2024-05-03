import mixbox
import numpy as np
from PIL import Image
import io


# Acts like a drop-in replacement for the jubilee_protocols module here so we can test code and such without firing up a Jubilee

def sample_point(query_point, in_silico_colors):
    """
    In-silico analog to sample_point in jubilee_protocols.py
    """
    assert len(query_point) == len(in_silico_colors), 'Number of colors in query_point must match number of colors in `in_silico_colors`'
    mixed_rgb = mix_colors(in_silico_colors, query_point)
    image = synthetic_image(mixed_rgb)

    return mixed_rgb, image



def mix_colors(colors, weights):
    """
    Mix colors using the mixbox package.
    
    Args:
    - colors: List of RGB tuples representing the colors to mix.
    - weights: List of weights corresponding to each color.
    
    Returns:
    - Tuple containing the mixed color in RGB space.
    """
    colors = [mixbox.rgb_to_latent(color) for color in colors]
    mixed_colors = [0]*mixbox.LATENT_SIZE

    for color, weight in zip(colors, weights):
        for i in range(mixbox.LATENT_SIZE):
            mixed_colors[i] += color[i] * weight
    
    mixed_rgb = mixbox.latent_to_rgb(mixed_colors)

    return mixed_rgb

def synthetic_image(mixed_rgb, image_size = (500,500)):
    """
    Generate jpg image bytes with specified color
    """

    rgb = [np.uint8(i) for i in mixed_rgb]
    image_arr = np.ones([image_size[0], image_size[1], 3], dtype = np.uint8)*rgb
    image = Image.fromarray(image_arr, 'RGB')

    synth_bytes = io.BytesIO()
    image.save(synth_bytes, format = 'JPEG')
    return synth_bytes.getvalue()