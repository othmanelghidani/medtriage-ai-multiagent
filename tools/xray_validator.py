from PIL import Image
import numpy as np


def validate_xray_image(image_path):
    """
    Basic validation to check whether the uploaded image looks like a chest X-ray.
    """

    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        height, width, channels = image_array.shape

        if height < 100 or width < 100:
            return False, "Image resolution is too small."

        r_channel = image_array[:, :, 0]
        g_channel = image_array[:, :, 1]
        b_channel = image_array[:, :, 2]

        rg_diff = np.mean(np.abs(r_channel - g_channel))
        rb_diff = np.mean(np.abs(r_channel - b_channel))
        gb_diff = np.mean(np.abs(g_channel - b_channel))

        color_difference = (rg_diff + rb_diff + gb_diff) / 3

        if color_difference > 15:
            return False, "Uploaded image does not appear to be a chest X-ray."

        brightness = np.mean(image_array)

        if brightness < 20 or brightness > 240:
            return False, "Invalid X-ray intensity range."

        return True, None

    except Exception as e:
        return False, str(e)