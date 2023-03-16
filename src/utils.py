import base64
import io
from PIL import Image


def base64_decode(encode):
    bytes_encoding = bytes(encode, encoding="utf-8")
    image_bytes = base64.b64decode(bytes_encoding)
    image = Image.open(io.BytesIO(image_bytes))
    # image.show()
    return image
