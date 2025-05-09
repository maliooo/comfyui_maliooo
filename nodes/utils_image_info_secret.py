import base64
from io import BytesIO
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # pip install cryptography
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import json

SECRET_KEY = b'YourSecretKey123'  # Must be 16, 24, or 32 bytes
ALGORITHM = algorithms.AES(SECRET_KEY)
ENCRYPTED_KEYWORD = "EncryptedStableDiffusionInfo"


def get_base64_from_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string


def extract_image_info(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    metadata = image.info
    return metadata


def save_image_with_encrypted_info(input_image_base64, encrypted_info):
    image_data = base64.b64decode(input_image_base64)
    image = Image.open(BytesIO(image_data))

    encrypted_image = BytesIO()
    image.save(encrypted_image, format="PNG", pnginfo=image.info)

    encrypted_image_data = encrypted_image.getvalue()
    new_image_base64 = base64.b64encode(encrypted_image_data).decode("utf-8")

    return new_image_base64


def encrypt(data):
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data.encode()) + padder.finalize()

    cipher = Cipher(ALGORITHM, modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_bytes = encryptor.update(padded_data) + encryptor.finalize()

    return base64.b64encode(encrypted_bytes).decode("utf-8")


def decrypt(encrypted_data):
    encrypted_bytes = base64.b64decode(encrypted_data)

    cipher = Cipher(ALGORITHM, modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded_data = decryptor.update(encrypted_bytes) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()

    return decrypted_data.decode()


def map_to_string(map_data):
    return json.dumps(map_data)


def string_to_map(string_data):
    return json.loads(string_data)


def main():
    # 加密后文件
    input_image_path = r"/home/zhangxuqi/malio/test/code/imgs/6d911ff8-3810-4b7f-8856-0dc351dcdbf5.png"

    input_image_base64 = get_base64_from_image(input_image_path)

    extracted_info = extract_image_info(input_image_base64)  # 取信息
    print("Extracted Info:", extracted_info['parameters'])


    decrypted_info = decrypt(extracted_info['parameters'])  # 解密信息
    print("Decrypted Info:", decrypted_info)
    return decrypted_info


if __name__ == "__main__":
    info = main()
