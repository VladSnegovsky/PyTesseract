from PIL import Image
import pytesseract
import os
from openCL_image.main import opencl_contrast_change, opencl_apply_blur, opencl_enhance_brightness



def delete_image(path):
    try:
        os.remove(path)
        print(f"Файл {path} успішно видалено.")
    except FileNotFoundError:
        print(f"Файл {path} не знайдено.")
    except Exception as e:
        print(f"Виникла помилка при видаленні файлу {path}: {e}")



# Set the path to the tesseract program if it is not included in PATH
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def recognize_handwritten_text(image_path):
    # Opening the image
    image = Image.open(image_path)

    # Converting an image to text using tesseract
    # boxes = pytesseract.image_to_boxes(img, config='--oem 3')
    # text = pytesseract.image_to_string(img, config='--psm 6')
    # text = pytesseract.image_to_string(img, lang='eng')
    text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')

    return text

# Provide the path to your handwritten image
image_path = '/Users/vladsnegovsky/Documents/University/Year2/Диплом/text1.png'

# cntrst = 1.5
# image_opencl_contrast = opencl_contrast_change(image_path, cntrst)
brightness = 2
brightness_image = opencl_enhance_brightness(image_path, brightness)
# kernel_size = 1
# image_opencl_blur = opencl_apply_blur(image_path, kernel_size)

# Text recognition and result output
result = recognize_handwritten_text(brightness_image)
print(result)

# delete_image(image_opencl_contrast)
# delete_image(image_opencl_blur)
delete_image(brightness_image)