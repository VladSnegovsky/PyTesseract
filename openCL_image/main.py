import cv2
import numpy as np
import pyopencl as cl



# Функція для перетворення зображення за допомогою OpenCL
def _enhance_contrast_opencl(image, contrast_multiplier):

    # Ініціалізація OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Створюємо буфери для вхідного та вихідного зображення
    image_shape = image.shape
    input_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image)
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, image.nbytes)

    # Визначення розмірів роботи та блоків
    global_size = (image_shape[1], image_shape[0])
    local_size = None

    # Завантаження та компіляція ядра OpenCL
    kernel_source = f"""
        __kernel void enhance_contrast(__global uchar* input, __global uchar* output, float contrast_multiplier) {{
            int gid_x = get_global_id(0);
            int gid_y = get_global_id(1);

            uchar pixel_value = input[gid_y * {image_shape[1]} + gid_x];
            
            // Формула для підвищення контрасту
            float enhanced_value = (pixel_value - 128) * contrast_multiplier + 128;

            // Обмеження пікселів у діапазоні 0-255
            enhanced_value = fmin(255.0, fmax(0.0, enhanced_value));

            // Запис результату у вихідне зображення
            output[gid_y * {image_shape[1]} + gid_x] = (uchar)enhanced_value;
        }}
    """

    program = cl.Program(context, kernel_source).build()

    # Запуск ядра OpenCL
    program.enhance_contrast(queue, global_size, local_size, input_buffer, output_buffer, np.float32(contrast_multiplier))

    # Отримання результату з буфера
    result = np.empty_like(image)
    cl.enqueue_copy(queue, result, output_buffer).wait()

    return result



def opencl_contrast_change(img_path, contrast_multiplier):
    # Загрузка зображення
    image_path = img_path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Підвищення контрасту
    contrast_processed_image = _enhance_contrast_opencl(image, contrast_multiplier)

    # Збереження обробленого зображення
    output_path_contrast = '/Users/vladsnegovsky/Documents/University/Year2/Tesseract/openCL_image/image_contrast.png'
    cv2.imwrite(output_path_contrast, contrast_processed_image)

    return output_path_contrast



# Розмиття
def _apply_blur_opencl(image, kernel_size):

    # Ініціалізація OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Створюємо буфери для вхідного та вихідного зображення
    image_shape = image.shape
    input_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image)
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, image.nbytes)

    # Визначення розмірів роботи та блоків
    global_size = (image_shape[1], image_shape[0])
    local_size = None

    # Завантаження та компіляція ядра OpenCL
    kernel_source = f"""
        __kernel void apply_blur(__global uchar* input, __global uchar* output, int kernel_size) {{
            int gid_x = get_global_id(0);
            int gid_y = get_global_id(1);

            float sum = 0.0;
            int count = 0;

            for (int i = -kernel_size/2; i <= kernel_size/2; i++) {{
                for (int j = -kernel_size/2; j <= kernel_size/2; j++) {{
                    int x = gid_x + i;
                    int y = gid_y + j;

                    if (x >= 0 && x < {image_shape[1]} && y >= 0 && y < {image_shape[0]}) {{
                        sum += input[y * {image_shape[1]} + x];
                        count++;
                    }}
                }}
            }}

            output[gid_y * {image_shape[1]} + gid_x] = sum / count;
        }}
    """

    program = cl.Program(context, kernel_source).build()

    # Запуск ядра OpenCL
    program.apply_blur(queue, global_size, local_size, input_buffer, output_buffer, np.int32(kernel_size))

    # Отримання результату з буфера
    result = np.empty_like(image)
    cl.enqueue_copy(queue, result, output_buffer).wait()

    return result



def opencl_apply_blur(img_path, kernel_size):
    # Загрузка зображення
    image_path = img_path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Застосування розмиття
    blur_processed_image = _apply_blur_opencl(image, kernel_size)

    # Збереження обробленого зображення
    output_path_blur = '/Users/vladsnegovsky/Documents/University/Year2/Tesseract/openCL_image/image_blur.png'
    cv2.imwrite(output_path_blur, blur_processed_image)

    return output_path_blur




def _enhance_brightness_opencl(image, brightness_multiplier):
    # Ініціалізація OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Створюємо буфери для вхідного та вихідного зображення
    image_shape = image.shape
    input_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image)
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, image.nbytes)

    # Визначення розмірів роботи та блоків
    global_size = (image_shape[1], image_shape[0])
    local_size = None

    # Завантаження та компіляція ядра OpenCL
    kernel_source = f"""
        __kernel void enhance_brightness(__global uchar* input, __global uchar* output, float brightness_multiplier) {{
            int gid_x = get_global_id(0);
            int gid_y = get_global_id(1);

            uchar pixel_value = input[gid_y * {image_shape[1]} + gid_x];
            
            // Формула для повышения яркости
            float enhanced_value = pixel_value * brightness_multiplier;

            // Ограничение значений пикселя в диапазоне 0-255
            enhanced_value = fmin(255.0, fmax(0.0, enhanced_value));

            // Запись результата в выходное изображение
            output[gid_y * {image_shape[1]} + gid_x] = (uchar)enhanced_value;
        }}
    """

    program = cl.Program(context, kernel_source).build()

    # Запуск ядра OpenCL
    program.enhance_brightness(queue, global_size, local_size, input_buffer, output_buffer, np.float32(brightness_multiplier))

    # Отримання результату з буфера
    result = np.empty_like(image)
    cl.enqueue_copy(queue, result, output_buffer).wait()

    return result



def opencl_enhance_brightness(img_path, brightness_multiplier):
    # Загрузка зображення
    image_path = img_path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Застосування підвищення яскравості
    brightened_image = _enhance_brightness_opencl(image, brightness_multiplier)

    # Збереження обробленого зображення
    output_path_contrast = '/Users/vladsnegovsky/Documents/University/Year2/Tesseract/openCL_image/image_brightness.png'
    cv2.imwrite(output_path_contrast, brightened_image)

    return output_path_contrast

