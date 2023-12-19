# import the necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import bisect
from keras.models import load_model

loaded_model = load_model("./cnn_ocr_tmnist_3.h5")

characters = [
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "W",
    "X",
    "Y",
    "Z",
    "!",
    '"',
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "{",
    "|",
    "}",
    "~",
]

label_mapping = {i: char for i, char in enumerate(characters)}


def thresholding(image, threshold_value):
    # Baca gambar menggunakan OpenCV
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Tresholding
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Menampilkan gambar asli dan hasil tresholding
    # cv2_imshow(image)
    # cv2_imshow(binary_image)
    return binary_image


def gaussian_blur(image, kernel_size):
    # Terapkan filter Gaussian Blur menggunakan fungsi GaussianBlur di OpenCV
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred_image


def erosion(image, kernel_size):
    # Elemen kernel untuk operasi erosi
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Operasi erosi menggunakan fungsi morphologyEx di OpenCV
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image


def dilation(image, kernel_size):
    # Elemen kernel untuk operasi dilasi
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Operasi dilasi menggunakan fungsi morphologyEx di OpenCV
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image


def otsu_thresholding(image):
    # Menggunakan metode Otsu untuk menentukan nilai ambang secara otomatis
    _, otsu_threshold = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return otsu_threshold


def try_add_bounding_boxes(
    img,
    oriImage,
    min_contour_area=100,
    dilate_iterations=2,
    min_width=30,
    min_height=30,
    max_width=1000,
    max_height=1000,
    method=cv2.RETR_EXTERNAL,
):
    # Lakukan thresholding
    # _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Lakukan operasi morfologi untuk menghilangkan noise dan menghubungkan karakter
    # kernel = np.ones((3, 3), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=dilate_iterations)

    # Temukan kontur
    contours, _ = cv2.findContours(img, method, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(
        contours, key=lambda x: cv2.boundingRect(x)[0]
    )  # Urutkan berdasarkan koordinat x
    # print(contours)
    # Gambar bounding box pada gambar asli
    img_with_boxes = oriImage.copy()
    cropped_images = []
    for contour in sorted_contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Tambahkan batasan minimum lebar dan tinggi
            if (w > min_width and w < max_width) and (
                h > min_height and h < max_height
            ):
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 5)
                cropped_img = img[y : y + h, x : x + w]
                cropped_images.append(cropped_img)

    return img_with_boxes, cropped_images


def add_bounding_boxes(
    img,
    oriImage,
    min_contour_area=100,
    dilate_iterations=2,
    min_width=30,
    min_height=30,
    method=cv2.RETR_EXTERNAL,
):
    # Lakukan thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Lakukan operasi morfologi untuk menghilangkan noise dan menghubungkan karakter
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=dilate_iterations
    )

    # Temukan kontur
    contours, _ = cv2.findContours(thresh, method, cv2.CHAIN_APPROX_SIMPLE)

    # Gambar bounding box pada gambar asli
    img_with_boxes = oriImage.copy()
    batas_bawah = []
    batas_atas = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Tambahkan batasan minimum lebar dan tinggi
            if w > min_width and h > min_height:
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 5)
                batas_bawah.append(y + h)
                batas_atas.append(y)

    return img_with_boxes, batas_bawah, batas_atas


def crop_bounding_boxes(
    img,
    oriImage,
    min_contour_area=100,
    dilate_iterations=2,
    min_width=30,
    min_height=30,
):
    # Lakukan thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Lakukan operasi morfologi untuk menghilangkan noise dan menghubungkan karakter
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=dilate_iterations
    )

    # Temukan kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Simpan cropped images ke dalam list
    cropped_images = []

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Tambahkan batasan minimum lebar dan tinggi
            if w > min_width and h > min_height:
                # Crop the bounding boxed image
                cropped_img = oriImage[y : y + h, x : x + w]
                cropped_images.append(cropped_img)

    return cropped_images


def img_to_array(img, padding=3):
    # Add padding to the image
    padded_img = cv2.copyMakeBorder(
        img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
    )

    # Resize the padded image
    resized_img = cv2.resize(padded_img, (28, 28))

    # Expand the dimensions to match the model's expected input shape
    img_array = np.expand_dims(resized_img, axis=0)

    # Reshape the array if needed (e.g., for CNNs)
    img_array = img_array.reshape((1, 28, 28, 1))

    return img_array


def resize_image(original_image, new_width, new_height):
    # Resize gambar
    resized_image = cv2.resize(original_image, (new_width, new_height))
    return resized_image


def crop_image(original_image, start_x, start_y, end_x, end_y):
    # Potong gambar menggunakan koordinat tertentu
    cropped_image = original_image[start_y:end_y, start_x:end_x]

    return cropped_image


def block_white_color(original_image, pixel_coordinates, block_size=(10, 10)):
    # Buat salinan gambar untuk diedit
    edited_image = original_image.copy()
    x, y = pixel_coordinates
    width, height = block_size

    # Ambil blok piksel
    block = edited_image[y : y + height, x : x + width]

    # Isi blok dengan warna putih
    block[:, :] = [255, 255, 255]

    return edited_image


def add_padding(
    original_image, top_pad, bottom_pad, left_pad, right_pad, color=(0, 0, 0)
):
    # Dapatkan dimensi gambar
    if len(original_image.shape) > 2:
        height, width, _ = original_image.shape

        # Buat gambar dengan padding
        padded_image = np.ones(
            (height + top_pad + bottom_pad, width + left_pad + right_pad, 3),
            dtype=np.uint8,
        )
        padded_image[:] = color

    else:
        height, width = original_image.shape

        # Buat gambar dengan padding
        padded_image = np.ones(
            (height + top_pad + bottom_pad, width + left_pad + right_pad),
            dtype=np.uint8,
        )

    # Tempelkan gambar asli ke dalam gambar dengan padding
    padded_image[
        top_pad : top_pad + height, left_pad : left_pad + width
    ] = original_image
    return padded_image


def apply_morphology(image, operation, kernel=(150, 1)):
    # Baca gambar menggunakan OpenCV dalam skala abu-abu
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Buat elemen struktur dengan bentuk kotak
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)

    # Terapkan operasi morfologi (erosi atau dilasi) pada gambar
    if operation == "erosion":
        result = cv2.erode(image, structuring_element, iterations=1)
    elif operation == "dilation":
        result = cv2.dilate(image, structuring_element, iterations=1)
    else:
        print("Invalid morphology operation specified.")
        return

    return result


def recognize_ktp(image):
    img_fixed = resize_image(image, 8000, 5500)
    foto = crop_image(img_fixed, 5700, 1000, 7700, 3700)
    img_fixed = block_white_color(img_fixed, (5700, 1000), block_size=(2000, 2700))

    provinsi = crop_image(img_fixed, 0, 0, img_fixed.shape[1] - 100, 400)
    kabupaten = crop_image(
        img_fixed, 0, provinsi.shape[0] - 100, img_fixed.shape[1], 800
    )
    nik = crop_image(img_fixed, 1800, 750, img_fixed.shape[1], 1200)
    left = crop_image(img_fixed, 0, 1200, 2100, img_fixed.shape[0])
    right = crop_image(img_fixed, left.shape[1] + 50, 1200, 5500, img_fixed.shape[0])

    predicted_text = {
        "nik": "",
        "nama": "",
        "ttl": "",
        "jk": "",
        "alamat": "",
    }

    pNik = nik.copy()

    pNik = cv2.cvtColor(pNik, cv2.COLOR_BGR2GRAY)

    pNik = gaussian_blur(pNik, kernel_size=11)
    pNik = thresholding(pNik, 70)
    pNik = erosion(pNik, kernel_size=17)
    pNik = cv2.bitwise_not(pNik)
    box_nik, cropped_nik = try_add_bounding_boxes(
        pNik, nik, min_width=20, min_height=200, method=cv2.RETR_CCOMP
    )

    for c in cropped_nik:
        img_array = img_to_array(c, padding=40)
        predictions = loaded_model.predict(img_array)
        predicted_label = label_mapping[np.argmax(predictions)]
        predicted_text["nik"] += predicted_label

    temp_pLeft = add_padding(
        left,
        top_pad=200,
        bottom_pad=200,
        left_pad=200,
        right_pad=200,
        color=(255, 255, 255),
    )
    pLeft = temp_pLeft.copy()

    pLeft = cv2.cvtColor(pLeft, cv2.COLOR_BGR2GRAY)

    pLeft = thresholding(pLeft, 70)
    pLeft = gaussian_blur(pLeft, kernel_size=9)
    pLeft = otsu_thresholding(pLeft)
    pLeft = erosion(pLeft, kernel_size=7)
    pLeft = dilation(pLeft, kernel_size=5)

    pLeft = apply_morphology(pLeft, operation="erosion", kernel=(100, 1))
    box_left, batas_bawah_left, batas_atas_left = add_bounding_boxes(
        pLeft, temp_pLeft, min_width=500, min_height=50
    )
    cropped_left = cropped_text = crop_bounding_boxes(
        pLeft, temp_pLeft, min_width=500, min_height=50
    )

    temp_pRight = add_padding(
        right,
        top_pad=200,
        bottom_pad=200,
        left_pad=200,
        right_pad=200,
        color=(255, 255, 255),
    )
    pRight = temp_pRight.copy()

    pRight = cv2.cvtColor(pRight, cv2.COLOR_BGR2GRAY)

    pRight = thresholding(pRight, 70)
    pRight = gaussian_blur(pRight, kernel_size=9)
    pRight = otsu_thresholding(pRight)
    pRight = erosion(pRight, kernel_size=7)
    pRight = dilation(pRight, kernel_size=5)

    pRight = apply_morphology(pRight, operation="erosion", kernel=(100, 1))
    box_right, batas_bawah_right, batas_atas_right = add_bounding_boxes(
        pRight, temp_pRight, min_width=500, min_height=50
    )
    cropped_right = cropped_text = crop_bounding_boxes(
        pRight, temp_pRight, min_width=500, min_height=50
    )

    labels = [
        "nama",
        "ttl",
        "jk",
        "alamat",
        "RT/RW",
        "Kel/Desa",
        "Kecamatan",
        "Agama",
        "Status Perkawinan",
        "Pekerjaan",
        "Kewarganegaraan",
        "Berlaku Hingga",
    ]
    batas_atas_left.sort(reverse=False)
    batas_atas_right.sort(reverse=False)
    for i, cr in enumerate(reversed(cropped_right)):
        index_to_insert = bisect.bisect_right(batas_atas_left, batas_atas_right[i])

        label = labels[index_to_insert - 1]

        pCr = cr.copy()

        pCr = cv2.cvtColor(pCr, cv2.COLOR_BGR2GRAY)

        pCr = gaussian_blur(pCr, kernel_size=21)
        pCr = thresholding(pCr, 70)
        pCr = erosion(pCr, kernel_size=17)
        pCr = dilation(pCr, kernel_size=7)
        pCr = cv2.bitwise_not(pCr)
        box_cr, cropped_cr = try_add_bounding_boxes(
            pCr, cr, min_width=20, min_height=100, method=cv2.RETR_EXTERNAL
        )

        for c in cropped_cr:
            if label == "nama":
                img_array = img_to_array(c, padding=35)
                predictions = loaded_model.predict(img_array)
                predicted_label = label_mapping[np.argmax(predictions)]
                predicted_text["nama"] += predicted_label
            elif label == "ttl":
                img_array = img_to_array(c, padding=35)
                predictions = loaded_model.predict(img_array)
                predicted_label = label_mapping[np.argmax(predictions)]
                predicted_text["ttl"] += predicted_label
            elif label == "jk":
                img_array = img_to_array(c, padding=35)
                predictions = loaded_model.predict(img_array)
                predicted_label = label_mapping[np.argmax(predictions)]
                predicted_text["jk"] += predicted_label
            elif label == "alamat":
                img_array = img_to_array(c, padding=35)
                predictions = loaded_model.predict(img_array)
                predicted_label = label_mapping[np.argmax(predictions)]
                predicted_text["alamat"] += predicted_label

    return predicted_text
