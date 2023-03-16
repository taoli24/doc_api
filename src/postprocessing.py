import numpy as np
from PIL import Image, ImageDraw
import pytesseract
import matplotlib.pyplot as plt
import pandas as pd
from src.bbox_manipulation import process_boxes, enlarge_boxes
import cv2
import json


def output_json(dataframe):
    output_dict = {"0": list(dataframe.columns)}
    value = dataframe.values
    for i in range(dataframe.shape[0]):
        output_dict[str(i + 1)] = list(value[i])
    with open("output_test.json", "w") as outfile:
        json.dump(output_dict, outfile)
    return output_dict


def remove_line_using_cv_2(binary_image, vertical_size, horizontal_size):
    binary_image = cv2.bitwise_not(binary_image)
    horizontal = np.copy(binary_image)
    vertical = np.copy(binary_image)
    h = horizontal.shape[1]
    horizontal_size = h // horizontal_size
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    w = vertical.shape[0]
    verticalsize = w // vertical_size
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    res = binary_image - vertical - horizontal
    res[res <= 0] = 0
    res_image = cv2.bitwise_not(res)
    edges = cv2.adaptiveThreshold(res_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.dilate(edges, kernel)
    smooth = np.copy(res_image)
    smooth = cv2.blur(smooth, (1, 1))
    (rows, cols) = np.where(edges != 0)
    res_image[rows, cols] = smooth[rows, cols]
    res_image = cv2.fastNlMeansDenoising(res_image, None, 20, 22, 66)
    edges = cv2.adaptiveThreshold(res_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.dilate(edges, kernel)
    smooth = np.copy(res_image)
    smooth = cv2.blur(smooth, (1, 1))
    (rows, cols) = np.where(edges != 0)
    res_image[rows, cols] = smooth[rows, cols]
    return res_image


def remove_line_using_cv(image, horizontal_threshold=160, vertical_threshold=50):
    h, w = image.shape
    thresh_vertical = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // horizontal_threshold))
    detected_lines = cv2.morphologyEx(thresh_vertical, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    thresh_hort = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // vertical_threshold, 1))
    detected_lines = cv2.morphologyEx(thresh_hort, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.dilate(edges, kernel)
    smooth = np.copy(image)
    smooth = cv2.blur(smooth, (1, 1))
    (rows, cols) = np.where(edges != 0)
    image[rows, cols] = smooth[rows, cols]
    return image


def rotated_image(binary_image):
    edges = cv2.Canny(binary_image, 20, 150, apertureSize=5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)
    k = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (x2 - x1) == 0:
            continue
        if ((y2 - y1) / (x2 - x1) > 1) | ((y2 - y1) / (x2 - x1) < -1):
            continue
        k.append((y2 - y1) / (x2 - x1))
    h, w = binary_image.shape[0], binary_image.shape[1]
    center = (w / 2, h / 2)
    angle = np.arctan(np.median(k)) * 180 / np.pi
    if abs(angle) > 0.2:
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(binary_image, rot_mat, dsize=(w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    else:
        result = binary_image
    return result


def detect_noise(header_image, window_size=5, threshold=14):
    h, w = header_image.shape
    header_image = cv2.resize(header_image, (w, 500), cv2.INTER_AREA)
    header_image = cv2.threshold(header_image, 200, 255, cv2.THRESH_BINARY)[1]
    points = []
    for i in range(window_size // 2, header_image.shape[0] - window_size // 2):
        for j in range(window_size // 2, header_image.shape[1] - window_size // 2):
            if header_image[i][j] == 255:
                continue
            else:
                if (header_image[i - window_size // 2:i + window_size // 2 + 1,
                    j - window_size // 2:j + window_size // 2 + 1] == 0).all():
                    continue
                else:
                    if np.count_nonzero(header_image[i - window_size // 2:i + window_size // 2 + 1,
                                        j - window_size // 2:j + window_size // 2 + 1] == 255) > threshold:
                        points.append([i, j])
    if len(points) > 5000:
        return True, points
    else:
        return False, points


def denoise(header_image, points):
    h, w = header_image.shape
    header_image = cv2.resize(header_image, (w, 500), cv2.INTER_AREA)
    for i in points:
        header_image[i[0], i[1]] = 255
    edges = cv2.adaptiveThreshold(header_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.dilate(edges, kernel)
    smooth = np.copy(header_image)
    smooth = cv2.blur(smooth, (1, 1))
    (rows, cols) = np.where(edges != 0)
    header_image[rows, cols] = smooth[rows, cols]
    denoise_image = cv2.resize(header_image, (w, h), cv2.INTER_AREA)
    denoise_image = cv2.threshold(denoise_image, 200, 255, cv2.THRESH_BINARY)[1]
    return denoise_image


def draw_boxes(boxes: list, image: Image):
    """
    :param boxes:
    :param image:
    :return:
    """
    output_image = image.copy()
    image_draw = ImageDraw.Draw(output_image)
    for box in boxes:
        image_draw.rectangle(box, outline=(255, 0, 0), fill=None)
    return output_image


def clean_lines_big_table_structure(image: np.ndarray, thres=0.5):
    """
    Remove lines if there is big table structure shown in the invoice
    :param image: ndarray
    :return: image array
    """
    output = image.copy()
    h, w = image.shape
    for i in range(w):
        if np.median(output[:, i]) == 0:
            return remove_lines(remove_lines(output), thres, axis=1)
    return output


def get_h_lines(image: np.ndarray, line_margin: int, padding=0):
    """
    This function get horizontal lines that separate table elements, it can be used
    on entire documents as well to separate block of elements using a bigger margin.
    :param padding: default 5
    :param image: original size image
    :param line_margin: line margins
    :return: all horizontal lines define upper and lower boundary of page elements
    """
    h, w = image.shape
    black_pixels = image == 0
    is_upper_line = True if np.sum(black_pixels[0: line_margin, :]) == 0 else False
    h_lines = [0] if not is_upper_line else []

    for i in range(h):
        if np.sum(black_pixels[i:i + line_margin + 1, :]) != 0 and is_upper_line:
            h_lines.append(i - padding + line_margin - 1 if i - padding + line_margin - 1 > 0 else 0)
            is_upper_line = False

        if np.sum(black_pixels[i:i + line_margin + 1, :]) == 0 and not is_upper_line:
            h_lines.append(i + padding)
            is_upper_line = True

    h_lines.append(h) if len(h_lines) % 2 != 0 else None

    return h_lines


def get_v_lines(image: np.ndarray, gap: int, padding=5):
    """
    This function get vertical lines that separate table elements, it should be used
    on small section after the bigger sections have been separated by horizontal lines
    :param padding: default 5
    :param image: original size image
    :param gap: gap between elements
    :return: all vertical lines define left and right boundary of line elements
    """
    h, w = image.shape
    black_pixels = image == 0
    is_left_line = True if np.sum(black_pixels[:, 0: gap]) == 0 else False
    v_lines = [0] if not is_left_line else []

    for i in range(w):
        if np.sum(black_pixels[:, i:i + gap]) != 0 and is_left_line:
            v_lines.append(i + gap - padding)
            is_left_line = False

        if np.sum(black_pixels[:, i:i + gap]) == 0 and not is_left_line:
            v_lines.append(i + padding)
            is_left_line = True

    v_lines.append(w) if len(v_lines) % 2 != 0 else None

    return v_lines


def form_and_get_page_sections(image, h_lines):
    """
    This function take in y locations of horizontal lines and forming and return large page sections
    :param image: array like images
    :param h_lines: list of y coordinates
    :return: bounding boxes of large page sections
    """
    section_boxes = []

    for i in range(0, len(h_lines), 2):
        section_boxes.append([0, h_lines[i], image.shape[1], h_lines[i + 1]])

    return section_boxes


def get_table_items(labels, bounding_box=list):
    """
    find all table items
    """
    table_bounding_box = []
    table_label = []
    for i in range(len(labels)):
        if "line" in labels[i].lower() or "header" in labels[i].lower():
            table_label.append(labels[i])
            table_bounding_box.append(bounding_box[i])
    return table_bounding_box, table_label


def get_line_margin(table_bbox, table_labels):
    """

    :param table_bbox:
    :param table_labels:
    :return:
    """
    lw, lh = 1000, 0

    item_boxes = [box for box, label in zip(table_bbox, table_labels) if "line" in label.lower()]

    boxes = enlarge_boxes(item_boxes, lh, lw)

    boxes = process_boxes(boxes)

    boxes.sort(key=lambda x: x[1])
    # print(np.array([boxes[i + 1][1] - boxes[i][3] for i in range(len(boxes) - 1))]))
    line_margin = 10 if len(boxes) == 1 else int(
        np.mean(np.array([boxes[i + 1][1] - boxes[i][3] for i in range(len(boxes) - 1)]))) \
        if 10 < int(np.mean(np.array([boxes[i + 1][1] - boxes[i][3] for i in range(len(boxes) - 1)]))) < 15 else 13
    section_margin = int(line_margin * 2.2)

    return line_margin, section_margin


def get_space_margin(table_bbox, table_labels):
    """
    Get token to token distance, and base on token distance, calculate distance between sections
    :param table_bbox: unnormalised table bounding boxes
    :param table_labels: labels of table bounding boxes
    :return: distance to be used to detect individual line items
    """

    line_item_box = [list(map(int, box)) for box, label in zip(table_bbox, table_labels) if "line" in label.lower()]

    # sort lines into position
    line_item_box.sort(key=lambda x: x[1])

    # Separate lines
    lines, line = [], []
    new_line = True
    line_y = -np.inf
    for box in line_item_box:
        if new_line:
            line = [box]
            new_line = False
            line_y = box[1]
        else:
            if box[1] - line_y <= 5:
                line.append(box)
            else:
                new_line = True
                lines.append(line)

    # Add line to lines if only one line detected
    lines.append(line) if len(lines) == 0 else None

    spaces = []

    for line in lines:
        _line = sorted(line, key=lambda x: x[0])
        for i in range(1, len(_line)):
            if 0 < _line[i][0] - _line[i - 1][2] < 30:
                spaces.append(_line[i][0] - _line[i - 1][2])
    if len(spaces) == 0:
        return 15
    else:
        space = int(np.mean(spaces)) + 5

    return space


def get_header_boxes(table_bbox, table_labels):
    """
    Get all bounding boxes for headers
    :param table_bbox:
    :param table_labels:
    :return: Bounding boxes of headers
    """
    return [list(map(int, box)) for box, label in zip(table_bbox, table_labels) if "header" in label.lower()]


def get_line_one_boxes(table_bbox, table_labels):
    """
    Get all bounding boxes for first line items
    :param table_bbox:
    :param table_labels:
    :return: Bounding boxes of first line items
    """
    item_boxes = [list(map(int, box)) for box, label in zip(table_bbox, table_labels) if "line" in label.lower()]
    item_boxes.sort(key=lambda x: x[1])

    line_item_box_median_height = np.median(np.asarray(item_boxes)[:, 3] - np.asarray(item_boxes)[:, 1])

    first_line = []
    y = item_boxes[0][1]

    for box in item_boxes:
        if box[1] - y <= line_item_box_median_height:
            first_line.append(box)

    return first_line


def get_header_y3(image, table_boxes, table_labels):
    header_box = get_header_boxes(table_boxes, table_labels)
    first_line_box = get_line_one_boxes(table_boxes, table_labels)

    # If there are more than 10 bounding boxes in first line item,
    # This give enough confident that this is valid line item
    # We can use this line to remove annoying headers which touching the line items

    # if len(first_line_box) >= 10:
    #     line_y1 = min(box[1] for box in first_line_box)
    #
    #     # Check for any headers that pass this y
    #     header_box = [box for box in header_box if box[3] < line_y1]

    header_median_y = np.median(np.asarray(header_box)[:, 3])

    median_header_height = int(np.median(np.asarray(header_box)[:, 3] - np.asarray(header_box)[:, 1]))
    header_box = [box for box in header_box if abs(box[3] - header_median_y) < 2.2 * median_header_height]
    header_y3 = np.max(np.asarray(header_box)[:, 3])

    final_header_y3 = header_y3
    for i in range(header_y3, header_y3 + 11):
        if np.sum(image[i, :] == 0) == 0:
            final_header_y3 = i
            break

    if header_y3 == final_header_y3:
        final_header_y3 += 1
    else:
        final_header_y3 += 2

    return final_header_y3


def get_header_top_y(image, unnormal_table_boxes, table_labels, threshold):
    is_black = image == 0
    header_boxes = get_header_boxes(unnormal_table_boxes, table_labels)
    header_box_heights = [int(box[3] - box[1]) for box in header_boxes]
    median_header_height = np.mean(header_box_heights)

    header_boxes = [box for box in header_boxes if
                    abs(box[1] - np.median(np.asarray(header_boxes)[:, 1])) < 3 * median_header_height]

    # Model need to predict most headers
    # median_header_y1 = np.median(header_boxes[:, 1])

    header_top = int(np.min(np.asarray(header_boxes)[:, 1])) - 3

    # for i in range(header_top, -1, -1):
    #     if np.sum(is_black[i - threshold:i, :]) == 0:
    #         header_top = i - 1
    #         break

    return header_top


def get_line_items_section(image, table_bbox, tabel_labels, page_sections: list, line_margin):
    """
    Return line_item_table section
    :param image:
    :param line_margin:
    :param tabel_labels:
    :param table_bbox:
    :param page_sections:
    :return: line_item_table bbox
    """

    res = "TABLE ERROR!"

    # sometimes, line_items are incorrectly classified.
    # In this situation, wrong section might be used as line table

    page_sections.sort(key=lambda x: x[3], reverse=True)

    line_item_boxes = [box for i, box in enumerate(table_bbox) if "line" in tabel_labels[i].lower()]

    is_black = image == 0
    header_boxes = get_header_boxes(table_bbox, tabel_labels)
    # min_line_y = min(np.asarray(header_boxes)[:, 1])
    min_line_y = get_header_top_y(image, table_bbox, tabel_labels, line_margin)
    # min_line_y = get_header_top_y(image, table_bbox, tabel_labels, line_margin)
    max_line_y = max(np.asarray(line_item_boxes)[:, 3])
    for section in page_sections:
        if len([box for box in table_bbox if section[1] < box[3] < section[3]]) / len(table_bbox) > 0.2:
            res = section
            break
        else:
            # remove boxes from this section
            line_item_boxes = [box for box in line_item_boxes if box[3] < section[1]]

    if not isinstance(res, list):
        res = [0, min_line_y, page_sections[0][2], max_line_y]
    else:
        res[3] = max_line_y \
            if res[3] < np.max(np.asarray(line_item_boxes)[:, 3]) else res[3]

    for i in range(min_line_y, image.shape[0]):
        if np.sum(is_black[i, :]) != 0:
            res[1] = i - 1
            break

    # sometimes header are far away from line_items,
    # we need to check if headers are included in the line_item table
    # if not, we should manually change the y-axis value so headers are within the range

    # header_boxes = get_header_boxes(table_bbox, tabel_labels)
    # correct_count = 0

    # for header in header_boxes:
    #     if res[1] < header[3] < res[3]:
    #         correct_count += 1
    #
    # # Calculate accuracy of headers within range
    # # Do not do this step if no header was detected
    # if len(header_boxes) != 0:
    #     if correct_count / len(header_boxes) < 0.75:
    #         res[1] = np.median(np.asarray(header_boxes)[:, 1]) - 1  # 3 is padding for heading y top

    res = list(map(int, res))
    return res


def remove_lines(image: np.ndarray, thres=0.7, axis=0):
    """
    This function remove lines from images
    :param thres:
    :param image: ndarray
    :param axis: 0 or 1
    :return: images with lines removed
    """
    h, w = image.shape
    output = image.copy()

    if axis == 0:
        for i in range(h):
            output[i - 3: i + 4, :] = 255 if np.sum(output[i, :] == 0) / w >= thres else output[i - 3: i + 4, :]

    if axis == 1:
        for i in range(w):
            output[:, i - 4: i + 4] = 255 if np.sum(output[:, i] == 0) / h >= thres else output[:, i - 4: i + 4]

    return output


def bbox_crop_ndarray(image, bbox):
    """
    This function used bounding box information to crop ndarray image
    :param image:
    :param bbox:
    :return: cropped ndarray image
    """
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def split_sections(image, lines, axis=0):
    """
    This function split page sections into list of arrays by using split line information
    :param image: ndarray
    :param lines: list of split lines
    :param axis: 0 or 1
    :return: list of cropped page sections
    """
    h, w = image.shape

    if len(lines) % 2 != 0:
        raise ValueError("Split lines should contain even number of elements.")

    sections = None

    if axis == 0:
        sections = [image[lines[i] - 3 if lines[i] > 3 else lines[i]: lines[i + 1] + 3, 0: w] for i in
                    range(0, len(lines), 2)]

    if axis == 1:
        sections = [image[0: h, lines[i] - 3 if lines[i] > 3 else lines[i]:lines[i + 1] + 3] for i in
                    range(0, len(lines), 2)]

    return sections


def split_lines(image):
    """
    This function split page sections into list of arrays by using split line information
    :param image: ndarray
    :return: list of cropped page sections
    """
    h, w = image.shape

    padding = np.full((3, w), 255, dtype='uint8')

    lines = get_h_lines(image, 1, 0)

    sections = [image[lines[i]:lines[i + 1], :] for i in range(0, len(lines), 2)]
    sections = [np.concatenate((padding, section, padding), axis=0) for section in sections]

    return sections


def check_for_invalid_header_section(input_header_image):
    """
    This function remove small texts sometimes appear above headers
    :param input_header_image:
    :return: new header image
    """
    h_lines = get_h_lines(input_header_image, 1)

    table_start = h_lines[0]
    table_start_h_line = 0

    # np_header_box = np.asarray(header_box) - start
    # min_header = np.min(np_header_box[:, 3])
    # print(min_header)

    line_heights = [h_lines[i + 1] - h_lines[i] for i in range(0, len(h_lines), 2)]

    for i in range(0, len(h_lines), 2):
        h, w = input_header_image[h_lines[i]:h_lines[i + 1], :].shape

        is_black_pixels = input_header_image[h_lines[i]:h_lines[i + 1], :] == 0

        # Check is more than half the space is white pixels,
        # If so, the section is invalid header section

        if (np.sum(is_black_pixels[:, 0: int(w * 0.5)]) == 0 or np.sum(
                is_black_pixels[:, int(0.5 * w): w]) == 0) and h_lines[i + 1] - h_lines[i] <= 18:  # small text size
            continue
        else:
            # print(h_lines[i + 1] - h_lines[i])
            table_start_h_line = i
            table_start = h_lines[table_start_h_line]
            break

    output_header = input_header_image[h_lines[table_start_h_line]:, :]
    header_max = output_header.shape[0]

    return output_header, header_max


def ocr_scan_image(image, second_scan=False):
    """
    Ocr read image and return scanned text
    :param second_scan: bool, if performing second scan
    :param image: ndarray
    :return: str --psm 10
    """
    ocr_config = "-c tessedit_char_blacklist=[]{}~»°§¢|!" if not second_scan \
        else "--psm 10 -c tessedit_char_blacklist=[]{}~»°§¢|!"
    return pytesseract.image_to_string(image, config=ocr_config)


def plot_items(images, texts):
    """
    Make plots of extracted texts and their corresponding images
    :param images: ndarray image
    :param texts: extracted text from images
    :return: None
    """
    fig, axes = plt.subplots(1, len(images), figsize=(30, 8), dpi=100)
    plt.subplots_adjust(wspace=0.8, hspace=0.6)

    for ind, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image)
        ax.set_title(texts[ind])
    plt.show()


def get_split_line_items_header(image: np.ndarray, v_lines: list, plot=False):
    """
    Split image and extract text, make plot is optional
    :param image: ndarray the table image
    :param v_lines: vertical lines split sections
    :param plot: boolean
    :return: list of extracted text items
    """
    texts = []
    image = image.copy()
    _, w = image.shape
    padding_space = np.full((2, w), 255, dtype="uint8")
    image = np.concatenate((padding_space, image, padding_space), axis=0)
    item_images = split_sections(image, v_lines, axis=1)

    for image in item_images:
        text = ocr_scan_image(image)
        text = ocr_scan_image(image, second_scan=True) if text == "" else text
        text = text.strip().replace("\n", "")
        text = "Qty" if text == "aty" else text
        text = "Item" if text == "(tem" else text
        text = "GST" if text == "1Izh=" else text
        text = "Fee" if text == "Fes=" else text
        text = "Item" if text == "ltem" else text
        text = "Item No" if text == "Ttem No" else text

        texts.append(text)

    x_range_text = [(v_lines[i], v_lines[i + 1]) for i in range(0, len(v_lines), 2)]

    reduced_x_range = []

    for ind, (header_image, x_range) in enumerate(zip(item_images, x_range_text)):
        is_black_pixel = header_image == 0

        if np.sum(is_black_pixel) == 0:
            texts[ind] = ""

        h, w = header_image.shape

        left_space = 0
        right_space = 0

        # scan from left to right
        for i in range(0, w):
            if np.sum(is_black_pixel[:, i]) < 2:
                left_space += 1
            else:
                break

        if left_space != w:
            # scan from right to left
            for j in range(w - 1, -1, -1):
                if np.sum(is_black_pixel[:, j]) < 2:
                    right_space += 1
                else:
                    break

        if right_space != w:
            reduced_x_range.append([x_range[0] + left_space, x_range[1] - right_space])
        else:
            reduced_x_range.append([np.mean(x_range) - 2, np.mean(x_range) + 2])

    if plot:
        plot_items(item_images, texts)

    # reduced_x_range = x_range_text

    # remove space and new line
    return texts, reduced_x_range


def get_split_line_items(image: np.ndarray, v_lines: list, plot=False):
    """
    Split image and extract text, make plot is optional
    :param image: ndarray
    :param v_lines: vertical lines split sections
    :param plot: boolean
    :return: list of extracted text items
    """
    texts = []
    item_images = split_sections(image, v_lines, axis=1)
    for image in item_images:
        text = ocr_scan_image(image)
        text = ocr_scan_image(image, second_scan=True) if text == "" else text
        texts.append(text)

    texts = [text.strip().replace("\n", "") for text in texts]

    x_range_text = [(v_lines[i], v_lines[i + 1]) for i in range(0, len(v_lines), 2)]

    if plot:
        plot_items(item_images, texts)

    # remove space and new line
    return texts, x_range_text


def process_line_items(images: list, gap, padding=15):
    """
    Returned a list of scanned lines
    :param images: ndarray
    :param gap: horizontal gap between items
    :param padding: padding size to enlarge boxes to increase ocr performance
    :return: list of lines
    """
    line_items = []

    for image in images:
        v_lines = get_v_lines(image, gap, padding)
        texts, x_range = get_split_line_items(image, v_lines, plot=False)
        line_items.append((texts, x_range))

    return line_items


def process_dataframes(headers, line_items):
    """
    This function returns a dataframe
    :param headers: list of headers
    :param line_items: list of line_items
    :return: df
    """

    header_count = len(headers)
    items_count = max(len(line_item) for line_item in line_items)

    assert header_count == items_count, f"Header count: {header_count} Table items count: {items_count}"

    line_items = [line_item for line_item in line_items if len(line_item) == header_count]

    df = pd.DataFrame(data=line_items, columns=headers)

    return df


def process_dataframes_v2(headers, line_items, h_lines):
    """
    This function returns a dataframe
    :param h_lines:
    :param headers: list of headers
    :param line_items: list of line_items
    :return: df
    """
    y_pos = [h_lines[i] + h_lines[i + 1] / 2 for i in range(0, len(h_lines), 2)]
    x_pos = [0 for y in y_pos]

    y_pos = pd.Series(y_pos)
    x_pos = pd.Series(x_pos)

    header_count = len(headers)
    items_count = max(len(line_item) for line_item in line_items)

    assert header_count == items_count, f"Header count: {header_count} Table items count: {items_count}"

    line_items = [line_item for line_item in line_items if len(line_item) == header_count]

    line_items.insert(0, headers)

    df = pd.DataFrame(data=line_items)

    column_number_counts = {}
    # Calculate number of clusters
    # for i, row in df.iterrows():
    #     for key, value in row.items():
    #         if re.search(r"^\$?[0-9,]+\.?\d*", value) or re.search(r"\$?[0-9,]+\.?\d*$", value):
    #             print("match found")
    #             num_clusters += 1
    #             break

    # df = df.applymap(lambda x: 1 if x != "" else 0)

    df["x"] = x_pos
    df["y"] = y_pos

    # clustering
    # kmeans = KMeans(n_clusters=10, random_state=0).fit(df[["x", "y"]].values)
    # df["cluster_labels"] = pd.Series(kmeans.labels_)

    return df


def head_split(image, max_y_input, threshold):
    """
    :param max_y_input:
    :param image: Array
    :param threshold: Integer
    :return: the x-axis of bounding box of every header
    """

    min_header_y = 0
    max_header_y = max_y_input

    distance = 0
    for j in range(max_header_y, image.shape[0]):
        distance = distance + 1 if np.sum(image[j, :] == 0) == 0 else 0

        if distance > 6:
            max_header_y = j - 6
            break

        # if distance > 12:
        #     max_header_y = j - 12
        #     break

    head_image = remove_lines(image[min_header_y:max_header_y, :], axis=1, thres=0.9)
    plt.imshow(head_image)
    plt.show()
    header_v_lines = get_v_lines(head_image, threshold, padding=0)

    x_head = [[header_v_lines[i], header_v_lines[i + 1]] for i in range(0, len(header_v_lines), 2)]

    return x_head, max_header_y, head_image


def remove_wrong_label(table_label, table_bounding_box, head_y):
    for i in range(len(table_label)):
        if table_bounding_box[i][1] >= head_y[0] | table_bounding_box[i][3] <= head_y[1]:
            if "header" not in table_label[i]:
                table_label[i] = "header"
    return table_label


def detect_table_line_by_line_items(image_no_lines, section_space, word_space, item_table_height):
    detect_image = image_no_lines[1:, :]

    h, w = detect_image.shape
    max_distance_line_list = [word_space, section_space]
    distance_line = 0
    max_distance_line = 2
    y_max_row_step_1 = -1
    threshold = word_space

    previous_row = -1
    flag = True

    for i in range(h):
        if np.count_nonzero(detect_image[i, :int(2 * w / 5)] == 0) <= 10:
            distance_line += 1
        else:
            # ignore the distance between header and line item.
            if flag:
                distance_line = 0
                flag = False
                if previous_row == -1:
                    previous_row = i
                continue
            else:
                # ignore the small distance_line which might affact by outliers.
                if (distance_line > 0) & (distance_line <= 2):
                    distance_line = 0
                    continue
                else:
                    if previous_row == -1:
                        previous_row = i
                    else:

                        if max_distance_line > threshold:
                            threshold = max_distance_line

                        if ((i - previous_row) > 1.1 * threshold) & (threshold > 1.5 * word_space):
                            y_max_row_step_1 = i - distance_line
                            break
                        elif (i - previous_row) > 50:
                            y_max_row_step_1 = i - distance_line
                            break
                        else:
                            previous_row = i

                if distance_line > max_distance_line:
                    max_distance_line = distance_line
                distance_line = 0
                y_max_row_step_1 = i
        if max_distance_line > word_space:
            if distance_line > 3.5 * max_distance_line:
                break

    max_distance_line_list.append(max_distance_line)

    distance_line = 0
    max_distance_line = 2
    y_max_row_step_2 = -1
    previous_row = -1
    flag = True
    for i in range(h):
        if np.count_nonzero(detect_image[i, int(2 * w / 3):] == 0) <= 10:
            distance_line += 1
        else:
            if flag:
                distance_line = 0
                flag = False
                if previous_row == -1:
                    previous_row = i
                continue
            else:
                if (distance_line > 0) & (distance_line <= 2):
                    distance_line = 0
                    continue
                else:
                    if previous_row == -1:
                        previous_row = i
                    else:

                        if (i - previous_row) > 100:
                            y_max_row_step_2 = i - distance_line
                            break
                        else:
                            previous_row = i

                if distance_line > max_distance_line:
                    max_distance_line = distance_line
                distance_line = 0
                # problem
                y_max_row_step_2 = i
        if max_distance_line > word_space:
            if distance_line > 3.5 * max_distance_line:
                break

    distance_line = 0
    max_distance_line = 2
    y_max_row_step_3 = -1
    previous_row = -1
    flag = True
    for i in range(h):
        # (detect_image[i, int(w / 3):int(2 * w / 3)] == 255).all()
        if np.count_nonzero(detect_image[i, int(2 * w / 5):int(3 * w / 5)] == 0) <= 10:
            distance_line += 1
        else:
            if flag:
                distance_line = 0
                flag = False
                if previous_row == -1:
                    previous_row = i
                continue
            else:
                if (distance_line > 0) & (distance_line <= 2):
                    distance_line = 0
                    continue
                else:
                    if previous_row == -1:
                        previous_row = i
                    else:
                        if (i - previous_row) > 75:
                            y_max_row_step_3 = i - distance_line
                            break
                        else:
                            previous_row = i

                if distance_line > max_distance_line:
                    # if distance_line > 100:
                    #     break
                    # if abs(max_distance_line - distance_line) > 100:
                    #     break
                    max_distance_line = distance_line
                distance_line = 0
                # problem
                y_max_row_step_3 = i
        if max_distance_line > word_space:
            if distance_line > 3.5 * max_distance_line:
                break

    y_row_step = np.asarray(sorted([y_max_row_step_1, y_max_row_step_2, y_max_row_step_3]))

    y_max_row = []
    for i in range(1, len(y_row_step)):
        distance_line = 0
        max_distance_line = 2
        y_max_row_step_whole = -1
        previous_row = -1
        flag = True
        for j in range(y_row_step[i - 1], y_row_step[i]):
            if np.count_nonzero(detect_image[j, :] == 0) <= 10:
                distance_line += 1
            else:
                if flag:
                    distance_line = 0
                    flag = False
                    if previous_row == -1:
                        previous_row = j
                    continue
                else:
                    if (distance_line > 0) & (distance_line <= 2):
                        distance_line = 0
                        continue
                    else:
                        if previous_row == -1:
                            previous_row = j
                        else:
                            if (j - previous_row) > 50:
                                y_max_row_step_whole = j - distance_line
                                break
                            else:
                                previous_row = j

                    if distance_line > max_distance_line:
                        max_distance_line = distance_line
                    distance_line = 0
                    y_max_row_step_whole = j
            if max_distance_line > word_space:
                if distance_line > 3.5 * max_distance_line:
                    break
        y_max_row.append(y_max_row_step_whole)
        max_distance_line_list.append(max_distance_line)
    max_distance_line_list = sorted(max_distance_line_list)
    # need to add the minimum value of y_row_step but how to choose?

    if -1 in y_max_row:
        if y_max_row[0] == -1:
            y_max_row[0] = max(y_row_step[0], y_row_step[1])

        if y_max_row[1] == -1:
            y_max_row[1] = max(y_row_step[1], y_row_step[2])

    y_max_row = sorted(y_max_row)
    if y_max_row[1] > y_max_row[0] + max(max_distance_line_list):
        if y_max_row_step_1 == max(y_row_step):
            y_max_row_final = y_max_row[1]
        else:
            y_max_row_final = y_max_row[0]
    else:
        y_max_row_final = y_max_row[1]

    if (y_max_row_step_1 == max(y_row_step)) & ((y_max_row_step_1 - y_max_row_final) >= (20 * word_space)):
        y_max_row_final = y_max_row_step_1

    if item_table_height - y_max_row_final > 2 * max(max_distance_line_list):
        y_max_row_final = item_table_height
    else:
        if (y_max_row_final - item_table_height < 1.5 * max(max_distance_line_list)) & (
                y_max_row_final - item_table_height > 0):
            y_max_row_final = item_table_height
    for i in range(len(max_distance_line_list)):
        if max_distance_line_list[i] > 10:
            if abs(item_table_height - y_max_row_step_1) < max_distance_line_list[i]:
                y_max_row_final = item_table_height
    return image_no_lines[1: y_max_row_final + 8, :], y_max_row_final + 8


def detect_column_by_line(table_image, header_y_max, threshold):
    items_image = table_image[header_y_max:, :]

    items_v_lines = get_v_lines(items_image, threshold, 0)

    x_head_lines = [[items_v_lines[i], items_v_lines[i + 1]] for i in range(0, len(items_v_lines), 2)]

    plt.imshow(items_image)
    plt.show()

    return x_head_lines


def assign_items_to_headers(column_split, header_info: list, items_info: list):
    """
    :param column_split:
    :param header_info: headers and their start_x and end_x positions
    :param items_info: list of lines which contain text and their start_x and end_x positions
    :return: cleaned header and line items
    """
    # Get length of headers
    header_count = len(header_info[0])

    # Get length of line items
    items_count = max(len(line[0]) for line in items_info)

    lines_output = []
    header_pos = header_info[1]

    for line in items_info:
        #     if len(line[0]) == header_count:
        #         line = [line for line in line[0]]
        #     else:

        # items_pos = [np.mean(pos) for pos in line[1]]
        # items_pos = [(pos[0] + 80) // 2 for pos in line[1]]  # normalise to 50 px wide
        item_header_idx = []

        # Calculate relative distance and get the min idx
        for pos in line[1]:

            # distances = [abs(pos - np.mean(header_x)) for header_x in header_pos]
            # distances = [abs(pos - (header_x[0] + 80) // 2) for header_x in
            #              header_pos]  # header box also normalise to 50 px wide
            # idx = [i for i, distance in enumerate(distances) if min(distances) == distance][0]
            #
            # item_header_idx.append(idx)
            index = -1
            flag_if = True
            for idx in range(len(column_split)):

                if ((pos[1] - 15 < column_split[idx][1]) & (pos[0] > column_split[idx][0])) | (
                        (pos[1] - 15 > column_split[idx][1]) & (pos[0] < column_split[idx][0])):
                    index = idx
                    flag_if = False
                    break
                if pos[1] - 15 > column_split[idx][0]:
                    index = idx
            if flag_if:
                index_list = [i for i in range(index + 2)] if index + 1 < len(column_split) else [i for i in range(
                    len(column_split))]
                distance = [abs(pos[0] - column_split[idx][0]) for idx in index_list]
                index = distance.index(min(distance))

            item_header_idx.append(index)

        line_dict = {}

        for header_index, line_text in zip(item_header_idx, line[0]):
            line_dict[header_index] = line_dict.get(header_index, "") + " " + line_text

        line = [line_dict[i].strip() if i in line_dict.keys() else "" for i in range(len(header_pos))]

        lines_output.append(line)

    return header_info[0], lines_output


def is_head_correct(line_columns, head_columns):
    if len(line_columns) != len(head_columns):
        return False
    return True


def compare_two_result(header_image, items_image, line_columns_input, head_columns_input):
    # if is_head_correct(line_columns, head_columns):
    image_new = np.concatenate((header_image, items_image), axis=0)
    image_new = remove_lines(remove_lines(image_new, axis=1), 0.3)
    h_lines = get_h_lines(image_new, 1, 2)

    images = split_sections(image_new, h_lines)
    line_columns = line_columns_input.copy()
    head_columns = head_columns_input.copy()
    index_of_line = len(line_columns)
    for i in range(len(line_columns) - 1, -1, -1):
        if line_columns[i][0] > head_columns[-1][1]:
            index_of_line = i
        else:
            break
    line_columns = line_columns[:index_of_line]
    line_columns[-1][1] = header_image.shape[1]
    result_column_split = []
    header_index = 0
    line_index = 0
    max_length = len(line_columns) if len(line_columns) > len(head_columns) else len(head_columns)
    while header_index < max_length:
        header_flag = False
        line_flag = False
        max_length = len(line_columns) if len(line_columns) > len(head_columns) else len(head_columns)
        if header_index + 1 == max_length:
            if line_index < len(line_columns):
                result_column_split.append([min(head_columns[header_index][0], line_columns[line_index][0]),
                                            max(head_columns[header_index][1], line_columns[line_index][1])])
                header_index += 1
                line_index += 1
            else:
                result_column_split.append([head_columns[header_index][0],
                                            head_columns[header_index][1]])
                header_index += 1
                line_index += 1
        elif line_index + 1 == max_length:
            if head_columns < len(head_columns):
                result_column_split.append([min(head_columns[header_index][0], line_columns[line_index][0]),
                                            max(head_columns[header_index][1], line_columns[line_index][1])])
                header_index += 1
                line_index += 1
            else:
                result_column_split.append([line_columns[line_index][0],
                                            line_columns[line_index][1]])
                header_index += 1
                line_index += 1
        else:
            if header_index + 1 < len(head_columns):
                if head_columns[header_index + 1][0] < line_columns[line_index][1]:

                    header_flag = True
                    header_image = images[0][:,
                                   min(head_columns[header_index][0], line_columns[line_index][0]):
                                   head_columns[header_index + 1][0]]
                    header_image = remove_lines(remove_lines(header_image, 0.8), axis=1)
                    find_line_image = np.concatenate(images[1:], axis=0)[:,
                                      min(head_columns[header_index][0], line_columns[line_index][0]):
                                      head_columns[header_index + 1][0]]
                    find_line_image = remove_lines(remove_lines(find_line_image, 0.7), axis=1)
                    image_new = np.concatenate((header_image, find_line_image), axis=0)
                    count_black = np.count_nonzero(image_new == 0, axis=0)
                    ver_index = np.where(count_black > 0)[0]
                    if len(count_black) not in ver_index:
                        ver_index = np.append(ver_index, len(count_black))
                    distance = 0

                    start = ver_index[1]
                    end = ver_index[0]
                    for i in range(1, len(ver_index)):
                        if ver_index[i] - ver_index[i - 1] >= distance:
                            if np.mean(count_black[end:ver_index[i - 1]]) > 1:
                                distance = ver_index[i] - ver_index[i - 1]
                                # times += 1
                                start = ver_index[i]
                                end = ver_index[i - 1]
                    diff = int(abs(end - start) / 2)
                    dis = min(head_columns[header_index][0], line_columns[line_index][0]) + end + diff
                    mini_y = min(line_columns[line_index][0], head_columns[header_index][0])
                    max_y = max(head_columns[header_index + 1][1] - diff + 5, head_columns[header_index + 1][1])

                    line_columns.append(
                        [dis, max(max_y, line_columns[line_index][1])])
                    line_columns.append([mini_y, dis])
                    line_columns.remove(line_columns[line_index])
                    result_column_split.append([mini_y, dis])
                    line_columns.sort(key=lambda row: row[0])
                    line_index += 1
                    header_index += 1
            if line_index + 1 < len(line_columns):
                if (line_columns[line_index + 1][0] < head_columns[header_index][1]) & (not header_flag):

                    line_flag = True
                    header_image = images[0][:,
                                   min(line_columns[line_index][0], head_columns[header_index][0]):
                                   line_columns[line_index + 1][0]]
                    header_image = remove_lines(remove_lines(header_image, 0.8), axis=1)
                    find_line_image = np.concatenate(images[1:], axis=0)[:,
                                      min(line_columns[line_index][0], head_columns[header_index][0]):
                                      line_columns[line_index + 1][0]]
                    find_line_image = remove_lines(remove_lines(find_line_image, 0.7), axis=1)
                    image_new = np.concatenate((header_image, find_line_image), axis=0)
                    count_black = np.count_nonzero(image_new == 0, axis=0)

                    ver_index = np.where(count_black > 0)[0]
                    if len(count_black) not in ver_index:
                        ver_index = np.append(ver_index, len(count_black))

                    distance = 0
                    start = ver_index[1]
                    end = ver_index[0]
                    for i in range(1, len(ver_index)):
                        if ver_index[i] - ver_index[i - 1] >= distance:
                            if np.mean(count_black[end:ver_index[i - 1]]) > 1:
                                distance = ver_index[i] - ver_index[i - 1]
                                # times += 1
                                start = ver_index[i]
                                end = ver_index[i - 1]
                    diff = int(abs(end - start) / 2)
                    # the distance need to be added
                    dis = min(line_columns[line_index][0], head_columns[header_index][0]) + end + diff
                    mini_y = min(head_columns[header_index][0], line_columns[line_index][0])
                    max_y = max(line_columns[line_index + 1][1] - diff + 5, line_columns[line_index + 1][1])

                    head_columns.append(
                        [dis, max(max_y, head_columns[header_index][1])])
                    head_columns.append([mini_y, dis])
                    head_columns.remove(head_columns[header_index])
                    result_column_split.append([mini_y, dis])
                    # result_column_split.append([dis, max_y])
                    head_columns.sort(key=lambda row: row[0])
                    header_index += 1
                    line_index += 1
            if (not line_flag) & (not header_flag):
                result_column_split.append([min(head_columns[header_index][0], line_columns[line_index][0]),
                                            max(head_columns[header_index][1], line_columns[line_index][1])])
                header_index += 1
                line_index += 1

    for i in range(len(result_column_split) - 1):
        if result_column_split[i][1] - result_column_split[i + 1][0] > 0:
            result_column_split[i + 1][0] = result_column_split[i][1] + 1
    result_column_split_final = []
    previous = -1
    for i in range(len(result_column_split)):
        if (result_column_split[i][1] - result_column_split[i][0]) < 20:
            previous = result_column_split[i][0]
            continue
        if previous != -1:
            result_column_split[i][0] = previous
            result_column_split_final.append(result_column_split[i])
            previous = -1
        else:
            result_column_split_final.append(result_column_split[i])
    return result_column_split_final


def revert_black_background(image_gray, table_labels, table_boxes):
    head_bounding_box = get_header_boxes(table_boxes, table_labels)
    head_bounding_box = np.asarray(head_bounding_box)
    x_min, y_min, x_max, y_max = int(min(head_bounding_box[:, 0])), int(min(head_bounding_box[:, 1])), \
        int(max(head_bounding_box[:, 2])), int(max(head_bounding_box[:, 3]))
    head_image = image_gray[y_min:y_max, x_min:x_max]

    # black = np.count_nonzero(head_image == 0)
    black = np.count_nonzero(head_image == 0)

    if black > 0.5 * (y_max - y_min) * (x_max - x_min):
        while y_min > 0:
            white_pixel = np.count_nonzero(image_gray[y_min, x_min:x_max] == 0)
            if white_pixel < 10:
                break
            else:
                y_min -= 1
        while y_max < image_gray.shape[0]:
            white_pixel = np.count_nonzero(image_gray[y_max, x_min:x_max] == 0)
            if white_pixel < 10:
                break
            else:
                y_max += 1
        while x_min > 0:
            white_pixel = np.count_nonzero(image_gray[y_min:y_max, x_min] == 0)
            if white_pixel < 5:
                break
            else:
                x_min -= 1
        while x_max < image_gray.shape[1]:
            white_pixel = np.count_nonzero(image_gray[y_min:y_max, x_max] == 0)
            if white_pixel < 5:
                break
            else:
                x_max += 1
    else:
        return image_gray
    head_image = cv2.bitwise_not(image_gray[y_min:y_max, x_min:x_max])
    head_image_left = image_gray[y_min:y_max, :x_min]
    head_image_right = image_gray[y_min:y_max, x_max:]
    head_image = np.concatenate((head_image_left, head_image, head_image_right), axis=1)
    head_image_top = image_gray[:y_min, :]
    head_image_bot = image_gray[y_max:, :]
    image_final = np.concatenate((head_image_top, head_image, head_image_bot), axis=0)
    return image_final


def validate_items_table(image):
    h_lines = get_h_lines(image, 1, 1)
    output_image = image.copy()
    for i in range(0, len(h_lines), 2):
        if h_lines[i + 1] - h_lines[i] < 15:
            output_image[h_lines[i]:h_lines[i + 1], :] = 255
    return output_image


def validate_items(items_image):
    # remove empty spaces
    lines = [line for line in items_image if np.sum(line == 0) / (line.shape[0] * line.shape[1]) > 0.001]
    return lines


def validate_items_key_value(items_image):
    # remove empty spaces
    lines = [line for line in items_image if np.sum(line == 0) / (line.shape[0] * line.shape[1]) > 0.005]
    return lines
