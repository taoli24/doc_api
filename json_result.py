import cv2
import numpy as np
from src.postprocessing import revert_black_background, remove_line_using_cv_2
from src.postprocessing import rotated_image
import key_value as kv
import table_postprocess as tp
from PIL import Image


def get_json(data, image: Image):
    id2label = kv.get_id_to_label("./label.txt")
    data["predictions"] = kv.transfer_label(id2label, data["predictions"])

    final_table = kv.combine(data)

    # convert pillow image to ndarray
    image_array = np.asarray(image)
    grey_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    (thresh, binary_image) = cv2.threshold(grey_image, 150, 255, cv2.THRESH_BINARY)
    if (binary_image.shape[0] > 2500) and (binary_image.shape[1] > 1800):
        binary_image = cv2.resize(binary_image, (1654, 2339), cv2.INTER_AREA)
    elif (binary_image.shape[0] < 2000) and (binary_image.shape[1] < 1300):
        binary_image = cv2.resize(binary_image, (1654, 2339), cv2.INTER_AREA)
    labels_list, bboxes_list = tp.get_data(data, binary_image)
    binary_image = revert_black_background(binary_image, labels_list, bboxes_list)
    (thresh, binary_image) = cv2.threshold(binary_image, 150, 255, cv2.THRESH_BINARY)
    binary_image_copy = remove_line_using_cv_2(binary_image, 80, 80)
    _, binary_image_copy = cv2.threshold(binary_image_copy, 200, 255, cv2.THRESH_BINARY)
    binary_image_copy = rotated_image(binary_image_copy)
    _, binary_image_copy = cv2.threshold(binary_image_copy, 200, 255, cv2.THRESH_BINARY)

    key_dict = kv.get_key_value(final_table, binary_image_copy)
    dict_table, table_bounding_box = tp.get_result(bboxes_list, labels_list, binary_image_copy)

    key_result = {"tokens": []}
    for key in key_dict.keys():
        if final_table[key][3] <= table_bounding_box[1] or final_table[key][1] >= table_bounding_box[3]:
            bbox = [final_table[key][0] / binary_image_copy.shape[1], final_table[key][1] / binary_image_copy.shape[0],
                    final_table[key][2] / binary_image_copy.shape[1], final_table[key][3] / binary_image_copy.shape[0]]
            key_result["tokens"].append({"text": key_dict[key], "label": key, "bbox": bbox})

    data = []
    for i in range(len(list(dict_table.values()))):
        val = list(list(dict_table.values())[i].values())
        for j in range(len(val)):
            if i == 0:
                data.append([val[j]])
            else:
                data[j].append(val[j])
    bbox = [table_bounding_box[0] / binary_image_copy.shape[1], table_bounding_box[1] / binary_image_copy.shape[0],
            table_bounding_box[2] / binary_image_copy.shape[1], table_bounding_box[3] / binary_image_copy.shape[0]]
    table_result = {"columns": list(dict_table.keys()), "index": list(list(dict_table.values())[0].keys()),
                    "data": data,
                    "bbox": bbox}

    json_result = key_result.copy()

    json_result["expense_table"] = table_result

    return json_result

