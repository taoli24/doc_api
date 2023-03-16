import cv2
from src.postprocessing import ocr_scan_image, split_lines
from src.postprocessing import validate_items_key_value
from src.bbox_manipulation import unnormalize_boxes
import numpy as np
import re


def get_id_to_label(unique_label_path):
    with open(unique_label_path, "r") as f:
        data_unique = f.readlines()
    for i in range(len(data_unique)):
        data_unique[i] = data_unique[i][:-1]
    data_unique.append("O")
    id2label = {i: v for i, v in enumerate(data_unique)}
    return id2label


def transfer_label(id2label, data_label):
    return [id2label[i] for i in data_label]


def get_data(data):
    labels_list = []
    bboxes_list = []
    for i in range(len(data["bbox"])):
        if data["predictions"][i].lower() != "o":
            _label = data["predictions"][i].lower()
            if (_label != "header") & ("line_item" not in _label) & ((data["bbox"][i][3] - data["bbox"][i][1]) *
                                                                     (data["bbox"][i][2] - data["bbox"][i][0]) > 2):
                labels_list.append(_label)
                bboxes_list.append(data["bbox"][i])
    dict_table = {}.fromkeys(list(set(labels_list)))
    for i in range(len(labels_list)):
        if dict_table[labels_list[i]] is None:
            dict_table[labels_list[i]] = [bboxes_list[i]]
        else:
            if bboxes_list[i] not in dict_table[labels_list[i]]:
                dict_table[labels_list[i]].append(bboxes_list[i])

    return dict_table


def compare_vertical_v1(dict_table: dict):
    for key in dict_table.keys():
        dict_table[key].sort(key=lambda row: (row[3] + row[1]) / 2)
        dict_table[key] = np.array(dict_table[key])
    dict_distance_label = {}.fromkeys(set(dict_table.keys()))
    for key, value in dict_table.items():
        for i in value:
            if dict_distance_label[key] is None:
                dict_distance_label[key] = [(i[3] + i[1]) / 2]
            else:
                dict_distance_label[key].append((i[3] + i[1]) / 2)
    dict_merge_label = {}.fromkeys(dict_distance_label.keys())
    for key in list(dict_distance_label.keys()):
        li = []
        part_li = []
        if len(dict_distance_label[key]) == 1:
            li.append([0])
        for j in range(1, len(dict_distance_label[key])):
            if abs(dict_distance_label[key][j] - dict_distance_label[key][j - 1]) < 10:
                part_li.append(j - 1)
                if j == (len(dict_distance_label[key]) - 1):
                    part_li.append(j)
                    li.append(part_li)
            else:
                part_li.append(j - 1)
                li.append(part_li)
                part_li = []
        dict_merge_label[key] = li
    final_key_value_table = {}.fromkeys(dict_distance_label.keys())
    for key in dict_merge_label.keys():
        final_list = []
        for i in range(len(dict_merge_label[key])):
            second_list = []
            if len(dict_merge_label[key][i]) == 1:
                second_list.append(list(dict_table[key][dict_merge_label[key][i][0]]))
            else:
                for j in range(len(dict_merge_label[key][i])):
                    second_list.append([dict_table[key][dict_merge_label[key][i][j]][0],
                                        min(dict_table[key][dict_merge_label[key][i][0]:dict_merge_label[key][i][-1],
                                            1]) - 2,
                                        dict_table[key][dict_merge_label[key][i][j]][2],
                                        max(dict_table[key][dict_merge_label[key][i][0]:dict_merge_label[key][i][-1],
                                            3]) + 2])
            final_list.append(second_list)
        final_key_value_table[key] = final_list
    return final_key_value_table


def compare_horizontal(final_key_value_table: dict):
    for key in final_key_value_table.keys():
        for i in range(len(final_key_value_table[key])):
            final_key_value_table[key][i].sort(key=lambda row: (row[2] + row[0]) / 2)
            final_key_value_table[key][i] = np.array(final_key_value_table[key][i])
    new_final = {}.fromkeys(final_key_value_table.keys())
    for key in final_key_value_table.keys():
        total_array = []
        for i in range(len(final_key_value_table[key])):
            part_li = []
            if len(final_key_value_table[key][i]) == 1:
                part_li.append(list(final_key_value_table[key][i][0]))
            else:
                start_index = 0
                for j in range(1, len(final_key_value_table[key][i])):
                    if abs(final_key_value_table[key][i][j][0] - final_key_value_table[key][i][j - 1][2]) < 15:
                        continue
                    else:
                        part_li.append([min(final_key_value_table[key][i][start_index:j, 0]),
                                        final_key_value_table[key][i][start_index, 1],
                                        max(final_key_value_table[key][i][start_index:j, 2]),
                                        final_key_value_table[key][i][start_index, 3]])
                        start_index = j

                part_li.append(
                    [min(final_key_value_table[key][i][start_index:, 0]), final_key_value_table[key][i][start_index, 1],
                     max(final_key_value_table[key][i][start_index:, 2]),
                     final_key_value_table[key][i][start_index, 3]])
            total_array.append(part_li)
        new_final[key] = total_array
    return new_final


def compare_vertical_v2(new_final: dict):
    final_result = {}.fromkeys(new_final.keys())
    for key in list(new_final.keys()):
        if len(new_final[key]) == 1:
            if len(new_final[key][0]) == 1:
                final_result[key] = new_final[key][0][0]
            else:
                li = []
                for box in new_final[key][0]:
                    li.append(box[2] - box[0])
                index_max_distance = li.index(max(li))
                final_result[key] = new_final[key][0][index_max_distance]
        else:
            new_final[key].sort(key=lambda row: row[0][1])
            part_li = new_final[key][0]
            for i in range(1, len(new_final[key])):
                for j in range(len(part_li)):
                    for z in range(len(new_final[key][i])):
                        if (abs(part_li[j][3] - new_final[key][i][z][1]) < 25) & \
                                (abs(part_li[j][0] - new_final[key][i][z][0]) < 25):
                            part_li[j][0] = min(part_li[j][0], new_final[key][i][z][0])
                            part_li[j][1] = min(part_li[j][1], new_final[key][i][z][1])
                            part_li[j][2] = max(part_li[j][2], new_final[key][i][z][2])
                            part_li[j][3] = max(part_li[j][3], new_final[key][i][z][3])
                        else:
                            continue

            li = []
            for box in part_li:
                li.append(box[2] - box[0])
            index_max_distance = li.index(max(li))
            final_result[key] = part_li[index_max_distance]
    return final_result


def combine(data: dict):
    dict_table = get_data(data)
    table_vertical = compare_vertical_v1(dict_table)
    table_horizontal = compare_horizontal(table_vertical)
    final_table = compare_vertical_v2(table_horizontal)
    return final_table


def get_key_value(dict_table, image_binary):
    key_dict = {}.fromkeys(list(dict_table.keys()))
    for key in list(dict_table.keys()):
        pixel_value = [int(pixel) for pixel in
                       unnormalize_boxes(dict_table[key], image_binary.shape[1], image_binary.shape[0])]
        dict_table[key] = pixel_value
        line_item_images = split_lines(image_binary[pixel_value[1]:pixel_value[3]+2, pixel_value[0]:pixel_value[2]+2])
        line_item_images = validate_items_key_value(line_item_images)
        li = []
        for image in line_item_images:
            image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
            word = ocr_scan_image(image, second_scan=False)
            if len(word) == 0:
                word = ocr_scan_image(image, second_scan=True)
                print(key, word, "bounding_box ",dict_table[key])
                if not bool(re.search(r'\d', word)):
                    word = ""
            else:
                if (key == "CUSTOMER_TAX_ID") | (key == "INVOICE_NUMBER") | (key == "SUPPLIER_TAX_ID") | (
                        key == "SUPPLIER_BSB") | (key == "SUPPLIER_ACCOUNT_NUMBER"):
                    if not (word[:-1]).isdigit():
                        word = ocr_scan_image(image, second_scan=True)
                        if not bool(re.search(r'\d', word)):
                            word = ""
            # if "\n" in word:
            #     word = word[:-1]
            word = re.sub(r'\n', ' ', word)
            li.append(word)
        words = " ".join(li)
        key_dict[key] = words
    return key_dict
