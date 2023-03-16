from src.postprocessing import get_h_lines, get_v_lines, form_and_get_page_sections, get_table_items, \
    get_line_items_section, get_split_line_items_header
from src.postprocessing import process_line_items, process_dataframes, assign_items_to_headers, get_line_margin, \
    check_for_invalid_header_section, get_space_margin, detect_table_line_by_line_items, \
    split_lines, get_header_boxes,validate_items_table, validate_items
from src.postprocessing import compare_two_result, get_header_y3
from src.bbox_manipulation import unnormalize_boxes
import numpy as np


def get_data(data, image):
    labels_list = []
    bboxes_list = []
    for i in range(len(data["bbox"])):
        if data["predictions"][i].lower() != "o":
            _label = data["predictions"][i].lower()
            if (_label == "header") | ("line_item" in _label):
                labels_list.append(_label)
                bboxes_list.append(data["bbox"][i])
    unnormalised_table_boxes = [unnormalize_boxes(box, image.shape[1], image.shape[0]) for box in
                                bboxes_list]
    return labels_list, unnormalised_table_boxes


def get_result(bboxes_list, labels_list, binary_image_copy):
    header_box = get_header_boxes(bboxes_list, labels_list)
    header_y3 = get_header_y3(binary_image_copy, bboxes_list, labels_list)
    space_margin = get_space_margin(bboxes_list, labels_list)
    line_margin, section_space = get_line_margin(bboxes_list, labels_list)
    variance_headers = int(np.std(np.asarray(header_box)[:, 3]))
    median_header_height = int(np.median(np.asarray(header_box)[:, 3] - np.asarray(header_box)[:, 1]))
    header_box_heights = np.asarray(header_box)[:, 3] - np.asarray(header_box)[:, 1]
    h_lines = get_h_lines(binary_image_copy, section_space)
    page_sections = form_and_get_page_sections(binary_image_copy, h_lines)
    items_table = get_line_items_section(binary_image_copy, bboxes_list, labels_list, page_sections,
                                         int(line_margin * 1.2))
    _, table_start_y, _, table_end_y = items_table
    table_height = table_end_y - table_start_y
    header_section = binary_image_copy[table_start_y: header_y3, :]
    items_section = binary_image_copy[header_y3 + 1:, :]

    header_section, header_max = check_for_invalid_header_section(header_section)
    h, w = header_section.shape
    item_table_height = table_height - header_max

    items_table, item_table_end_y = detect_table_line_by_line_items(items_section, section_space, space_margin,
                                                                    item_table_height)

    items_table = validate_items_table(items_table)
    line_item_images = split_lines(items_table)
    line_item_images = validate_items(line_item_images)
    length = []
    for i, line in enumerate(line_item_images):
        count_black = np.count_nonzero(line == 0, axis=0)
        index_pixel = np.where(count_black > 0)[0]

        distance = index_pixel[-1] - index_pixel[0]
        length.append(distance)
    max_length = max(length)
    index = length.index(max(length))
    for i in range(len(length)):
        if length[i] + 50 > max_length:
            index = i
            break
    max_line_item = line_item_images[index]
    column_by_header = get_v_lines(header_section, int(space_margin * 1.2))
    column_by_header = [[column_by_header[i], column_by_header[i + 1]] for i in range(0, len(column_by_header), 2)]
    column_by_lines = get_v_lines(max_line_item, section_space)
    column_by_lines = [[column_by_lines[i], column_by_lines[i + 1]] for i in range(0, len(column_by_lines), 2)]
    final_header_split = compare_two_result(header_section, items_table, column_by_lines, column_by_header)
    line_item_images = split_lines(items_table)
    line_item_images = validate_items(line_item_images)
    for i in line_item_images:
        print(i.shape)
    v_line_header = list(np.asarray(final_header_split).reshape(1, -1)[0])
    headers_with_x_pos = get_split_line_items_header(header_section, v_line_header, plot=False)
    headers, header_x_range = headers_with_x_pos
    line_items_and_x_pos = process_line_items(line_item_images, space_margin, padding=15)  # default 15
    line_items = [item[0] for item in line_items_and_x_pos]
    assign_column_list = []
    for i in range(len(final_header_split)):
        if i == 0:
            assign_column_list.append([0, final_header_split[i + 1][0] - 1])
        elif i == len(final_header_split) - 1:
            assign_column_list.append([final_header_split[i - 1][1], items_table.shape[1]])
        else:
            assign_column_list.append([final_header_split[i][0], final_header_split[i + 1][0] - 1])
    headers, line_items = assign_items_to_headers(assign_column_list, headers_with_x_pos, line_items_and_x_pos)
    df = process_dataframes(headers, line_items)
    dict_table = df.to_dict()
    table_bounding_box = [0, table_start_y, binary_image_copy.shape[1], item_table_end_y + header_y3 + 1]
    return dict_table, table_bounding_box
