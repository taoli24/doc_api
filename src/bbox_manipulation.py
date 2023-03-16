def enlarge_boxes(boxes: list, lh, lw):
    """
    This function simply make bounding boxes biggher based on width and height info
    :param boxes: bounding boxes
    :param lh: enlarge on y-axis
    :param lw: enlarge on x-axis
    :return: enlarged bounding boxes
    """
    return [[box[0] - lw, box[1] - lh, box[2] + lw, box[3] + lh] for box in boxes]


def _merge_boxes(boxes: list):
    # Internal function, should not be called directly
    # create copy of the list, so it won't mass with the list from outer scope
    # parr of the process_box function
    boxes = boxes.copy()

    output_boxes = []

    def _merging(bbox: list, output):
        if len(bbox) == 0:
            return
        bbox = sorted(bbox, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
        min_x, min_y, max_x, max_y = bbox[0]

        input_boxes = []

        for i in range(1, len(bbox)):
            # upper left corner is within range
            if min_x <= bbox[i][0] <= max_x and min_y <= bbox[i][1] <= max_y:
                max_x = bbox[i][2] if max_x < bbox[i][2] else max_x
                max_y = bbox[i][3] if max_y < bbox[i][3] else max_y
                continue

            # upper right corner is within range
            if min_x <= bbox[i][2] <= max_x and min_y <= bbox[i][1] <= max_y:
                min_x = bbox[i][0] if min_x > bbox[i][0] else min_x
                max_y = bbox[i][3] if max_y < bbox[i][3] else max_y
                continue

            # lower left corner is within range
            if min_x <= bbox[i][0] <= max_x and min_y <= bbox[i][3] <= max_y:
                max_x = bbox[i][2] if max_x < bbox[i][2] else max_x
                min_y = bbox[i][1] if min_y > bbox[i][1] else min_y
                continue

            # lower right corner is within range
            if min_x <= bbox[i][2] <= max_x and min_y <= bbox[i][3] <= max_y:
                min_x = bbox[i][0] if min_x > bbox[i][0] else min_x
                min_y = bbox[i][1] if min_y > bbox[i][1] else min_y
                continue

            input_boxes.append(bbox[i])

        output.append([min_x, min_y, max_x, max_y])

        _merging(input_boxes, output)

    _merging(boxes, output_boxes)

    return output_boxes


def process_boxes(bbox):
    """
    merge overlapping bounding boxes
    :param bbox: iterable bounding boxes informarion
    :return: merged bounding box, only return when there is no more overlapping bounding boxes
    """
    output = _merge_boxes(bbox)
    next_output = _merge_boxes(output)

    while len(output) != len(next_output):
        return process_boxes(output)

    return output


def resize_normal_box(boxes, lw, lh):
    """
    return box to original size
    :param boxes: bounding boxes
    :param lw: line width
    :param lh: line height
    :return: list of normal bounding boxes
    """
    return [[box[0] + lw, box[1] + lh, box[2] - lw, box[3] - lh] for box in boxes]


def process_tales(bbox: list, lw: int, lh: int):
    """
    process bounding boxes in single page
    :param bbox: single bounding box
    :param lw: line width
    :param lh: line height
    :return: processed bounding box
    """
    larger_boxes = enlarge_boxes(bbox, lh, lw)

    return resize_normal_box(process_boxes(larger_boxes), lw, lh)


def unnormalize_boxes(normalized_boxes, width, height):
    w_scale = width / 1000
    h_scale = height / 1000
    return [normalized_boxes[0] * w_scale,
            normalized_boxes[1] * h_scale,
            normalized_boxes[2] * w_scale,
            normalized_boxes[3] * h_scale]
