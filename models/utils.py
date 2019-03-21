

def list_image_sets():
    """
    Summary: 
        List all the image sets from Pascal VOC. Don't bother computing
        this on the fly, just remember it. It's faster.
    """
    return [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']


def class_to_index():
    class_to_index = {}
    for i, cat_name in enumerate(list_image_sets()):
        class_to_index[cat_name] = i
    return class_to_index


def index_to_class():
    index_to_class = {}
    for i, cat_name in enumerate(list_image_sets()):
        index_to_class[i] = cat_name
    return index_to_class