from models.vocparseclslabels import PascalVOC
import torch
import pandas as pd
import os.path
import os

class ResultViewer(object):
    def __init__(self, root, results_path):
        self.cls = None
        self.pv = PascalVOC(os.path.join(root, 'VOCdevkit', 'VOC2012'))
        self.res = torch.load(results_path)

    def get_class_results(self, cls):
        img_list = self.pv.imgs_from_category_as_list(cls, 'val')
        print(len(img_list))
        cls_idx = self.class_to_index()[cls]
        img_list = pd.DataFrame(img_list)
        img_list = img_list.set_index(0)

        res = pd.DataFrame(self.res)
        res = res.set_index(0)

        class_res = img_list.join(res,on=0, how='left')
        class_res = class_res.sort_values(cls_idx+1, ascending=False)

        return class_res
        # print(img_list)
        
    def get_classes(self):
        return self.pv.list_image_sets()

    def get_imgs_from_class(self, cat):
        return self.pv.imgs_from_category_as_list(cat, 'val')

    def class_to_index(self):
        class_to_index = {}
        for i, cat_name in enumerate(self.pv.list_image_sets()):
            class_to_index[cat_name] = i
        return class_to_index

    def get_img_path(self, img_name):
        return os.path.join(self.pv.img_dir, img_name + '.jpg')