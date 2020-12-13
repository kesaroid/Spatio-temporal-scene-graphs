import os
from collections import defaultdict
import numpy as np
import copy
import pickle
import scipy.sparse
from PIL import Image
import h5py, json
import torch
import random
from tqdm import tqdm
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


class AGDataset(Dataset):
    def __init__(self, split, img_dir, ann_file, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False, custom_path=''):
        
        
        assert split in {'train', 'val', 'test'}

        self.flip_aug = flip_aug
        self.split = split
        self.transforms = transforms
        self.img_dir = img_dir
        self.ann_file = ann_file 
        self.transforms = transforms
        self.coco = COCO(self.ann_file)
        self.image_index = list(sorted(self.coco.imgs.keys()))
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.custom_eval = custom_eval

        assert os.path.exists(self.ann_file), "Cannot find the Action Genome dataset at {}".format(self.ann_file)
        self.ind_to_classes = []; self.class_to_ind = dict(); self.ind_to_predicates = []; self.predicate_to_ind = dict()

        for cat in self.coco.dataset['categories']:
            if cat['supercategory'] == 'object':
                self.class_to_ind[cat['name']] = cat['id']
                self.ind_to_classes.append(cat['name'])
            else: 
                self.predicate_to_ind[cat['name']] = cat['id']
                self.ind_to_predicates.append(cat['name'])

        self.ind_to_classes.insert(0, '__background__'); self.class_to_ind['__background__'] = 0        # 37 including __background__
        self.ind_to_predicates.insert(0, '__background__'); self.predicate_to_ind['__background__'] = 0 # 27 including __background__
        self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        if self.custom_eval:
            self.get_custom_imgs(custom_path)
        else:
            self.filenames, self.im_sizes, self.gt_boxes, self.gt_classes, self.relationships = self.load_graphs(self.coco.dataset, mode=self.split)

        print(self.__getitem__(1))
        print(self.filenames[1], self.gt_boxes[1], self.gt_classes[1], self.relationships[1])
        exit()

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.img_dir, self.filenames[index]))        
        
        if self.custom_eval:
            target = torch.LongTensor([-1])
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, index
        
        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')
        
        target = self.get_groundtruth(index, flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target, index


    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        # get object bounding boxes, labels and relations
        obj_boxes = np.array(self.gt_boxes[index].copy())
        obj_labels = np.array(self.gt_classes[index].copy())
        obj_relation_triplets = self.relationships[index].copy()
        
        obj_relations = np.zeros((obj_boxes.shape[0], obj_boxes.shape[0]))
        obj_attributes = obj_relations.copy()

        #TODO No obj_id should be t >= 0 && t < n_classes
        for i in range(obj_relation_triplets.shape[0]):
            subj_id = obj_relation_triplets[i][0] # is always zero
            obj_id = obj_relation_triplets[i][1]
            pred = obj_relation_triplets[i][2]
            obj_relations[subj_id, obj_id] = pred

        img_info = self.get_img_info(index)
        width, height = img_info['width'], img_info['height']
        target = BoxList(obj_boxes, (width, height), mode="xyxy")
        
        target.add_field("labels", torch.from_numpy(obj_labels))
        # target.add_field("pred_labels", torch.from_numpy(obj_relations))
        target.add_field("relation", torch.from_numpy(obj_relations), is_triplet=True)
        target.add_field("attributes", torch.from_numpy(obj_attributes)) # Useless
        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(obj_relation_triplets)) # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=False)
            return target

    def __len__(self):
        return len(self.image_index)

    def get_img_info(self, img_id):
        w, h = self.im_sizes[img_id, :]
        return {"height": h, "width": w}

    def map_class_id_to_class_name(self, class_id):
        return self.ind_to_classes[class_id]

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width':int(img.width), 'height':int(img.height)})
    
    def get_statistics(self):
        fg_matrix, bg_matrix = get_AG_statistics(img_dir=self.img_dir, ann_file=self.ann_file, must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
        }
        return result

    def load_graphs(self, annotation, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                    filter_non_overlap=False, cache=True):
        
        # Check mode
        if mode not in ('train', 'val', 'trainval'):
            raise ValueError('{} invalid'.format(mode))
        cache_file = os.path.join('datasets', 'ag_{}_cache.pkl'.format(mode))
        
        if cache:
            # Load cache if exists
            if os.path.isfile(cache_file):
                with open(cache_file, 'rb') as handle:
                    (filenames, sizes, boxes, gt_classes, relationships) = pickle.load(handle)
                print('Read imdb from cache at location: {}'.format(cache_file))
                return filenames, sizes, boxes, gt_classes, relationships
        
        filenames = []; sizes = []; boxes = []; gt_classes = []; relationships = []

        for i, imgs in tqdm(enumerate(self.image_index)):
            # Load ag['image'] & ag['annotations']
            _img = self.coco.dataset['images'][i]
            _anno = list(filter(lambda item: item['image_id'] == imgs, self.coco.dataset['annotations']))

            filenames.append(_img['filename'])
            sizes.append(np.array([_img['height'], _img['width']]))
            
            box_i = []; rels = []; gt_ci = [1]
            for item in _anno:
                # Append all annotations of an image into one array
                assert len(item['bbox']) == 4
                box_i.append(item['bbox'])
                # Append all relationship triplets [0, <catergory_index from gt_ci>, <relationship>]
                if not item['isperson'] and len(_anno) > 1:
                    gt_ci.append(item['category_id'])
                    rels.append([gt_ci.index(1), gt_ci.index(item['category_id']), item['contacting_id'].pop()])
                    #TODO Same for other relations
                # elif len(_anno) == 1: # When it is only 1 person
                #     gt_ci.append(item['category_id'])
                #     rels.append([gt_ci.index(1), 0, 0]) # Cannot be just [0, 0, 0] background
                #     # TODO assert self.num_att_cls==len(att_classes)
                assert np.asarray(rels).shape[0] <= np.asarray(gt_ci).shape[0]

            box_i = np.array(box_i)
            assert box_i.ndim == 2, 'bbox missing for image_index: {}'.format(_anno)

            boxes.append(box_i)
            gt_classes.append(np.array(gt_ci))
            relationships.append(np.array(rels))

        sizes = np.stack(sizes, 0)
        
        # Create cache to save time
        if cache:
            with open(cache_file, 'wb') as handle:
                pickle.dump((filenames, sizes, boxes, gt_classes, relationships), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Wrote the imdb to cache at location: {}'.format(cache_file))
        
        return filenames, sizes, boxes, gt_classes, relationships


def get_AG_statistics(img_dir, ann_file, must_overlap=True):
    train_data = AGDataset(split='train', img_dir=img_dir, ann_file=ann_file, 
                        num_val_im=5000, filter_duplicate_rels=False)
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        if len(gt_relations) == 0:
            continue
        
        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        
        for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
            fg_matrix[o1, o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1
    return fg_matrix, bg_matrix

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter
    
# [{'id': 372891829902578154, 'label': 'table', 'bbox': [222.10317460317458, 143.829365079365, 479.88095238095235, 244.9404761904761], 
# 'area': 26064.19753086419, 'iscrowd': 0, 'category_id': 31, 'image_id': 0, 'attention_relationship': ['unsure'], 'attention_id': [38], 
# 'spatial_relationship': ['in_front_of'], 'spatial_id': [41], 'contacting_relationship': ['not_contacting'], 'contacting_id': [53]}]
# {'id': 372891834197545450, 'label': 'chair', 'bbox': [56.34126984126985, 179.16666666666663, 249.11904761904762, 269.7355687782746],
#  'area': 17459.671684848872, 'iscrowd': 0, 'category_id': 7, 'image_id': 0, 'attention_relationship': ['not_looking_at'], 'attention_id': [37],
#  'spatial_relationship': ['beneath', 'behind'], 'spatial_id': [40, 42], 'contacting_relationship': ['sitting_on', 'leaning_on'], 'contacting_id': [51, 55]}
# {'id': 372933345056461290, 'label': 'person', 'bbox': [24.297740936279297, 71.44395446777344, 259.23602294921875, 268.202880859375], 
# 'category_id': 0, 'area': 46226.20413715328, 'image_id': 0, 'iscrowd': 0}


    # if 'attention_id' in item:
    #     gt_ci.extend(item['attention_id'])
    # if 'spatial_id' in item:
    #     gt_ci.extend(item['spatial_id'])
    # if 'contacting_id' in item:
    #     gt_ci.extend(item['contacting_id'])
    
    # Append all relationship triplets [0. <relationship>, <category_id>]
    # if item['label'] != 'person':
    #     rels = []
    #     for j in range(len(item['attention_relationship'])):
    #         rels.append([0, item['category_id'], item['attention_id'][j]])
    #     for j in range(len(item['contacting_relationship'])):
    #         rels.append([0, item['category_id'], item['contacting_id'][j]])
    #     for j in range(len(item['spatial_relationship'])):
    #         rels.append([0, item['category_id'], item['spatial_id'][j]])

    #     relationships.append(np.array(rels))