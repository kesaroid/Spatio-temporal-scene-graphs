import imp
import json
import os
import json
import torch
import utils.paths_catalog as paths
from easydict import EasyDict as edict

class AGDataset():
    def __init__(self, cache=False) -> None:
        
        self.data_file = json.load(open(paths.ag_data_root + paths.test_data))
        self.data_stats = json.load(open(paths.ag_data_root + paths.data_stats))

        # load detected results
        if not cache:
            print('Loading detection results..')
            self.detected_origin_result = torch.load(paths.detected_origin_path + 'eval_results.pytorch')
            self.detected_info = json.load(open(paths.detected_origin_path + 'visual_info.json'))

        self.instances = self.get_instances()
        # Create dict with video 
        self.videos = edict.fromkeys(self.instances, {'objs':[], 'rels':[], 'frames':[], 'groundtruth':[], 'prediction':[]})
        self.idx2label = self.data_stats['idx_to_label']
        self.idx2pred = self.data_stats['idx_to_predicate']
        self.cat2pred = {
            "attention": ["looking_at", "not_looking_at", "unsure"],
            "spatial": ["above", "beneath", "in_front_of", "behind", "on_the_side_of", "in"],
            "contact": ["carrying","covered_by","drinking_from","eating","have_it_on_the_back",
                        "holding","leaning_on","lying_on","not_contacting","other_relationship",
                        "sitting_on","standing_on","touching","twisting","wearing","wiping","writing_on"]}

        # video: {'objs': {}, 'rels': {}, 'frames': [], 'groundtruth': [[], [], [], []], 'prediction':[[], [], [], []]}

    def create_db(self):
        prev_vid = 'MLWB5.mp4'; all_labels = []; all_rels = []
        for idx, results in enumerate(self.detected_info):
            img_path = results['img_file']
            frame = img_path.split('/')[-1]
            video = img_path.split('/')[-2]
            self.videos[video]['frames'].append(frame)
            
            if video in self.videos.keys():
                img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label = self.get_info_by_idx(idx, thres=0.5)
                self.videos[video]['objs'].extend(labels)
                self.videos[video]['rels'].extend(pred_rel_label)
                self.videos[video]['groundtruth'].append(gt_rels)
                self.videos[video]['prediction'].append(pred_rels)

                # self.videos[video][frame].append(frame_info)
                # pred_rels = self.clean_preds(gt_rels, pred_rels, labels, pred_rel_label)
                if prev_vid != video:
                    # TODO if cleaning required
                    self.videos[prev_vid]['objs'] = set(self.videos[prev_vid]['objs'])
                    self.videos[prev_vid]['rels'] = set(self.videos[prev_vid]['rels'])
                    prev_vid = video


    # def get_ids(self, video):
    #     idx = 0
    #     all_ids = []; gts=[]; preds=[]
    #     while idx < len(self.detected_info):
    #         img_path = self.detected_info[idx]['img_file']
    #         if img_path.split('/')[-2] == video:
    #             img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label = self.get_info_by_idx(idx, thres=0.5)
    #             gts.append(gt_rels)
    #             preds.append(pred_rels)
    #             all_ids.append(idx)
    #         idx += 1
    #     return all_ids, gts, preds

    def get_instances(self):
        instances = []
        for images in self.data_file['images']:
            instances.append(images['video'])
        
        return list(set(instances))
    
    # get image info by index
    def get_info_by_idx(self, idx, thres=0.5):
        groundtruth = self.detected_origin_result['groundtruths'][idx]
        prediction = self.detected_origin_result['predictions'][idx]
        # image path
        img_path = self.detected_info[idx]['img_file']
        boxes = groundtruth.bbox
        
        labels = ['{}'.format(self.idx2label[str(i)]) for i in groundtruth.get_field('labels').tolist()]
        pred_labels = ['{}'.format(self.idx2label[str(i)]) for i in prediction.get_field('pred_labels').tolist()]
        # groundtruth relation triplet
        gt_rels = groundtruth.get_field('relation_tuple').tolist()
        gt_rels = [(labels[i[0]], self.idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
        # prediction relation triplet
        pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
        pred_rel_label = prediction.get_field('pred_rel_scores')
        pred_rel_label[:,0] = 0
        pred_rel_score, pred_rel_label = pred_rel_label.max(-1)

        pred_rels = [(pred_labels[i[0]], self.idx2pred[str(j)], pred_labels[i[1]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]
        pred_rel_label = [self.idx2pred[str(i)] for i in pred_rel_label.tolist()]

        return img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label
    
    def clean_preds(self, gt_rels, pred_rels, objs, rels):
        final_pred_rels = []
        all_objs = []; all_rels = []

        for idx, pred in enumerate(pred_rels):
            objs = list(set(triplets[2] for triplets in pred))
            rels = list(set(triplets[1] for triplets in pred))
            print(objs, rels)
            if len(all_objs) > 1:
                if all_objs[-1] == objs:
                    if all_rels[-1] == rels:
                        continue
                else:
                    for relationship in self.category2predicate.keys():
                        rel_count = 0; sit_count = 0; stand_count = 0
                        for triplet in pred:
                            if triplet[1] in self.category2predicate[relationship]:
                                rel_count += 1
                                if rel_count > len(objs):
                                    pred.remove(triplet)
                            
                            if relationship == 'attention': # Only to be run once
                                if any(item in ['other_relationship', 'unsure'] for item in triplet):
                                    pred.remove(triplet)
                            
                                if triplet[1] == 'standing':
                                    if sit_count > 1:
                                        pred.remove(triplet)
                                    stand_count += 1
                                elif triplet[1] == 'sitting':
                                    if sit_count > 1:
                                        pred.remove(triplet)
                                    sit_count += 1
        
                    final_pred_rels.append(pred)
            else:
                final_pred_rels.append(pred)
            all_objs.append(objs)
            all_rels.append(rels)
        return final_pred_rels