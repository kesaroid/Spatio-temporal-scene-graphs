import os


ag_data_root = '/media/engineering/Millenium Falcon/Thesis/ActionGenome/dataset/ag/'

image_path = os.path.join(ag_data_root, 'frames')
data_stats = 'annotations/COCO/AG-SGG-test.json'
test_data = "annotations/COCO/AG-test.json"

# load detected results
detected_origin_path = '/home/engineering/Documents/Thesis/Scene-Graph-Benchmark.pytorch/output/Saved_models/RL=Transformer_v2/inference_final/AG_v3Graph_test/'
db_path = 'videos_cache.pkl'

corpus_path = 'finalRun/ag_corpus.pickle'

g_embed_path = 'finalRun/graph2vec'
train_path = 'finalRun/ag_train.pickle'
gfeat_path = 'finalRun/G_features.pickle'

model_state_dict = 'finalRun/mgat.pth'