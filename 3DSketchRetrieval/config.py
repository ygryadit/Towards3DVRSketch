#DATASET = 'shape2vec'

ROOT_DIR = ''
VIEW_DIR = ROOT_DIR + '/data/view'
POINT_DIR = ROOT_DIR + '/data/point'
LIST_DIR = ROOT_DIR + '/data/list'

PRETRAINED_PATH = ROOT_DIR +'/vgg11_bn-6002323d.pth'

INPUT_W = 224
INPUT_H = 224

NUM_VIEWS = 12
NUM_POINTS = 1024


DATASETS = {
    'ModelNet10': {
        'list_file':
            LIST_DIR + '/ModelNet10/equal/{}.txt',
        'n_classes': 10
    },
    'ShapeNet40': {
        'list_file':
            LIST_DIR + '/ShapeNet40/{}.txt',
        'n_classes': 40
    }
}