
import argparse
import os
from data_utils.DDateLoader import AllDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


classes = ['power','low_veg','imp_surf','car','fence_hedge','roof','fac','shrub','tree']
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def cf_mat(all_result, seg_label_to_cat, pred_val, batch_label):
    for id in seg_label_to_cat.keys():
        idx = np.where(batch_label==id)
        if len(idx) == 0:
            continue
        for i in pred_val[idx]:
            all_result[id][i] += 1
    return all_result

def overall_acc(all_result):
    correct, all_seen = 0, 0
    for id in seg_label_to_cat.keys():
        correct += all_result[id][id]
        all_seen += sum(all_result[id])
    return correct / float(all_seen)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=128, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=512, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int,  default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--block_size', type=int, default=30, help='the size of the patch')

    return parser.parse_args()

def main(args):
    def log_string(str):
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('{}_{}_{}'.format(args.model, args.npoint, args.block_size))
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/3DLabeling/'
    NUM_CLASSES = 9
    stride = 10
    padding = 0.02
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    all_result = {}

    for i in range(len(classes)):
        all_result[i] = [0] * len(classes)
    
    print("start loading test data ...")
    TEST_DATASET = AllDataset(split='test', data_root=root, num_point=NUM_POINT, block_size=args.block_size, stride=stride, padding=padding, transform=None)

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module('models.%s' % args.model)
    
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    start_epoch = checkpoint['epoch']
    classifier.load_state_dict(checkpoint['model_state_dict'])
    log_string('Use pretrain model')


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    epoch = 0


    lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
    if momentum < 0.01:
        momentum = 0.01
    classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))

    '''Evaluate on chopped scenes'''
    with torch.no_grad():
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        log_string('---- EVALUATION ----')
        for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            classifier = classifier.eval()
            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            pred_val = np.argmax(pred_val, 2)
            all_result = cf_mat(all_result, seg_label_to_cat, pred_val, batch_label)
            
        log_string("- \t\tpower\tlow_veg\t imp_surf\tcar\tfence_hedge\troof\t fac\t shrub\t tree")
        trim = '%s' % args.model
        for i in range(NUM_CLASSES):
            record = all_result[i]
            record = np.array(record) / (sum(record)  + 1e-6) * 100
            print("%s \t%.1f\t %.1f\t  %.1f\t       %.1f\t %.1f   \t%.1f \t %.1f\t  %.1f\t %.1f" % (seg_label_to_cat[i] + ' ' * (12 - len(seg_label_to_cat[i])),
            record[0],record[1],record[2],record[3],record[4],record[5],record[6],record[7],record[8]))
            trim += ",%.1f" % record[i]
        print(trim)
        print("Overall Accuary : %.1f" % (overall_acc(all_result) * 100))


if __name__ == '__main__':
    args = parse_args()
    main(args)

