import os
import numpy as np
from torch.utils.data import Dataset

class ISPRSDataset(Dataset):
    def __init__(self, data_root, split='train', num_point=512, stride=10, block_size=30, padding=0.02, transform=None):
        super().__init__()
        self.block_size = block_size
        self.num_point = num_point
        self.transform = transform
        self.stride = stride

        data_name = 'Vaihingen3D_Traininig.pts' if split == 'train' else 'Vaihingen3D_EVAL_WITH_REF.pts'
        labelweights = np.zeros(9)
        
        data_path = os.path.join(data_root, data_name)
        data = np.loadtxt(data_path)  # xyzrgbl, N*7
        tmp, _ = np.histogram(data[:,6], range(10))
        labelweights += tmp
        
        self.coord_min, self.coord_max = np.amin(data, axis=0)[:4], np.amax(data, axis=0)[:4]
        
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        self.labelweights = labelweights

        grid_x = int(np.ceil(float(self.coord_max[0] - self.coord_min[0] - self.block_size) / stride) + 1)
        grid_y = int(np.ceil(float(self.coord_max[1] - self.coord_min[1] - self.block_size) / stride) + 1)
        
        all_datapoint = []
        all_s_x, all_s_y = [], []
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = self.coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, self.coord_max[0])
                s_x = e_x - block_size
                s_y = self.coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, self.coord_max[1])
                s_y = e_y - block_size
                point_idxs = np.where(
                    (data[:, 0] >= s_x - padding) & (data[:, 0] <= e_x + padding) & (data[:, 1] >= s_y - padding) & (
                                data[:, 1] <= e_y + padding))[0]
                if point_idxs.size == 0:
                    continue
                all_s_x.append(s_x)
                all_s_y.append(s_y)
                all_datapoint.append(point_idxs)
        
        self.data = data
        self.all_s_x = all_s_x
        self.all_s_y = all_s_y
        self.all_datapoint = all_datapoint

        print("Totally {} samples in {} set.".format(len(all_datapoint), split))

    def __getitem__(self, idx):
        point_idxs = self.all_datapoint[idx]
        s_x = self.all_s_x[idx]
        s_y = self.all_s_y[idx]

        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])

        num_batch = int(np.ceil(point_idxs.size / self.num_point))
        point_size = int(num_batch * self.num_point)
        replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
        point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
        point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
        np.random.shuffle(point_idxs)
        data_batch = self.data[point_idxs, :6]
        normlized_xyz = np.zeros((point_size, 3))
        normlized_xyz[:, 0] = data_batch[:, 0] / self.coord_max[0]
        normlized_xyz[:, 1] = data_batch[:, 1] / self.coord_max[1]
        normlized_xyz[:, 2] = data_batch[:, 2] / self.coord_max[2]

        coord_iter = self.coord_max - self.coord_min
        data_batch[:, 0] = (data_batch[:, 0] - (s_x + self.block_size / 2.0)) / coord_iter[0]
        data_batch[:, 1] = (data_batch[:, 1] - (s_y + self.block_size / 2.0)) / coord_iter[1]
        data_batch[:, 2] = data_batch[:, 2] / coord_iter[2]
        data_batch[:, 3] = data_batch[:, 3] / self.coord_max[3]

        data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
        label_batch = self.data[point_idxs,6].astype(int)
        batch_weight = self.labelweights[label_batch]
        
        data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
        label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
        sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
        index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        
        data_room = data_room.reshape((-1, self.num_point, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.num_point))
        sample_weight = sample_weight.reshape((-1, self.num_point))
        index_room = index_room.reshape((-1, self.num_point))

        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.all_datapoint)

if __name__ == '__main__':
    data_root = 'data/'
    num_point, block_size, stride, padding = 512, 30, 10, 0.02

    point_data = ISPRSDataset(split='train', data_root=data_root, num_point=num_point, block_size=block_size, stride=stride, padding=padding, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)

    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()