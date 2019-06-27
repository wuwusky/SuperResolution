# @Author: wuwuwu.ZhouHao 
# @Date: 2019-06-02 14:07:01 
# @Last Modified by:   wuwuwu.ZhouHao  
# @Last Modified time: 2019-06-02 14:07:01

import models
import torch
# import cv2
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import time
import random
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
import numbers
from PIL import ImageFilter

class RandomHorizontalFlip_pairs(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img_1, img_2):
        if random.random() < self.p:
            out1 = F.hflip(img_1)
            out2 = F.hflip(img_2)
            return out1, out2
        return img_1, img_2
class RandomVerticalFlip_pairs(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img_1, img_2):
        if random.random() < self.p:
            out1 = F.vflip(img_1)
            out2 = F.vflip(img_2)
            return out1, out2
        return img_1, img_2
def hflip(img):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return img.transpose(Image.FLIP_LEFT_RIGHT)
def vflip(img):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return img.transpose(Image.FLIP_TOP_BOTTOM)
class RandomCrop_pairs(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_1, img_2):
        if self.padding is not None:
            img_1 = F.pad(img_1, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and img_1.size[0] < self.size[1]:
            img_1 = F.pad(img_1, (self.size[1] - img_1.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img_1.size[1] < self.size[0]:
            img_1 = F.pad(img_1, (0, self.size[0] - img_1.size[1]), self.fill, self.padding_mode)
        
        if self.padding is not None:
            img_2 = F.pad(img_2, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and img_2.size[0] < self.size[1]:
            img_2 = F.pad(img_2, (self.size[1] - img_2.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img_2.size[1] < self.size[0]:
            img_2 = F.pad(img_2, (0, self.size[0] - img_2.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img_1, self.size)

        out_1 = F.crop(img_1, i, j, h, w)
        out_2 = F.crop(img_2, i*4, j*4, h*4, w*4)
        return out_1, out_2

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)
class ColorJitter(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):

        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
# # #----------------------------------------windows 测试用-----------------------------------------
# test_model = models.net_FSRCNN(3).cuda()

# data_dir = 'e:/temp_file/data_youku.txt'
# with open(data_dir,'r') as f:
#     lines = f.readlines()
# f.close()

# list_line = lines[0].strip('\n').split(' ')
# # input_img = cv2.imread(list_line[0])
# # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
# # input_img = np.transpose(input_img, (2,0,1))
# # input_tensor = torch.from_numpy(input_img)
# # print(input_tensor.shape)
# # output_img = cv2.imread(list_line[1])
# # output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
# # output_img = np.transpose(output_img, (2,0,1))
# # output_tensor = torch.from_numpy(output_img)
# # print(output_tensor.shape)

# # input_tensor_batch = torch.stack([input_tensor], 0)
# # input_tensor_batch = input_tensor_batch.cuda()
# # print(input_tensor_batch.shape)
# # out = test_model(input_tensor_batch)
# # print(out.shape)
# my_transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
# input_img = Image.open(list_line[0])
# input_tensor = my_transform(input_img)
# print(input_tensor.shape)
# input_tensor_batch = torch.stack([input_tensor], 0)
# print(input_tensor_batch.shape)

# input_tensor_batch = input_tensor_batch.cuda()
# out = test_model(input_tensor_batch)
# print(out.shape)
# # #----------------------------------------windows 测试用-----------------------------------------
class my_dataset(Dataset):
    def __init__(self, list_imgs_pairs, transform=None):
        self.imgs_pairs = list_imgs_pairs
        self.transform = transform
    def __getitem__(self, item):
        in_img = Image.open(self.imgs_pairs[item][0])
        ou_img = Image.open(self.imgs_pairs[item][1])

        self.transform_out = transforms.Compose([
                transforms.Resize((1080, 1920)),
                transforms.ToTensor(),
            ])
        h_flip = RandomHorizontalFlip_pairs()
        in_img, ou_img = flip(in_img, ou_img)
        v_flip = RandomVerticalFlip_pairs()
        in_img, ou_img = v_flip(in_img, ou_img)

        if self.transform != None:
            in_tensor = self.transform(in_img)
            ou_tensor = self.transform_out(ou_img)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((270, 480)),
                transforms.ToTensor(),
            ])
            in_tensor = self.transform(in_img)
            ou_tensor = self.transform_out(ou_img)

        return in_tensor, ou_tensor
    def __len__(self):
        return len(self.imgs_pairs)

class my_dataset_aux(Dataset):
    def __init__(self, list_imgs_pairs, transform=None, flag_pre=False):
        self.imgs_pairs = list_imgs_pairs
        self.transform = transform
        self.flag_pre = flag_pre
    def __getitem__(self, item):
        in_img = Image.open(self.imgs_pairs[item][0])
        ou_img = Image.open(self.imgs_pairs[item][1])

        self.transform_out = transforms.Compose([
            transforms.Resize((1080, 1920)),
        ])
        if self.flag_pre:
            h_flip = RandomHorizontalFlip_pairs()
            in_img, ou_img = h_flip(in_img, ou_img)
            v_flip = RandomVerticalFlip_pairs()
            in_img, ou_img = v_flip(in_img, ou_img)


        in_img = self.transform(in_img)
        ou_img = self.transform_out(ou_img)
        in_img_edge = in_img.filter(ImageFilter.EDGE_ENHANCE)
        in_img_blur = in_img.filter(ImageFilter.SMOOTH)

        trans_out = transforms.ToTensor()
        in_tensor_edge = trans_out(in_img_edge)
        in_tensor_blur = trans_out(in_img_blur)
        in_tensor = trans_out(in_img)
        ou_tensor = trans_out(ou_img)

        return in_tensor, in_tensor_edge, in_tensor_blur, ou_tensor
    def __len__(self):
        return len(self.imgs_pairs)

def get_list_img_pairs(data_dir):
    with open(data_dir, 'r') as f:
        lines = f.readlines()
        f.close()
        for i in range(len(lines)):
            lines[i] = lines[i].strip('\n').split(' ')
    return lines

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1-lam)*x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_mixup = lam * criterion(pred, y_a) + (1-lam)*criterion(pred, y_b)
    return loss_mixup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    LR = 1e-3
    train_data_dir = '/data/data_youku/data_train.txt'
    val_data_dir = '/data/data_youku/data_val.txt'
    val_data_dir_test = '/data/data_youku/data_test_1.txt'
    flag_train = True
    Batch_size = 2
    sample_step = 2
    flag_for_test = False
    flag_aux = False
    flag_mixup = False
    if flag_for_test:
        model_dir = './model/model_test.pb'
    else:
        model_dir = './model/model_SR_5_filter.pb'

    if flag_for_test:
        max_epoch = 1000
    else:
        max_epoch = 30

    if flag_train:
        model_train = models.net_SR_5_filter().to(device)
        # model_train = models.SRFBN(in_channels=3, out_channels=3, num_features=48, num_steps=2, num_groups=3, upscale_factor=4, act_type = 'prelu').to(device)
        model_train = torch.nn.DataParallel(model_train, device_ids=[0,1])
        criterion = torch.nn.MSELoss()
        # criterion = torch.nn.L1Loss()
        # optimizer = torch.optim.SGD(model_train.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(model_train.parameters(), lr=LR)

        transform_train = transforms.Compose([
            transforms.Resize((270, 480)),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((270, 480)), 
        ])            
        if flag_for_test:
            list_imgs_pairs_train = get_list_img_pairs(val_data_dir_test)
            list_imgs_pairs_val = get_list_img_pairs(val_data_dir_test)
        else:
            list_imgs_pairs_train = get_list_img_pairs(train_data_dir)
            list_imgs_pairs_val = get_list_img_pairs(val_data_dir)
        list_imgs_pairs_all = list_imgs_pairs_train + list_imgs_pairs_val
        list_imgs_pairs_all_slim = []
        for i in range(len(list_imgs_pairs_all)):
            if i == 0 or (i+1) % sample_step == 0:
                list_imgs_pairs_all_slim.append(list_imgs_pairs_all[i])

        data_train = my_dataset_aux(list_imgs_pairs_train, transform_train, flag_pre=True)
        data_val = my_dataset_aux(list_imgs_pairs_val, transform_val, flag_pre=True)
        data_train_all = my_dataset_aux(list_imgs_pairs_all_slim, transform_train, flag_pre=True)

        if flag_for_test:
            loader_train = DataLoader(data_train, batch_size=Batch_size, shuffle=True, num_workers=16)
        else:
            loader_train = DataLoader(data_train_all, batch_size=Batch_size, shuffle=True, num_workers=16)
        loader_val = DataLoader(data_val, batch_size=Batch_size, shuffle=True, num_workers=16)
        loader_train_all = DataLoader(data_train_all, batch_size=Batch_size, shuffle=True, num_workers=16)

        step_list = []
        loss_list = []
        total_step = len(loader_train)
        cur_lr = LR
        step_iter = 0
        for epoch in range(max_epoch):
            model_train.train()
            time_start = time.time()
            for i,(in_tensor, in_tensor_edge, in_tensor_blur, ou_tensor) in enumerate(loader_train):
                in_tensor = in_tensor.to(device)
                ou_tensor = ou_tensor.to(device)
                in_tensor_edge = in_tensor_edge.to(device)
                in_tensor_blur = in_tensor_blur.to(device)
                
                if flag_mixup:
                    in_tensor_mixup, ou_tensor_mix_a, ou_tensor_mix_b, lam = mixup_data(in_tensor, ou_tensor, alpha=0.25)
                    in_tensor_mixup, ou_tensor_mix_a, ou_tensor_mix_b = map(Variable, (in_tensor_mixup, ou_tensor_mix_a, ou_tensor_mix_b))

                    optimizer.zero_grad()
                    outputs = model_train(in_tensor, in_tensor_edge)
                    loss = mixup_criterion(criterion, outputs, ou_tensor_mix_a, ou_tensor_mix_b, lam)
                else:
                    if flag_aux:
                        optimizer.zero_grad()
                        outputs, outputs_aux = model_train(in_tensor)
                        loss = criterion(outputs, ou_tensor)
                        loss_aux = criterion(outputs_aux, ou_tensor_aux)
                        loss_aux.backward(retain_graph=True)
                        optimizer.step()
                    else:
                        optimizer.zero_grad()
                        outputs = model_train(in_tensor)
                        loss = criterion(outputs, ou_tensor)
                loss.backward()
                optimizer.step()

                if step_iter % int(max_epoch*len(loader_train_all)/1000) == 0:
                    loss_list.append(loss.item())
                    step_list.append(step_iter+1)
                step_iter += 1

                # if (i+1) % int(total_step/100) == 0:
                if flag_for_test:
                    print('Epoch:{}/{}, step:{}/{}, loss:{:.6f}, lr:{:.7f}'
                        .format(epoch+1, max_epoch, i+1, total_step, loss.item(), cur_lr))
                else:
                    if (i+1) % 2 == 0 or i == 0:
                        print('Epoch:{}/{}, step:{}/{}, loss:{:.6f}, lr:{:.7f}'
                            .format(epoch+1, max_epoch, i+1, total_step, loss.item(), cur_lr))
            time_epoch = time.time() - time_start
            if not flag_for_test:
                print('time cost: {:.2f}s'.format(time_epoch))
            if flag_for_test:
                if (epoch + 1) % int(max_epoch/2) == 0:
                    cur_lr = cur_lr * 0.1
                    update_lr(optimizer, cur_lr)
            else:
                if (epoch + 1) % int(max_epoch/6) == 0:
                    cur_lr = cur_lr * 0.5
                    update_lr(optimizer, cur_lr)
            if not flag_for_test:
                model_train.eval()
                torch.save(model_train.state_dict(), model_dir)
            else:
               if (epoch + 1 ) % 50 == 0 :
                   model_train.eval()
                   torch.save(model_train.state_dict(), model_dir)

        plt.figure(1)
        plt.plot(step_list, loss_list)
        plt.show()
    else:
        print('this is for test!')    



                


