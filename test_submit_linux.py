# @Author: wuwuwu.ZhouHao 
# @Date: 2019-06-11 21:14:02 
# @Last Modified by:   wuwuwu.ZhouHao  
# @Last Modified time: 2019-06-11 21:14:02
import ffmpeg
import os
import shutil
import models
import torch
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def video2img(dir_video, dir_imgs):
    stream = ffmpeg.input(dir_video)
    stream = ffmpeg.output(stream, dir_imgs)
    ffmpeg.run(stream)

# # 1转换y4m格式为bmp格式图片序列
# input_dir = '/data/data_youku/data_test_1/'
# list_files = os.listdir(input_dir)

# for i in range(len(list_files)):
#     file_name = list_files[i]
#     dir_video = input_dir + file_name
#     dir_out = '/data/data_youku/data_test_img_1/' + file_name[:-4] + '/'
#     if not os.path.exists(dir_out):
#         os.makedirs(dir_out)
#     dir_out = dir_out + '%3d.bmp'
#     video2img(dir_video, dir_out)

# # 2将bmp格式图片序列进行抽帧操作
# input_dir = '/data/data_youku/data_test_img_1/'
# list_imgs = os.listdir(input_dir)

# for i in range(len(list_imgs)):
#     imgs_file_name = list_imgs[i]
#     dir_out = '/data/data_youku/data_test_img_sub_1/' + imgs_file_name + '/'
#     if not os.path.exists(dir_out):
#         os.makedirs(dir_out)
#     dir_in = input_dir + imgs_file_name + '/001.bmp'
#     dir_out_copy  = dir_out + '001.bmp'
#     shutil.copyfile(dir_in, dir_out_copy)
#     dir_in = input_dir + imgs_file_name + '/026.bmp'
#     dir_out_copy  = dir_out + '002.bmp'
#     shutil.copyfile(dir_in, dir_out_copy)
#     dir_in = input_dir + imgs_file_name + '/051.bmp'
#     dir_out_copy  = dir_out + '003.bmp'
#     shutil.copyfile(dir_in, dir_out_copy)
#     dir_in = input_dir + imgs_file_name + '/076.bmp'
#     dir_out_copy  = dir_out + '004.bmp'
#     shutil.copyfile(dir_in, dir_out_copy)

# 3依次进行读取测试图片并完成高清图像生成
input_dir = '/data/data_youku//data_test_img_1/'
input_dir_sub = '/data/data_youku/data_test_img_sub_1/'

model_g = models.net_SR_5().to(device)
# model_g = torch.nn.DataParallel(model_g, device_ids=[0,1])
model_dir = './model/model_SR5.pb'
model_g.load_state_dict(torch.load(model_dir))

img_transform = transforms.Compose([
    transforms.Resize((270, 480)),
    transforms.ToTensor(),
])
img_untransform = transforms.ToPILImage()

list_imgs = os.listdir(input_dir)
list_imgs_sub = os.listdir(input_dir_sub)
for i in range(len(list_imgs)):
    imgs_name = list_imgs[i]
    list_imgs_name = os.listdir(input_dir + '/' +imgs_name)
    # print(imgs_name)
    for j in range(len(list_imgs_name)):
        img_name = list_imgs_name[j]
        # print(img_name)
        img_dir = input_dir + '/' + imgs_name + '/' + img_name
        img = Image.open(img_dir)
        img_tensor = img_transform(img)
        img_tensor = torch.stack([img_tensor], 0)
        img_tensor = img_tensor.to(device)
        out_tensor = model_g(img_tensor)
        out_tensor = out_tensor.cpu().clone()
        out_tensor = out_tensor.squeeze(0)
        out = img_untransform(out_tensor)
        
        img_save_dir = '/data/data_youku/result_submit/data_test_img_h_1/' + imgs_name + '/'
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        img_save_dir = img_save_dir + img_name
        out.save(img_save_dir)
        # out.show()


for i in range(len(list_imgs_sub)):
    imgs_name = list_imgs[i]
    list_imgs_name = os.listdir(input_dir_sub + '/' +imgs_name)
    # print(imgs_name)
    for j in range(len(list_imgs_name)):
        img_name = list_imgs_name[j]
        # print(img_name)
        img_dir = input_dir_sub + '/' + imgs_name + '/' + img_name
        img = Image.open(img_dir)
        img_tensor = img_transform(img)
        img_tensor = torch.stack([img_tensor], 0)
        img_tensor = img_tensor.to(device)
        out_tensor = model_g(img_tensor)
        out_tensor = out_tensor.cpu().clone()
        out_tensor = out_tensor.squeeze(0)
        out = img_untransform(out_tensor)
        
        img_save_dir = '/data/data_youku/result_submit/data_test_img_sub_h_1/' + imgs_name + '/'
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        img_save_dir = img_save_dir + img_name
        out.save(img_save_dir)

# 4将图像合成视频
def imgs2dideo(imgs_dir, video_dir):
    stream = ffmpeg.input(imgs_dir)
    stream = ffmpeg.output(stream, video_dir, pix_fmt='yuv420p')
    ffmpeg.run(stream)

input_dir = '/data/data_youku/result_submit/data_test_img_h_1/'
input_sub_dir = '/data/data_youku/result_submit/data_test_img_sub_h_1/'
list_imgs_h = os.listdir(input_dir)
list_imgs_sub_h = os.listdir(input_sub_dir)
for i in range(len(list_imgs_h)):
    imgs_name = list_imgs_h[i]
    out_dir = '/data/data_youku/result_submit/data_test_h_1/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir = out_dir + imgs_name[:-2] + '_h_Res.y4m'
    imgs2dideo(input_dir+imgs_name+'/%3d.bmp' , out_dir)

for i in range(len(list_imgs_sub_h)):
    imgs_name = list_imgs_sub_h[i]
    out_dir = '/data/data_youku/result_submit/data_test_h_sub_1/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir = out_dir + imgs_name[:-2] + '_h_Sub25_Res.y4m'
    imgs2dideo(input_sub_dir+imgs_name+'/%3d.bmp' , out_dir)

    


