import os
import numpy as np
import cv2
import time
import argparse
import random
import torch
import torch.nn as nn
import importlib.util
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR



from utils.save_tool import save_results_to_txt, plot_and_save
from utils.DA import expand_dataset_with_augmentation_tensor, data_augmentation

def lr_lambda(step):#学习率调度函数
    warmup_steps = 30
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))  # 线性预热
    else:
        return 1 / (step ** 0.5)  # 反向比例衰减

def get_image_names_from_txt(txt_file_path):
    """
    从指定的 txt 文件中读取图片名称，每行一个名称，
    并返回一个包含所有图片名称的列表。
    :param txt_file_path: txt 文件的绝对路径
    :return: 包含图片名称的列表
    """
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        # 读取所有行并去除换行和多余空白，过滤掉空行
        image_names = [line.strip() for line in f if line.strip()]
    return image_names

def input_bag(dataset_base_folder):
    whole_flow0_path = os.path.join(dataset_base_folder, 'flow(on-ap)')
    whole_flow1_path = os.path.join(dataset_base_folder, 'flow(of-ap)')
    whole_ECG_path = os.path.join(dataset_base_folder, 'ECG')


    whole_flow0_imgs = os.listdir(whole_flow0_path)
    img_bag = {}

    for n_img in whole_flow0_imgs:

        if n_img.endswith('.jpg'):
            img_bag[n_img] = []

            # 使用 cv2.IMREAD_UNCHANGED 确保读取16位深度图 和 8位光流图
            flow0 = cv2.imread(os.path.join(whole_flow0_path, os.path.splitext(n_img)[0] + '.jpg'),
                                     cv2.IMREAD_UNCHANGED)
            flow1 = cv2.imread(os.path.join(whole_flow1_path, os.path.splitext(n_img)[0] + '.jpg'),
                                    cv2.IMREAD_UNCHANGED)
            ECG = cv2.imread(os.path.join(whole_ECG_path, os.path.splitext(n_img)[0] + '.jpg'),
                                    )
            # 读取光流图像（假设是8位彩色图像）

            # 检查文件是否加载成功
            if flow0 is None:
                print(f"缺失起始-顶点帧: {os.path.join(whole_flow0_path, os.path.splitext(n_img)[0] + '.jpg')}")
            if flow1 is None:
                print(f"缺失偏移—顶点帧: {os.path.join(whole_flow1_path, os.path.splitext(n_img)[0] + '.jpg')}")
            if ECG is None:
                print(f"缺失ECG图: {os.path.join(whole_ECG_path, os.path.splitext(n_img)[0] + '.jpg')}")


            # 转换图像为float32类型
            flow0 = flow0.astype(np.float32)  # 转换为 float32类型
            flow1 = flow1.astype(np.float32)    # 转换为 int32
            ECG = ECG.astype(np.float32)  # 转换为 int32




            # 将图像添加到 img_bag 字典
            img_bag[n_img].append(flow0)
            img_bag[n_img].append(flow1)
            img_bag[n_img].append(ECG)


    return img_bag    # (165,3)

# 创建保存基路径
def create_unique_directory(base_path, exp_name):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    exp_path = os.path.join(base_path, exp_name)
    counter = 1

    while os.path.exists(exp_path):
        exp_path = os.path.join(base_path, f"{exp_name}{counter}")
        counter += 1
    os.makedirs(exp_path)

    return exp_path


# 动态导入模型
def load_model(model_file_path, model_name, device, imgs, classes, pretrained_weights_path=None, subName=None):
    """
    动态加载模型，传递输入通道数和缩放后的图像尺寸
    """
    if not os.path.isfile(model_file_path):
        raise FileNotFoundError(f"模型文件缺失: {model_file_path}")

    spec = importlib.util.spec_from_file_location("模型文件", model_file_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    if not hasattr(model_module, model_name):
        raise AttributeError(f"未找到模型的类 {model_name} ")

    model_class = getattr(model_module, model_name)

    # 创建模型时通过**kwargs传递额外的参数
    model_kwargs = {}

    if imgs  in [1 , 3 ,] :
        model_kwargs['imgs'] = imgs
    else:
        raise ValueError(f"{imgs}错误，期望值为1、3或4")
    if classes:
        model_kwargs['classes'] = classes
    else:
        print("未传入classes参数，默认三分类任务")

    if pretrained_weights_path:
        model_kwargs['pretrained_weights_path'] = pretrained_weights_path
    if subName:
        model_kwargs['subName'] = subName
    # 实例化模型并将其移动到指定设备
    model = model_class(**model_kwargs).to(device)
    return model

# 学习率调度器（线性预热 + 反向比例衰减）
def lr_lambda(step, warmup_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))  # 线性预热
    else:
        return 1 / (step ** 0.5)  # 反向比例衰减

def main(config):
    warmup_steps = 30
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    epochs = config.epochs
    num_workers = config.num_workers
    accum_iter = config.accum_iter
    model_file_path = config.model_file
    model_name = config.model_name
    exp_name = config.exp_name
    dataset_base_folder = config.dataset_base_folder  # 数据集根文件夹
    save_base_path = config.save_base_path
    all_accuracy_dict = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.train:
        exp_path = create_unique_directory(save_base_path, exp_name)
    else:
        exp_path = os.path.join(save_base_path, exp_name)

    print(
        f'lr={learning_rate}, '
        f'epochs={epochs}, '
        f'batch_size={batch_size}, '
        f'device={device}, '
        f'model_file={model_file_path}, '
        f'model_name={model_name}, '
        f'exp_path={exp_path}, '
        f'num_workers={num_workers}')

    total_gt = []
    total_pred = []
    best_total_pred = []
    t = time.time()

    # 根据dataset_base_folder动态生成路径
    main_path = os.path.join(dataset_base_folder, 'loso')
    subName = os.listdir(main_path)
    input = input_bag(dataset_base_folder)

    for n_subName in subName:
        print('Subject:', n_subName)
        y_train = []
        y_test = []
        imgs_train = []
        imgs_test = []
        # 用于绘制损失曲线的列表
        train_losses = []
        train_acces = []

        val_losses = []
        val_acces = []

        pred_lists= []

        best_epoch = 0

        # loss_fn = get_loss_fn(os.path.join(dataset_base_folder, 'loso', n_subName, 'u_train'), device)  # 根据训练集路径得到
        loss_fn = nn.CrossEntropyLoss()

        expression = os.listdir(os.path.join(main_path, n_subName, 'u_train'))

        for n_expression in expression:
            img = get_image_names_from_txt(os.path.join(main_path, n_subName, 'u_train',n_expression))
            n_expression, _ = os.path.splitext(n_expression)
            for n_img in img:
                y_train.append(int(n_expression))

                flow0, flow1 , ECG ,  = input[n_img]

                if config.imgs == 3:
                    flow0=torch.tensor(flow0)       #   224 224 3
                    flow1 = torch.tensor(flow1)        #  224 224 3
                    ECG = torch.tensor(ECG)         #  224 224 3

                    imgs = torch.stack([flow0, flow1, ECG], dim=0)  # 3 224 224 3

                else:
                    print("imgs错误")
                imgs_train.append(imgs)
        imgs_train = torch.stack(imgs_train, dim=0)          #  imgs=3 --(nums, 3, 224, 224 ,3)
        imgs_train =imgs_train.permute(0, 1, 4, 2, 3)          # imgs=3 --(nums, 3=die, 3=C , 224, 224 )    ......
        y_train = torch.Tensor(y_train).to(dtype=torch.long)

        expression = os.listdir(os.path.join(main_path, n_subName, 'u_test'))
        for n_expression in expression:
            img = get_image_names_from_txt(os.path.join(main_path, n_subName, 'u_test', n_expression))
            n_expression, _ = os.path.splitext(n_expression)
            for n_img in img:
                y_test.append(int(n_expression))

                flow0, flow1, ECG = input[n_img]

                if config.imgs == 3:
                    flow0 = torch.tensor(flow0)  # 224 224 3
                    flow1 = torch.tensor(flow1)  # 224 224 3
                    ECG = torch.tensor(ECG)  # 224 224 3

                    imgs = torch.stack([flow0, flow1, ECG],dim=0)  # 3 224 224 3
                else:
                    print("imgs错误")
                imgs_test.append(imgs)
        imgs_test = torch.stack(imgs_test,dim=0)  # imgs=3 --(165, 3, 224, 224 ,3)
        imgs_test = imgs_test.permute(0, 1, 4, 2, 3)  # imgs=1 --(165, 3, 3 , 224, 224 )    ......
        y_test = torch.Tensor(y_test).to(dtype=torch.long)





        if config.pretrained_weights_path:
            model = load_model(model_file_path, model_name, device, imgs=config.imgs, classes=config.classes,
                               pretrained_weights_path=config.pretrained_weights_path, subName=n_subName)
        else:
            model = load_model(model_file_path, model_name, device, imgs=config.imgs, classes=config.classes)

        weights_dir = os.path.join(exp_path, 'weights')  # # 创建 weights 文件夹路
        weight_path = os.path.join(weights_dir, f'{n_subName}.pth')
        if config.train:
            #print('train')
            if not os.path.exists(weights_dir):  ## 如果 weights 文件夹不存在，则创建它
                os.makedirs(weights_dir)
            else:
                pass


        else:
            # 验证模式下
            model.load_state_dict(torch.load(weight_path))

        if config.pretrained_weights_path:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)  # ,weight_decay=1e-3

        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # ,weight_decay=1e-3

        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps))

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []


        if config.use_data_augmentation == 1:  # 数据扩充
           imgs_train, y_train = expand_dataset_with_augmentation_tensor(imgs_train, y_train, data_augmentation)


        train_dl = DataLoader(TensorDataset(imgs_train, y_train), batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)

        test_dl = DataLoader(TensorDataset(imgs_test, y_test), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

        if config.train:
            # 添加进度条
            progress_bar = tqdm(total=epochs, desc=f"Training {n_subName}", unit="epoch")


        for epoch in range(1, epochs + 1):
            if config.train:
                model.train()
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0
                for step, batch in enumerate(train_dl):
                    optimizer.zero_grad()
                    x = batch[0].to(device)  # 图像
                    y = batch[1].to(device)  # 标签
                    yhat = model(x)
                    loss = loss_fn(yhat, y) / accum_iter  # 计算损失，并进行归一化
                    loss.backward()   # 只计算梯度，每次反向传播计算的梯度会和之前的梯度累加,不会对模型参数进行任何修改
                    # 梯度累加
                    if (step + 1) % accum_iter == 0 or (step + 1) == len(train_dl):
                        optimizer.step()  # 更新模型
                        optimizer.zero_grad()  # 清空梯度

                    train_loss += loss.item()  * x.size(0)  # loss.data.item() 不安全 可以改用loss.item()
                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]


                train_acc = num_train_correct / num_train_examples
                train_loss = train_loss / len(train_dl.dataset)
                # 保存当前 epoch 的训练损失
                train_losses.append(train_loss)
                train_acces.append(train_acc)

                scheduler.step()  # 这里是在训练部分进行学习率更新

            # 定义一个列表来累积所有批次的真实标签和预测标签
            total_gt = []
            total_pred = []

            model.eval()
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0

            with torch.no_grad():
                for batch in test_dl:
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    yhat = model(x)
                    loss = loss_fn(yhat, y)
                    val_loss += loss.item()  * x.size(0)
                    num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_val_examples += y.shape[0]

                    # 累积所有batch的真实标签
                    total_gt.extend(y.tolist())
                    # 累积所有batch的预测标签  # torch.max(yhat, 1)[1] 代表预测类别
                    total_pred.extend(torch.max(yhat, 1)[1].tolist())  # 添加到每轮的预测列表中

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(test_dl.dataset)

            # 保存当前 epoch 的验证结果
            pred_lists.append(total_pred)  # total_pred：该轮次内 所有验证结果     pred_lists： 所有轮次 验证结果
            val_losses.append(val_loss)
            val_acces.append(val_acc)

            # 更新最佳结果
            if best_accuracy_for_each_subject <= val_acc:
                best_epoch = epoch  # 记录当前满足条件的 epoch
                best_accuracy_for_each_subject = val_acc
                best_each_subject_pred = total_pred
                if config.train:
                    torch.save(model.state_dict(), weight_path)


            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix({'Train Acc': train_acc, 'Val Acc': val_acc})
        progress_bar.close()

        print('Best Predicted:', best_each_subject_pred)
        print('Ground Truth  :', total_gt)
        print(f"Best Epoch: {best_epoch}")

        accuracydict = {'pred': best_each_subject_pred, 'truth': total_gt}
        all_accuracy_dict[n_subName] = accuracydict


        best_total_pred.extend(best_each_subject_pred)

        plot_and_save(train_losses, val_losses,train_acces,val_acces,exp_path, n_subName)  # 训练完成，绘制并保存损失曲线

        save_results_to_txt(train_losses, val_losses,  train_acces, val_acces, pred_lists, total_gt, config, exp_path, best_epoch, n_subName)



    print('Final Evaluation: ')
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict)
    print(exp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1, help='Train(1) or use pre-trained weight for prediction(0)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument("--use_data_augmentation", type=int, default=0,help="Whether to apply data augmentation (True(1)/False(0))")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--accum_iter', type=int, default=4, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')  # .00005
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--imgs', type=int, default=3,help='Type of images to use for training (2 for optical and apex)')

    parser.add_argument('--classes', type=int, default=4, help='Number of class for training ')

    parser.add_argument('--pretrained_weights_path', type=str, default='', help='示例：3categories/HTN(150)-f-0.8871-0.8812/weights')

    parser.add_argument('--model_file', type=str, default=r'model_dual_eda_flow.py', help='Path to the model file')
    parser.add_argument('--model_name', type=str, default='test', help='Model name to use for training')

    parser.add_argument('--save_base_path', type=str, default='4categories', help='Experiment path for saving models')
    parser.add_argument('--exp_name', type=str, default='PartC(100)_test', help='Experiment name for saving models')

    parser.add_argument('--dataset_base_folder', type=str, default='PartC',help='Base folder for onset, apex, and flow data')

    config = parser.parse_args()
    main(config)
