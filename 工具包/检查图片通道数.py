import cv2

def check_image_dimensions(image_path):
    # 读取图像，使用 IMREAD_UNCHANGED 来避免颜色空间转换，读取原始图像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 检查图像是否加载成功
    if image is not None:
        # 获取图像的形状 (height, width, channels)
        # 如果是灰度图，或者图像没有alpha通道，channels 可能会不存在
        if len(image.shape) == 3:  # 如果是彩色图像或包含 alpha 通道
            height, width, channels = image.shape
            print(f"Image shape: {image.shape}")  # 打印图像的尺寸 (height, width, channels)
            print(f"Height: {height}, Width: {width}, Channels: {channels}")
        elif len(image.shape) == 2:  # 如果是灰度图
            height, width = image.shape
            print(f"Image shape: {image.shape}")  # 打印图像的尺寸 (height, width)
            print(f"Height: {height}, Width: {width}, Channels: 1")
    else:
        print(f"Error: Unable to load the image from {image_path}")

# 设定图像路径
image_path = r'D:\HTNet-master\NEW_MODEL\ECG_TU_224_1\01_01.jpg'  # 替换为你的图像路径

# 调用函数查看图像的尺寸和通道数
check_image_dimensions(image_path)
