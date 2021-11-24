# Author: Little-Chen
# Emial: Chenxiuyan_t@163.com
import torch
import cv2
from PIL import Image
from torch._C import Size
from torchvision import transforms


if __name__ == '__main__':

    # 加载模型
    # model = torch.load('./model/CNN.pth',map_location=torch.device('cpu'))
    model = torch.load(r'F:\QQPCmgr\Desktop\GR_CNN\results\epoch_100_lr_0.001_batch_size_train_128_2021-11-22 12-42-10\CNN.pth',map_location=torch.device('cpu'))
    model.eval()

    # 定义预训练变换
    preprocess_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    capture = cv2.VideoCapture(0)
    
    if cv2.VideoCapture.isOpened(capture):
        print("open camera succeed !!")
    else:
        print("open camera failed !!")


    class_names = ['Nothing!', 'Paper!', 'Rock!', 'Scissors']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnt = 0
    while(True):
        cnt = [0,0,0,0,0] #统计三帧图片不同手势出现的次数
        for i in range(3):

            # 读取3帧 减缓切换手势识别错误
            _, frame = capture.read()

            frame2 = frame

            frame = cv2.resize(frame, (128, 128))
            cv2.imwrite('./tmp.jpg', frame)

            frame = cv2.imread('./tmp.jpg')

            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) # mat to PIL

            image_tensor = preprocess_transform(image).unsqueeze_(0).to(device)

            out = model(image_tensor)
            
            # 得到预测结果，并且从大到小排序
            _, indices = torch.sort(out, descending=True)

            # 返回每个预测值的百分数
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

            # print([(class_names[idx], percentage[idx].item()) for idx in indices[0][:5]])

            # 概率大于等于80时才显示

            if(percentage[indices[0][0]].item() >= 70):
                cnt[indices[0][0]] += 1

        #连续三帧相同表示识别成功
        if (cnt[indices[0][0]] == 3):
            cv2.putText(frame2, (class_names[indices[0][0]]), (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        cv2.imshow("Img", frame2)
            
        cv2.waitKey(1)