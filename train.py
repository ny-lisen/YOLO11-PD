import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO('')            #Model Path
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='',        # Dataset Path
                cache=False,
                imgsz=640,
                epochs=600,
                batch=16,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                patience=50, # close earlystop
                amp=False, # close amp
                # fraction=0.2,
                project='',     #The project path you want to save
                name='',        #Name your weight
                )