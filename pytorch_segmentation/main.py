import torch
import torch.nn as nn
from torch.nn.modules import loss
from torchvision import transforms
from torchvision.transforms.functional import crop
from torchsummary import summary
from torch.optim import SGD, lr_scheduler
from torch.utils.tensorboard import  SummaryWriter
import os, time, cv2, argparse, functools
import numpy as np
import pandas as pd
from PIL import Image
from models import DeepLabv3_plus, build_unet
from functions import CustomImageDataset, imshow, AverageMeter, inter_and_union

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("-exp", "--exp_name", type=str, help='expirement name', default='exp0')
parser.add_argument("-ds", "--dataset_dir", type=str, help='dataset directory', default='data')
parser.add_argument("-m", "--model_name", type=str, help='model name', default='unet')
parser.add_argument("-ne", "--num_epochs", type=int, help='number of training epochs', default=10)
parser.add_argument("-bs", "--batch_size", type=int, help='batch_size', default=2)
parser.add_argument("-img_h", "--image_height", type=int, help='model input image height (and width)', default=64)
parser.add_argument("-pd", "--pred_dir", type=str, help='prediction directory in dataset directory', default=None)
parser.add_argument("-sr", "--sample_resize", type=int, help='sample resize, default=None', default=None)
parser.add_argument("-ic", "--inf_class", type=str, help='inference class name', default=None)
parser.add_argument("-nw", "--num_workers", type=int, help='num_workers for dataloader', default=0)
args = parser.parse_args()


ROOT_DIR = os.path.dirname(os.getcwd())
os.chdir(ROOT_DIR)
EXP_NAME = args.exp_name                                # default='exp0'
DATASET_DIR = args.dataset_dir                          # default='data' 
MODEL_DIR = 'pytorch_segmentation'
MODEL = args.model_name
CSV_FILE = os.path.join(DATASET_DIR, 'dataset_labels.csv')
IMG_HEIGHT = args.image_height                          # default=64
IMG_WIDTH = IMG_HEIGHT
N_CLASSES = len(pd.read_csv(CSV_FILE).name)
BATCH_SIZE = args.batch_size                            # default=2
N_EPOCHS = args.num_epochs                              # default=10
SAMPLE_RESIZE = args.sample_resize                      # default=None
NUM_WORKERS = args.num_workers                          # default=0
RESUME = False
SAVING_STEP = 10 #10 if N_EPOCHS >= 1000 else 10            # int( N_EPOCHS / 10 )
PRED_DIR = os.path.join(args.pred_dir) if not args.pred_dir==None else None
CLASS_LABELS = [str(x) for x in pd.read_csv(CSV_FILE).name]
INF_CLASS_IDX = CLASS_LABELS.index(args.inf_class) if not args.inf_class== None else None

print('\n', '*   -----   ' * 7, '*\n')
# Experiment directory check
if not os.path.isdir(EXP_NAME):
    os.mkdir(EXP_NAME)
    os.mkdir(os.path.join(EXP_NAME, 'weights'))
    os.mkdir(os.path.join(EXP_NAME, 'tb_log'))
    print("Experiment : '{}' has begin.".format(EXP_NAME))
else:
    if not PRED_DIR:
        RESUME = True

# Tensorboard log
writer = SummaryWriter(os.path.join(EXP_NAME, 'tb_log'))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = os.path.join(EXP_NAME, 'weights', '{}_epoch_%d.pth'.format(MODEL))
    print('{}:{}\n'.format( device.type, device.index))


    # model preparation
    if MODEL == 'unet':
        model = build_unet(num_classes=N_CLASSES)
    elif MODEL == 'deeplabv3p':
        model = DeepLabv3_plus(n_classes=N_CLASSES)
    else:
        print('No {} model defined'.format(MODEL))
        exit(0)    
    model.to(device)

    # training
    if not PRED_DIR:
        # dataset preparation
        image_transform = transforms.Compose([
            # transforms.RandomCrop(IMG_HEIGHT),
            transforms.ColorJitter(),
            transforms.ToTensor()
            ])
        # target_transform = transforms.Compose([
        #     transforms.RandomCrop(IMG_HEIGHT)
        #     ])
          
        train_dataset = CustomImageDataset('train', 
                                        DATASET_DIR, 
                                        CSV_FILE, 
                                        image_transform=image_transform)#, 
                                        # target_transform=target_transform)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                        BATCH_SIZE,
                                                        pin_memory=True,
                                                        shuffle=True,
                                                        num_workers=NUM_WORKERS)
    
        val_dataset = CustomImageDataset('val', 
                                         DATASET_DIR, 
                                         CSV_FILE, 
                                         image_transform=image_transform)#, 
                                        #  target_transform=target_transform)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, 
                                                      BATCH_SIZE,
                                                      pin_memory=True, 
                                                      shuffle=True,
                                                      num_workers=NUM_WORKERS)
        
        
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        start_epoch = 0
        train_loss = []
        val_loss = []
        best_miou = 0
        iou = 0

        # check if is it resume training
        if RESUME:
            print("Continue the training of experiment: '{}'".format(EXP_NAME))
            try:
                os.remove(os.path.join(EXP_NAME, 'weights', '.DS_Store'))
            except:
                pass
            chkpts_list = os.listdir(os.path.join(EXP_NAME, 'weights'))
            if len(chkpts_list) != 0:
                latest_epoch_saved = np.amax(np.array([int( x.split('.')[0].split('_')[-1] ) for x in chkpts_list]))
                checkpoint = torch.load(model_name % latest_epoch_saved)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                best_miou = checkpoint['mIoU']
                print('\tresuming from:', os.path.join(EXP_NAME, 'weights', '{}_epoch_%d.pth'.format(MODEL) % latest_epoch_saved),'\n')
                if start_epoch >= N_EPOCHS:
                    print('')
                    print("Training epoch is {}, but loaded epoch is {}.".format(N_EPOCHS, start_epoch))
                    print("Try again with higher epoch number.\n")
                    exit(0)


        print('Training..')
        for epoch in range(start_epoch, N_EPOCHS):
            #train
            model.train()
            epoch_start = time.time()
            train_epoch_loss = 0

            for i, (inputs, target) in enumerate(train_data_loader):
                inputs = inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                cntr = 0
                batch_window_loss = 0
                for h in range(0, inputs.shape[1], IMG_HEIGHT):
                    for w in range(0, inputs.shape[0], IMG_HEIGHT):
                        cntr += 1
                        # pil_inputs = transforms.ToPILImage()(inputs)
                        # input_window = transforms.ToTensor()(crop(inputs, h, w, IMG_HEIGHT, IMG_HEIGHT))
                        input_window = crop(inputs, h, w, IMG_HEIGHT, IMG_HEIGHT)
                        # target_window = transforms.ToTensor()(crop(target, h, w, IMG_HEIGHT, IMG_HEIGHT))
                        target_window = crop(target, h, w, IMG_HEIGHT, IMG_HEIGHT)
                        output_winow = model(input_window)
                        loss_window = criterion(output_winow, target_window)
                        batch_window_loss += loss_window.item()
                train_epoch_loss += batch_window_loss/cntr
            # for i, (inputs, target) in enumerate(train_data_loader):
            #     inputs = inputs.to(device, non_blocking=True)
            #     target = target.to(device, non_blocking=True)

            #     outputs = model(inputs)
            #     loss = criterion(outputs, target)
            #     train_epoch_loss += loss.item()
                if device.type == 'cpu':
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)
                # loss.backward()
                loss_window.backward()
                optimizer.step()
            train_loss.append( train_epoch_loss/ len(train_data_loader) )
            

            # validation
            model.eval()
            inter_meter = AverageMeter()
            union_meter = AverageMeter()
            val_epoch_loss = 0
            with torch.no_grad():
                for i, (inputs, target) in enumerate(val_data_loader):
                    inputs = inputs.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    cntr = 0
                    batch_window_loss = 0
                    for h in range(0, inputs.shape[1], IMG_HEIGHT):
                        for w in range(0, inputs.shape[0], IMG_HEIGHT):
                            cntr += 1
                            # pil_inputs = transforms.ToPILImage()(inputs)
                            # input_window = transforms.ToTensor()(crop(inputs, h, w, IMG_HEIGHT, IMG_HEIGHT))
                            # target_window = transforms.ToTensor()(crop(target, h, w, IMG_HEIGHT, IMG_HEIGHT))
                            input_window = crop(inputs, h, w, IMG_HEIGHT, IMG_HEIGHT)
                            target_window = crop(target, h, w, IMG_HEIGHT, IMG_HEIGHT)
                        
                            output_winow = model(input_window)
                            loss_window = criterion(output_winow, target_window)
                            batch_window_loss += loss_window.item()

                            pred_window = torch.argmax(output_winow, dim=1).data.cpu().numpy().squeeze().astype(np.uint8)
                            inter, union = inter_and_union(pred_window, target_window.cpu(), N_CLASSES)
                            inter_meter.update(inter)
                            union_meter.update(union)

                    val_epoch_loss += batch_window_loss/cntr
                    # outputs = model(inputs)
                    # loss = criterion(outputs, target)
                    # val_epoch_loss += loss.item()
                    # pred = torch.argmax(outputs, dim=1).data.cpu().numpy().squeeze().astype(np.uint8)
                    # inter, union = inter_and_union(pred, target.cpu(), N_CLASSES)
                    # inter_meter.update(inter)
                    # union_meter.update(union)
                    
                iou = inter_meter.sum / (union_meter.sum + 1e-10)
                val_loss.append( val_epoch_loss/len(val_data_loader) )
                
            scheduler.step()
            epoch_end = time.time()
            
            print('-- Epoch {} -- train_loss: {:.4f}, val_loss: {:.4f}   -- miou: {:.4f}  ({:.4f} mins)'.format(epoch, 
                                                                                                                train_loss[epoch - start_epoch], 
                                                                                                                val_loss[epoch - start_epoch],
                                                                                                                iou.mean(),
                                                                                                                (epoch_end - epoch_start) / 60))
        

            writer.add_scalars('Loss', {'train loss':train_loss[epoch - start_epoch], 
                                        'val loss': val_loss[epoch - start_epoch],
                                        'mIoU': iou.mean()}, epoch)                       


            if epoch % SAVING_STEP == (SAVING_STEP - 1):
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mIoU': iou.mean()
                }, model_name % (epoch + 1))

            if best_miou <= iou.mean(): # and best_val_loss >= val_loss[epoch - start_epoch]:
                best_miou = iou.mean()
                print("\t\t\t\t\t\t\tBest mIoU model: {}: {:.4f} mIoU".format(model_name % 0, best_miou))
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mIoU': best_miou
                }, model_name % 0)
      
        writer.close()
    
    # inference on data   
    elif PRED_DIR:
        print('Inference..')
        if SAMPLE_RESIZE is not None:
            s = 'inference_{}'.format(SAMPLE_RESIZE)
        else:
            s = 'inference'
        if not os.path.isdir(os.path.join(EXP_NAME, s)):
            os.mkdir(os.path.join(EXP_NAME, s ))
            print("Prediction result will be saved in '{}'\n".format(os.path.join(EXP_NAME, s)))
        checkpoint = torch.load(model_name %  0)
        f = open(os.path.join(EXP_NAME, s, 'inference result.txt'), 'w+')
        print("\t(mIoU: {:.4f} model loaded: '{}')\n\n".format(checkpoint['mIoU'], model_name % 0))
        f.writelines("\nmIoU: {:.4f} model loaded: '{}'\n\n".format(checkpoint['mIoU'], model_name % 0))
        
        
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # Color dictionary
        df = pd.read_csv(CSV_FILE)
        rgb_df = df[['r', 'g', 'b']]
        color_dict = [tuple(x) for x in rgb_df.values.astype(np.int) ]
        f.writelines('\nInference file name, size, fps\n')
        try:
            os.remove(os.path.join(PRED_DIR, '.DS_Store'))
        except:
            pass
        
        # inference on all '*.png', '*.jpg' and '*.mp4' files
        for file_name in sorted(os.listdir(PRED_DIR)):
            # except directory 
            if not os.path.isfile(os.path.join(PRED_DIR, file_name)):
                continue
            if file_name[0] == '.':
                continue

            # inference on '*.mp4' video files
            elif file_name.split('.')[1] == 'mp4':
                file_path = os.path.join(PRED_DIR, file_name)
                out_file_path = os.path.join(EXP_NAME, s, file_name)
                cap = cv2.VideoCapture(file_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                out_video = cv2.VideoWriter(out_file_path.split('.')[0]+'_masked.mp4', 
                                            cv2.VideoWriter_fourcc(*'mp4v'), 
                                            fps, 
                                            (frame_width, frame_height) )
                mask_video = cv2.VideoWriter(out_file_path.split('.')[0]+'_mask_only.mp4', 
                                            cv2.VideoWriter_fourcc(*'mp4v'), 
                                            fps, 
                                            (frame_width, frame_height) )
                duration = 0
                frm_cntr = 0

                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)


                for frame in frames:
                    frm_cntr += 1
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    output = img
                    if SAMPLE_RESIZE:
                        img = img.resize((SAMPLE_RESIZE, SAMPLE_RESIZE))
                    
                    start_time = time.time()

                    # image_tensor = transforms.ToTensor()(output)
                    image_tensor = transforms.ToTensor()(img)

                    # mask = Image.new("RGB", output.size)
                    mask = Image.new("RGB", img.size)

                    for h in range(0, image_tensor.shape[1], IMG_HEIGHT):
                        for w in range(0, image_tensor.shape[2], IMG_HEIGHT):
                            window = transforms.ToTensor()(crop(img, h, w, IMG_HEIGHT, IMG_HEIGHT))
                            window_pred = model(window.unsqueeze(0).to(device, non_blocking=True))
                            window_pred = torch.argmax(window_pred, dim=1).cpu().squeeze()
                            window_pred = imshow(window_pred, num_classes=N_CLASSES, colors=color_dict, inf_class_idx=INF_CLASS_IDX, mode='pred')
                            # window_pred = window_pred.resize( output.size , Image.NEAREST)
                            Image.Image.paste(mask, window_pred, (w,h))
                    
                    mask = mask.resize(output.size, Image.NEAREST)
                    output = Image.composite(mask, output , mask.convert('L'))
                    out_video.write(np.array(output)[:, :, :: -1] )
                    mask_video.write(np.array(mask)[:, :, :: -1] )
                    end_time = time.time()
                    duration += end_time - start_time
                    print("\t\tvideo frame segmentation: {}/{}".format(frm_cntr, n_frames))

                cap.release()
                out_video.release()
                mask_video.release()
                str = '{} : size= {} (model input size: {}), original fps: {:.4f}, model fps: {:.4f}'.format(file_name, 
                                                                                                             (frame_width, frame_height),
                                                                                                              IMG_HEIGHT, 
                                                                                                              fps,
                                                                                                              n_frames / duration * 1.0 )
                print(str)
                f.writelines('\n\t' + str)

            # inference on '*.png' '*.jpg' image files
            elif file_name.split('.')[1] == 'png' or file_name.split('.')[1] == 'jpg':
                file_path = os.path.join(PRED_DIR, file_name)
                out_file_path = os.path.join(EXP_NAME, s, file_name)
    
                img = Image.open(file_path).convert('RGB')
                start_time = time.time()
                
                blend_output = img
                masked_output = img

                if SAMPLE_RESIZE:
                    img = img.resize((SAMPLE_RESIZE, SAMPLE_RESIZE))
                image_tensor = transforms.ToTensor()(img)
                mask = Image.new("RGB", img.size)
                    
                # sliding window                 
                for h in range(0, image_tensor.shape[1], IMG_HEIGHT):
                    for w in range(0, image_tensor.shape[2], IMG_HEIGHT):
                        window = transforms.ToTensor()(crop(img, h, w, IMG_HEIGHT, IMG_HEIGHT))
                        window_pred = model(window.unsqueeze(0).to(device, non_blocking=True))
                        window_pred = torch.argmax(window_pred, dim=1).cpu().squeeze()
                        window_pred = imshow(window_pred, num_classes=N_CLASSES, colors=color_dict, inf_class_idx=INF_CLASS_IDX, mode='pred')
                        # window_pred = window_pred.resize( mask.size , Image.NEAREST)
                        Image.Image.paste(mask, window_pred, (w,h))
                mask = mask.resize(blend_output.size, Image.NEAREST)
                blend_output = Image.composite(mask, blend_output , mask.convert('L'))
                masked_output = mask
                end_time = time.time()
                blend_output.save(out_file_path.split('.')[0]+'_blend_slidingWindow.png')
                masked_output.save(out_file_path.split('.')[0]+'_mask_slidingWindow.png')
                str = '{}: size={} (model input size: {}), fps:{:.4f}'.format(file_name, 
                                                                                  img.size, 
                                                                                  IMG_HEIGHT,
                                                                                  1/(end_time-start_time))
                print(str)
                f.writelines('\n\t' + str)

            # other files are not compatible 
            else:
                print("Your file: ", file_name ,"\n\tChoose '.png','.jpg' image file or '.mp4' video file. \n")
                continue

        f.close()


if __name__ == "__main__":
    main()
    print('')
