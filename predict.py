#Import modules
import argparse
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from models import utils
from tqdm import tqdm
import numpy as np
#Import Pytorch
import torch
from torch.utils.data import Dataset, DataLoader
#Import custom modules
import os
from datasets.tokenization_kobert import KoBertTokenizer
import json
from datasets.CustomDataset import CustomDataset
from models import caption
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
from datasets import coco, utils
from configuration import Config


def collate_fn(batch): #collate_fn: dataloader에서 batch를 불러올 때 그 batch 데이터를 어떻게 전처리할 지를 정의 
    batch = list(filter(lambda x: x is not None, batch)) #해당 dataloader에서는 손상된 이미지 파일에 대해서 None으로 처리하기 때문에, None인 파일은 batch에서 제외
    return torch.utils.data.dataloader.default_collate(batch)

def predict(config, model, images, caps, cap_masks, device):
    for i in range(config.max_position_embeddings-1): 
        predictions = model(images, caps, cap_masks).to(device)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis = -1)
                
        if predicted_id[0] == 102:
            return caps
                
        caps[:, i+1] = predicted_id
        cap_masks[:, i+1] = False

    return caps

def visualize_and_save_json(json_file_name, caps, data_loader, images, file_name, tokenizer, fig_idx):
    
    font_path = 'C:/Windows/Fonts/gulim.ttc'
    font = font_manager.FontProperties(fname = font_path).get_name()
    rc('font', family=font)
    
    fig = plt.figure(figsize=(20,5))
    json_list = []
    
    with open(json_file_name+'.json', 'w', encoding='utf-8') as make_file:
        for i in range(len(caps)): 

            result = tokenizer.decode(caps[i].tolist(), skip_special_tokens=True)

            #save captioning result to json
            json_list.append({"file_name":file_name[i], "caption": result.capitalize()})
            #list에 저장된 caption을 json 파일에 작성 

            #visualize result through matplotlib
            ax = fig.add_subplot(data_loader.batch_size/2, 2 , i+1, xticks=[], yticks=[])
            imgs = images[i].cpu().numpy().transpose(1,2,0)
            ax.imshow(imgs)
            ax.set_title(result.capitalize())
        fig.tight_layout()
        plt.savefig(os.path.join('./figures',json_file_name+'_'+str(fig_idx)))
        fig_idx = fig_idx + 1
        
        json.dump({json_file_name:json_list},make_file,ensure_ascii=False,indent="\t")    

        
  

@torch.no_grad()
def evaluate(model, data_loader, args, device):
    
    model.eval()
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') #kobert tokenizer 호출 
    fig_idx = 0

    for images, caps, cap_masks, file_name in tqdm(data_loader):
        images = images.to(device)
        caps = caps.to(device)
        cap_masks = cap_masks.to(device)
        file_name = file_name

        #predict captions
        caps = predict(config, model, images, caps, cap_masks, device)

        #predict 된 file, caption들을 pair단위로 list에 저장 
        visualize_and_save_json(args.json_file_name, caps, data_loader, images, file_name, tokenizer, fig_idx)

           

if __name__=='__main__':


    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--path', type=str, default='./웹레트로' ,help='path to image') #caption을 뽑고자 하는 image folder의 경로 
    parser.add_argument('--v', type=str, help='version', default='v4') #본 모델은 torchhub에 있기 때문에 v4이상으로 지정해야 함
    parser.add_argument('--checkpoint', type=str, help='checkpoint path', default='./checkpoint3.pth') #한국어 COCO 데이터셋으로 훈련한 checkpoint load
    parser.add_argument('--json_file_name', type=str, help='json file name', default="webretro") #저장하고자 하는 json 파일 경로 
    args = parser.parse_args()

    image_path = args.path
    version = args.v
    checkpoint_path = args.checkpoint

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device='cpu'

    json_list = list() 
    config = Config()
    
    #load dataloader
    predict_Dataset = CustomDataset(data_path = args.path, transform=coco.val_transform)
    sampler_val = torch.utils.data.SequentialSampler(predict_Dataset)
    predict_dataloader = DataLoader(dataset=predict_Dataset, batch_size=config.batch_size, 
                                sampler = sampler_val, collate_fn = collate_fn, drop_last=False, 
                                shuffle=False, num_workers=config.num_workers)

    if version == 'v1':
        model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
    elif version == 'v2':
        model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
    elif version == 'v3':
        model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    else:
        print("Checking for checkpoint.")
        if checkpoint_path is None:
            raise NotImplementedError('No model to chose from!')
        else:
            if not os.path.exists(checkpoint_path):
                raise NotImplementedError('Give valid checkpoint path')
            print("Found checkpoint! Loading!")
            model,_ = caption.build_model(config)
            print("Loading Checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])

    model.to(device)

    evaluate(model, predict_dataloader, args, device)
        