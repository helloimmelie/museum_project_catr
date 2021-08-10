import torch
from transformers import BertTokenizer
from datasets.tokenization_kobert import KoBertTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets.CustomDataset import CustomDataset
from PIL import Image
import argparse
import json
from models import caption
from datasets import coco, utils
from configuration import Config

from models import utils
import os
import tqdm

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, default='./bnj' ,help='path to image') #caption을 뽑고자 하는 image folder의 경로 
parser.add_argument('--v', type=str, help='version', default='v4')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default='./checkpoint3.pth')
parser.add_argument('--json_file_name', type=str, help='json file name', default="bnj")
args = parser.parse_args()
image_path = args.path
version = args.v
checkpoint_path = args.checkpoint


if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)




json_list = list() 





    #for i in range(config.max_position_embeddings - 1):
        #predictions = model(image, caption, cap_mask)
        #predictions = predictions[:, i, :]
        #predicted_id = torch.argmax(predictions, axis=-1)

        #if predicted_id[0] == 102:
            #return caption

        #caption[:, i+1] = predicted_id[0]
        #cap_mask[:, i+1] = False

    #return caption

config = Config()

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
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
model.to(device)

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)


predict_Dataset = CustomDataset(data_path = args.path, transform=coco.val_transform)
sampler_val = torch.utils.data.SequentialSampler(predict_Dataset)
predict_dataloader = DataLoader(dataset=predict_Dataset, batch_size=8, sampler = sampler_val, collate_fn = collate_fn, drop_last=False, shuffle=False, num_workers=2)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total = len(data_loader)
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)

    with tqdm.tqdm(total=total) as pbar:
        
        with open(args.json_file_name+'.json', 'w', encoding = "utf-8") as make_file: 
            for images, caps, cap_masks, file_name in data_loader:
                samples = images.to(device)

        

                caps = caps.to(device)
                cap_masks = cap_masks.to(device)

                file_name = file_name


                for i in range(config.max_position_embeddings-1):
                    predictions = model(samples, caps, cap_masks).to(device)
                    predictions = predictions[:, i, :]
                    predicted_id = torch.argmax(predictions, axis = -1)
                
                    if predicted_id[0] == 102:
                        return caps
                
                    caps[:, i+1] = predicted_id[0]
                    cap_masks[:, i+1] = False

                result = tokenizer.decode(caps[0].tolist(), skip_special_tokens=True)
                json_list.append({"file_name":file_name, "caption": result.capitalize()})
                print(result.capitalize())
            
            json.dump({args.json_file_name:json_list},make_file,ensure_ascii=False,indent="\t")

if __name__=='__main__':

    evaluate(model, predict_dataloader, device)
        