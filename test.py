import torch
from transformers import BertTokenizer
from datasets.tokenization_kobert import KoBertTokenizer
from PIL import Image
import argparse
import json
from models import caption
from datasets import coco, utils
from configuration import Config
import os

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


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask).to(device)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption

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
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model'])
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

#이미지 디렉토리 단위로 open & caption 

dir_element = os.listdir(args.path) #이미지 폴더 내에 있는 이미지 파일명들을 뽑아주는 명령어 : os.listdir(이미지 파일 경로)
json_list = list() 


with open(args.json_file_name+'.json', 'w', encoding = "utf-8") as make_file: 

    for element in dir_element:    
       
        if '.jpg' in element:
            try:
                image = Image.open(os.path.join(args.path,element))
                image = coco.val_transform(image)
                image = image.unsqueeze(0)


                caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)


                output = evaluate()
                result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
                json_list.append({"file_name":element, "caption": result.capitalize()})
                print(result.capitalize())
            except:
                pass
    json.dump({args.json_file_name:json_list},make_file,ensure_ascii=False,indent="\t") 