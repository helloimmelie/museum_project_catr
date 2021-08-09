#import system module(?)
import os
from configuration import Config 
import torch
from PIL import Image
#import torch modules
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .tokenization_kobert import KoBertTokenizer
from .utils import nested_tensor_from_tensor_list

class CustomDataset(Dataset):

    def __init__(self, data_path, transform):
    

        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        config = Config()
        
        self.dir_element = os.listdir(data_path) 
        self.transform = transform
        self.data_path = data_path
        self.start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
        self.max_length = config.max_position_embeddings

    def __len__(self):
        return len(self.dir_element)

    def __getitem__(self, idx):
        
        img_element = self.dir_element[idx]

        if ".jpg" in img_element:
            try:
                image = Image.open(os.path.join(self.data_path, img_element))
                image = self.transform(image)
                image = nested_tensor_from_tensor_list(image.unsqueeze(0))
                caption = torch.zeros(self.max_length, dtype=torch.long)
                cap_mask = torch.ones(self.max_length, dtype=torch.bool)
                caption[0] = self.start_token
                cap_mask[0] = False

            except:
                return None
        else:
            return None

        return image.tensors.squeeze(0), caption, cap_mask, self.dir_element[idx]

    