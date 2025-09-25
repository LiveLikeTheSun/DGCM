import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ClickDataset(Dataset):
    def __init__(self, args, data_path):
        self.args = args
        self.data = self._load_data(data_path)
        
    def _load_data(self, path):
        click_data = open(path, 'r')
        input_data = []
        for line in click_data:
            line = line.strip()
            if not line:
                continue
            ids = []
            session = line.split()[0]
            query = line.split()[1]
            result_idx = line.split()[2:12]
            url_idx = [int(_) for _ in line.split()[12:22]]
            clicks = [0] + [int(_) for _ in line.split()[22:32]]
            
            if clicks == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                continue
            for result in result_idx:
                query_idx = query + '_' + result
                #id = ids2id[query_idx]
                ids.append(query_idx)#(id)
                
            input_data.append({
                'session': session,
                'query': int(query),
                'result_id': ids,
                'url_ids': url_idx,
                'click': clicks
            })
        return input_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
