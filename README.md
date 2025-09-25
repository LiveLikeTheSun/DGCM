# A Content- and Context-Aware Click Model based on Dynamic Graph Neural Networks (DGCM))

# Dataset Format
```text
378466	5756	0 4 2 1 5 6 3 7 8 9	27106 27107 52257 27108 52259 52260 52258 52261 27115 52262	1 0 0 0 0 0 0 0 0 0	
```
1. 378466：session_id
2. 5756:query_id
3. 0 4 2 1 5 6 3 7 8 9:results index
4. 27106 27107 52257 27108 52259 52260 52258 52261 27115 52262: 10 documents_id
5. 1 0 0 0 0 0 0 0 0 0:clicks

# Before training model
##1. data preparation
   The folder 'data' contains dataset during the training, validation and testing of the model.
##2. file preparation saved in the folder 'pretrained files'
   ###(1) pretrained embedding：img_feature.npy and query_snippet2fea.npy mean the embedding obtained by pretrained model such as ResNet and BERT.
   ###(2) all_sess_adj.npy represents the adjacency matrix corresponding to the session
       all_sess_show_idx.npy indicates the order of the documents corresponding to the session, as the order of the documents returned by the same query may be different
       query_title2id.npy means the title_id of document corresponding to each query
       all_sess_img_id.npy represents the image_id that exists in each session
##3. The folder 'models' stores the trained model DGCM

# train model
```text
python run.py --train
```

# test model
```text
python run.py --test
```


