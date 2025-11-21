# A Content- and Context-Aware Click Model based on Dynamic Graph Neural Networks (DGCM))

# Dataset Format
```text
378466	5756	0 4 2 1 5 6 3 7 8 9	27106 27107 52257 27108 52259 52260 52258 52261 27115 52262	1 0 0 0 0 0 0 0 0 0	3 3 2 1 2 2 1 2 1 2
```
1. 378466：session_id
2. 5756:query_id
3. 0 4 2 1 5 6 3 7 8 9:results index
4. 27106 27107 52257 27108 52259 52260 52258 52261 27115 52262: 10 documents_id
5. 1 0 0 0 0 0 0 0 0 0:clicks
6. 3 3 2 1 2 2 1 2 1 2:relevances

# Before training model
## 1. data preparation
   The folder 'data' contains dataset during the training, validation and testing of the model.  
   The input format and data are as shown in the data folder, and the data processing code will parse them into the format required for subsequent training and testing.  
   The output is the evaluation matrix.
## 2. file preparation saved in the folder 'pretrained files'
   ### (1) pretrained embedding：img_feature_160.npy and query_snippet2feature.npy store features extracted by pretrained models: image features are obtained using ResNet, and snippet text features are obtained using BERT.   
   Before training, you need unzip the "img_feature_160.npy.zip" first.
   ### (2) graph-related files
       #### all_sess_adj.npy represents the adjacency matrix corresponding to the session
       #### all_sess_show_idx.npy indicates the order of the documents corresponding to the session, as the order of the documents returned by the same query may be different
       #### query_title2id.npy means the title_id of document corresponding to each query
       #### all_sess_img_id.npy represents the image_id that exists in each session
## 3. The folder 'models' stores the trained model DGCM

# train model
```text
python run.py --train
```

# test model
```text
python run.py --test
```

After testing, the embedding of every document is stored in 'pretrained_files/query_output.npy' and we utilize learning to rank (LTR) algorithm to output the relevance score based on the representation learned by DGCM.
```text
java -jar RankLib-2.18.jar -train rank/ltr_train -test rank/ltr_test -ranker 6 -metric2t NDCG@k
```
k = 1,3,5

About the explanation of this command, we have placed a **`readme.txt`**  file inside the **`rank/`** folder



