'''
click_data = open('/Users/sun/Documents/click model dataset and baselines/srr dataset/click_data', 'r')
idx = 0
with open('click_data', 'a') as f:
    for line in click_data:
        line = line.strip().split('\t')
        sess = line[0]
        query = line[1]
        index = line[2]
        docs = line[3]
        clicks = line[4]

        idx += 1
        if idx <= 100:
            f.write(sess + '\t' + query + '\t' + index + '\t' + docs + '\t' + clicks)
            f.write('\n')
'''



import numpy as np
# query_sni2fea = np.load('/Volumes/Seagate Expansion Drive/博士数据集备份/click_model/data/query_snippet2feature.npy', allow_pickle=True).item()
# query_title2id = np.load('/Volumes/Seagate Expansion Drive/博士数据集备份/click_model/data/query_title2id.npy', allow_pickle=True).item()
# img_fea = np.load('/Volumes/Seagate Expansion Drive/博士数据集备份/click_model/data/img_feature_160.npy', allow_pickle=True).item()
# sess:str
# all_sess_adj = np.load('/Volumes/Seagate Expansion Drive/博士数据集备份/click_model/npy1/id_bert_add_title/sess_adj.npy', allow_pickle=True).item()
# all_sess_img_id = np.load('/Volumes/Seagate Expansion Drive/博士数据集备份/click_model/npy1/id_bert_add_title/sess_img_id.npy', allow_pickle=True).item()
# all_sess_show_idx = np.load('/Volumes/Seagate Expansion Drive/博士数据集备份/click_model/npy1/id_bert_add_title/sess_show_idx.npy', allow_pickle=True).item()
new_query_sni2fea, new_query_title2id, new_img_fea, new_all_sess_adj, new_all_sess_img_id, new_all_sess_show_idx = {},{},{},{},{},{}

click_data = open('click_data', 'r')
for line in click_data:
    line = line.strip().split('\t')
    sess = line[0]
    query = int(line[1])
    index = line[2]
    docs = line[3]
    clicks = line[4]
    if clicks == '0 0 0 0 0 0 0 0 0 0':
        continue
    # new_query_sni2fea[query] = query_sni2fea[query]
    # new_query_title2id[query] = query_title2id[query]
    new_all_sess_adj[sess] = all_sess_adj[sess]
    new_all_sess_img_id[sess] = all_sess_img_id[sess]
    new_all_sess_show_idx[sess] = all_sess_show_idx[sess]
# np.save('pretrained_files/new_query_sni2fea', new_query_sni2fea)
# np.save('pretrained_files/new_query_title2id', new_query_title2id)
np.save('pretrained_files/new_all_sess_adj', new_all_sess_adj)
np.save('pretrained_files/new_all_sess_img_id', new_all_sess_img_id)
np.save('pretrained_files/new_all_sess_show_idx', new_all_sess_show_idx)
# print(len(new_query_sni2fea))
# print(len(new_query_title2id))
print(len(new_all_sess_adj))
print(len(new_all_sess_img_id))
print(len(new_all_sess_show_idx))
