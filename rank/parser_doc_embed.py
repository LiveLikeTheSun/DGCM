features = np.load('../pretrained_files/query_output.npy', allow_pickle=True).tolist()
# print(features)  # (85, 10, 45)
test = open('../data/new/test', 'r')
for line in test:
    line = line.strip().split('\t')
    clicks = line[4]
    if clicks == '0 0 0 0 0 0 0 0 0 0':
        continue
    query = int(line[1])  # 5756
    docs = [int(_) for _ in line[3].split()]  # [27106, 27107, 52257, 27108, 52259, 52260, 52258, 52261, 27115, 52262]
    relevances = [int(_) for _ in line[5].split()]  # [3, 2, 1, 2, 2, 1, 2, 1, 2]


    index, idx = 0, 0
    for rele in relevances:
        with open('ltr_test', 'a') as f:
            f.write(str(rele)+' ')
            f.write('qid:')
            f.write(str(query)+' ')
            for i in range(45):  # the output dimension of every document
                f.write(str(i+1) + ':' + str(features[index][idx][i]) + ' ')
            f.write('\n')
        idx += 1
    index += 1