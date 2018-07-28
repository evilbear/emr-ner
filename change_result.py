#encoding:utf-8
import chardet, os, codecs, json

save_path = os.path.dirname(os.path.abspath(__file__)) + '/result.txt'
save_data = codecs.open(save_path, 'w')
file_path1 = os.path.dirname(os.path.abspath(__file__)) + '/result/ner/doc2vec_att/ner_result_999.txt'
data1 = codecs.open(file_path1, 'r')
file_path2 = os.path.dirname(os.path.abspath(__file__)) + '/data/test.txt'
data2 = codecs.open(file_path2, 'r')
list1, list2 = [], []
list_temp1, list_temp2 = [], []
for line in data1:
    line = line.strip('\n')
    if (len(line) != 0):
        list_temp1.append(line.split(' ')[-1])
    else:
        list1.append(list_temp1)
        list_temp1=[]
data1.close()
for line in data2:
    line = line.strip('\n')
    if (len(line) != 0):
        word = line.split(' ')[0]
        if word == '':
            list_temp2.append(' ')
        else:
            list_temp2.append(word)
    else:
        list2.append(list_temp2)
        list_temp2=[]
data2.close()

def fun_min(AS, SYM, ISYM, DRUG, SUR):
    min_list = [99999, 99999, 99999, 99999, 99999]
    if AS:
        min_list[0]=AS[0]
    if SYM:
        min_list[1]=SYM[0]
    if ISYM:
        min_list[2]=ISYM[0]
    if DRUG:
        min_list[3]=DRUG[0]
    if SUR:
        min_list[4]=SUR[0]
    return min(min_list)

def fun_judge(seq, step, listlist, start, tag1, tag2):
    if start == listlist[0]:
        end = start + 1
        while(seq[end] == tag1):
            end += 1
        words = "".join(list2[step][start:end])
        if ';' not in words:
            save_data.write(words+'\t'+str(start)+'\t'+str(end)+'\t'+tag2+';')
        del listlist[0]
        return listlist
    else:
        return listlist

for step, seq in enumerate(list1):
    save_data.write(str(step+1)+',')

    AS = [i for i in range(len(seq)) if seq[i]=='B-AS']
    SYM = [i for i in range(len(seq)) if seq[i]=='B-SYM']
    ISYM = [i for i in range(len(seq)) if seq[i]=='B-ISYM']
    DRUG = [i for i in range(len(seq)) if seq[i]=='B-DRUG']
    SUR = [i for i in range(len(seq)) if seq[i]=='B-SUR']

    while(AS or SYM or ISYM or DRUG or SUR):
        start = fun_min(AS, SYM, ISYM, DRUG, SUR)
        if AS:
            AS = fun_judge(seq, step, AS, start, 'I-AS', '解剖部位')
        if SYM:
            SYM = fun_judge(seq, step, SYM, start, 'I-SYM', '症状描述')
        if ISYM:
            ISYM = fun_judge(seq, step, ISYM, start, 'I-ISYM', '独立症状')
        if DRUG:
            DRUG = fun_judge(seq, step, DRUG, start, 'I-DRUG', '药物')
        if SUR:
            SUR = fun_judge(seq, step, SUR, start, 'I-SUR', '手术')
    if step != (len(list1)-1):
        save_data.write('\n')
save_data.close()

path1 = os.path.dirname(os.path.abspath(__file__)) + '/result.txt'
data1 = codecs.open(path1, 'r')
path2 = os.path.dirname(os.path.abspath(__file__)) + '/result.json'
data2 = codecs.open(path2, 'w')
result_dict = {}
for line in data1:
    line = line.strip('\n').split(',')
    result_dict[int(line[0])] = line[1]
json.dump(result_dict, data2, ensure_ascii =False)
data1.close()
data2.close()