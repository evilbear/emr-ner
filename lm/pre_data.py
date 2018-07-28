#encoding:utf-8
import os

#Replace words with dictionary subscripts.
def conversion(vocab_words, path, save_path):
    data = open(path, 'r')
    save_data = open(save_path, 'w')
    for line in data:
        line = line.strip().split(' ')
        new_line = []
        for word in line:
            if word.isdigit():
                new_line.append(vocab_words["$NUM$"])
            elif word in vocab_words:
                new_line.append(vocab_words[word])
            else:
                new_line.append(vocab_words["$UNK$"])
        for i in new_line[:-1]:
            save_data.write(str(i)+' ')
        save_data.write(str(new_line[-1])+'\n')
    data.close()
    save_data.close()

#Convert the word of conll format annotated data into sentence form.
#Because the corpus contains spaces, after split is '', so replaced with ' '.
def unite(vocab_words, path, save_path):
    data = open(path, 'r')
    save_data = open(save_path, 'w')
    for line in data:
        line = line.strip('\n')
        if len(line) != 0:
            word = line.split(' ')[0]
            if word in vocab_words:
                pass
            elif word.isdigit():
                word = "$NUM$"
            elif word == '':
                word = ' '
            else:
                word = "$UNK$"
            save_data.write(str(vocab_words[word]) + ' ')
        else:
            save_data.write('\n')
    data.close()
    save_data.close()

#Because NER once put in a sentence, and use '。' to split. so here’s the same treatment.
def split_file(file_name, period):
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/' + file_name + '/' + file_name + '.txt'
    temp_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/' + file_name + '/' + file_name + '_'
    data = open(path, 'r')
    for num, line in enumerate(data):
        save_path = temp_path + str(num)+'.txt'
        save_file = open(save_path, 'w')
        line = line.strip().split(' ')
        for i in line:
            save_file.write(str(i) + ' ')
            if int(i) == period:
                save_file.write('\n')
        save_file.close()
    data.close()

def con_data(vocab_words):
    #conversion lm data
    train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/train.txt'
    con_train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/lm_train.txt'
    conversion(vocab_words, train_path, con_train_path)

    dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/dev.txt'
    con_dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/lm_dev.txt'
    conversion(vocab_words, dev_path, con_dev_path)

    #conversion ner data
    ner_train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/train.txt'
    con_ner_train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/train/train.txt'
    unite(vocab_words, ner_train_path, con_ner_train_path)

    ner_dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/dev.txt'
    con_ner_dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/dev/dev.txt'
    unite(vocab_words, ner_dev_path, con_ner_dev_path)

    ner_test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/test.txt'
    con_ner_test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/test/test.txt'
    unite(vocab_words, ner_test_path, con_ner_test_path)

    #split ner data
    split_file('train', vocab_words['。'])
    split_file('dev', vocab_words['。'])
    split_file('test', vocab_words['。'])