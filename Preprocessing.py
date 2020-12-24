"""
预处理部分2：
从原始数据中提取名称同时删除停用词，再写入文件中
"""

import os

import jieba
import jieba.posseg as ps

class_list = {'财经': 'Economics', '房产': 'House', '股票': 'Stock', '家居': 'Household', '教育': 'Education', '科技': 'Technology',
              '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}
with open('stop_words_ch.txt', 'r') as file:
    stop_words = [i.strip() for i in file.readlines()]
with open('stop_sign.txt', 'r', encoding='utf-8') as file:
    stop_sign = [i.strip() for i in file.readlines()]


def isStopWord(word_str: str):
    if word_str in stop_words:
        return True
    for i in word_str:
        if i in stop_sign:
            return True
    return False


def creatTexts():
    for class_name, class_name_en in class_list.items():
        # 生成文件夹目录
        if not os.path.exists('data_test/' + class_name_en):
            os.mkdir('data_test/' + class_name_en)
        if not os.path.exists('data_train/' + class_name_en):
            os.mkdir('data_train/' + class_name_en)
        for i in range(5000):
            print(class_name + ':' + str(i))
            # 生成测试集
            string_to_write = ''
            with open('source_data_test/' + class_name_en + '/' + str(i) + '.txt', 'r', encoding='utf-8') as f:
                lines = f.read()
                # print(lines)
                words = ps.cut(lines, use_paddle=True)
                for word, flag in words:
                    # print('%s %s' % (word, flag))
                    # 若为名词且不在停用词表中，则加入写入串
                    if flag == 'n' and not isStopWord(word):
                        string_to_write += (word + ' ')
            with open('data_test/' + class_name_en + '/' + str(i) + '.txt', 'w', encoding='utf-8') as f:
                f.write(string_to_write)
            # 生成训练集
            string_to_write = ''
            with open('source_data_train/' + class_name_en + '/' + str(i) + '.txt', 'r', encoding='utf-8') as f:
                lines = f.read()
                # print(lines)
                words = ps.cut(lines, use_paddle=True)
                for word, flag in words:
                    # print('%s %s' % (word, flag))
                    # 若为名词且不在停用词表中，则加入写入串
                    if flag == 'n' and not isStopWord(word):
                        string_to_write += (word + ' ')
            with open('data_train/' + class_name_en + '/' + str(i) + '.txt', 'w', encoding='utf-8') as f:
                f.write(string_to_write)


if __name__ == '__main__':
    jieba.enable_paddle()  # 启动paddle模式
    creatTexts()
