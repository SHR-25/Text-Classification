"""
预处理部分1：
利用清华语料库，处理10类的新闻数据
从清华数据源文件中提取需要的5万数据
"""
import os
import shutil

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

for class_name, class_name_en in class_list.items():
    dir_path = 'D:/下载/THUCNews/THUCNews/' + class_name
    file_list = os.listdir(dir_path)
    print(class_name + ':' + str(len(file_list)))

    if not os.path.exists('source_data_train/' + class_name_en):
        os.mkdir('source_data_train/' + class_name_en)
    for i in range(5000):
        print(i)
        shutil.copy(dir_path + '/' + file_list[i], 'source_data_train/' + class_name_en + '/' + str(i) + '.txt')

    if not os.path.exists('source_data_test/' + class_name_en):
        os.mkdir('source_data_test/' + class_name_en)
    for i in range(5000, 10000):
        print(i)
        shutil.copy(dir_path + '/' + file_list[i], 'source_data_test/' + class_name_en + '/' + str(i - 5000) + '.txt')

# class_name = '社会'
# class_name_en = 'Society'
# dir_path = 'D:/下载/THUCNews/THUCNews/' + class_name
# file_list = os.listdir(dir_path)
# print(class_name + ':' + str(len(file_list)))
#
# if not os.path.exists('source_data_train/' + class_name_en):
#     os.mkdir('source_data_train/' + class_name_en)
# for i in range(5000):
#     print(i)
#     shutil.copy(dir_path + '/' + file_list[i], 'source_data_train/' + class_name_en + '/' + str(i) + '.txt')
#
# if not os.path.exists('source_data_test/' + class_name_en):
#     os.mkdir('source_data_test/' + class_name_en)
# for i in range(5000, 10000):
#     print(i)
#     shutil.copy(dir_path + '/' + file_list[i], 'source_data_test/' + class_name_en + '/' + str(i - 5000) + '.txt')
