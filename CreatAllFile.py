"""
预处理部分3：
整合数据至一个文件all.txt，避免重复IO
"""
class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

if __name__ == '__main__':
    all_data_test = ''
    all_data_train = ''
    for class_name, class_name_en in class_list.items():
        string_to_write_test = ''
        string_to_write_train = ''
        for i in range(5000):
            print(class_name + ':' + str(i))
            with open('data_test/' + class_name_en + '/' + str(i) + '.txt', 'r', encoding='utf-8') as f:
                string_to_write_test += (f.read() + '\n')
            with open('data_train/' + class_name_en + '/' + str(i) + '.txt', 'r', encoding='utf-8') as f:
                string_to_write_train += (f.read() + '\n')
        with open('data_test/' + class_name_en + '/all.txt', 'w', encoding='utf-8') as f:
            f.write(string_to_write_test)
        with open('data_train/' + class_name_en + '/all.txt', 'w', encoding='utf-8') as f:
            f.write(string_to_write_train)
        all_data_test += string_to_write_test
        all_data_train += string_to_write_train
    with open('data_test/all.txt', 'w', encoding='utf-8') as f:
        f.write(all_data_test)
    with open('data_train/all.txt', 'w', encoding='utf-8') as f:
        f.write(all_data_train)
