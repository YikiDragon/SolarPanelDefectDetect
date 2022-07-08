import pathlib
import random
import xlwt

def get_dataset(root_dir, shuffle=False):
    """
    读数据集中各图片路径与标签
    输入：root_dir 根目录文件夹
    输出：all_image_paths 各图片的路径
         all_image_labels 各图片的标签
    """
    data_root = pathlib.Path(root_dir)  # 设置数据根目录
    # for item in data_root.iterdir():
    #     print(item)  # 遍历根目录下的项目
    all_image_paths = list(data_root.glob('*/*'))  # 生成图片路径列表
    all_image_paths = [str(path) for path in all_image_paths]  # 转为字符串列表
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())  # 获取标签
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    label_to_index["damaged"] = 0  # 损坏的标签为0，完好的标签为1
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]  # 根据图片所在文件夹名划分标签
                        for path in all_image_paths]
    data_list = []
    for i in range(len(all_image_paths)):
        data_list.append([all_image_paths[i], all_image_labels[i]])
    if shuffle:
        random.shuffle(data_list)
    return data_list

if __name__ == '__main__':
    wb = xlwt.Workbook(encoding="UTF-8")    # 新工作簿
    ws_1 = wb.add_sheet("sheet")          # 新sheet
    ws_1.write(0, 1, "图片编号")
    ws_1.write(0, 2, "标签（0：损坏 1：完好）")
    data_list = get_dataset("./dataset/train")
    for i, data in enumerate(data_list):
        image_names = pathlib.Path(data[0]).name[:-4]
        image_label = data[1]
        ws_1.write(i+1, 1, image_names)
        ws_1.write(i+1, 2, image_label)
    wb.save("LabelList.xls")
