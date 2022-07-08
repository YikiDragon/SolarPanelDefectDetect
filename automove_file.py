import shutil
import xlrd

src_dir = "./dataset/all/"

perfect_dir = "dataset/train/perfect/"
damaged_dir = "dataset/train/damaged/"

data = xlrd.open_workbook('FinalLabel.xls')
table = data.sheet_by_name('Sheet1')
file_names = table.col_values(0)[1:]
labels = table.col_values(3)[1:]

for i, name in enumerate(file_names):
    label = labels[i]
    src_path = src_dir + name + '.jpg'
    if label == 0:  # 完好
        dst_path = perfect_dir + name + '.jpg'
        shutil.move(src_path, dst_path)
    elif label == 1:  # 损坏
        dst_path = damaged_dir + name + '.jpg'
        shutil.move(src_path, dst_path)
print('move completed!')
