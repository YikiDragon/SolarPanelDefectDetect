from image_utils import correct, segment
import cv2
import pathlib

if __name__ == '__main__':
    all = True
    save_dir = "./dataset/all/"
    data_root = pathlib.Path('./photos')
    all_image_names = sorted(
        item.name for item in data_root.glob('*.JPG'))  # 获取文件名
    if not all:
        image = all_image_names[0]
        print("image name: " + image)
        img_src = cv2.imread("photos/" + image)
        image_corrected = correct(img_src, debug=False)
        seg = segment(image_corrected, seg_method=4, debug=False)
        for seg_ele in seg:
            savename = image[0:-4]+"_"+str(seg_ele[2][0])+"row"+str(seg_ele[2][1])+"col.jpg"
            cv2.imwrite(save_dir+savename, seg_ele[1])
    elif all:
        for image in all_image_names:
            print("image name: " + image)
            img_src = cv2.imread("photos/" + image)
            image_corrected = correct(img_src, debug=False)
            seg = segment(image_corrected, seg_method=4, debug=False)
            for seg_ele in seg:
                savename = image[0:-4] + "_" + str(seg_ele[2][0]) + "row" + str(seg_ele[2][1]) + "col.jpg"
                cv2.imwrite(save_dir + savename, seg_ele[1])