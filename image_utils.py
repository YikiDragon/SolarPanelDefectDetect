import random
from scipy import signal
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pathlib


def correct(image: np.ndarray, hist_th=25, approx_eps=0.01, debug=False):
    '''
    太阳能板姿态校正
    :param image: 待校正图像
    :param hist_th: 直方图双峰判断阈值，设在直方图里两波峰之间，实验综合最佳值为25
    :param approx_eps:  四边形逼近系数初值，会动态调整
    :param debug:   调试开关，默认关闭
    :return: img_K 校正后图像
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
    hist = cv2.calcHist([gray], [0], None, [256], [0, 255])
    if debug:
        plt.title("ColorHist")
        plt.xlabel("grayscale")
        plt.ylabel("pixel number")
        plt.plot(hist)
        plt.savefig("./saved_plot/GrayscaleImageHistogram.pdf", dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
        plt.show()
    threshold = (hist_th + np.argmax(hist[hist_th:255]) - np.argmax(hist[0:hist_th])) / 3 + np.argmax(
        hist[0:hist_th])  # 以直方图中的双波峰中点为二值化阈值
    # threshold = (np.argmax(hist[0:hist_th]) + hist_th + np.argmax(hist[hist_th:255])) / 3 + np.argmax(
    #     hist[0:hist_th])  # 以直方图中的双波峰中点为二值化阈值
    ret, img_thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)  # 二值化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    # 目标框选阶段
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓，树形结构输出轮廓，存储轮廓点信息
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # 按轮廓面积由大到小排序
    peri = cv2.arcLength(cnts[0], True)  # 求最大轮廓的周长,首尾相连
    approx_num = 30  # 更改eps的最大次数
    approx = np.array([], dtype=np.float32)
    # 动态调整eps值，直到逼近出四边形
    for i in range(approx_num):
        approx = cv2.approxPolyDP(cnts[0], approx_eps * peri, True)  # 多边形逼近,输出近似轮廓的角点坐标
        if approx.shape[0] > 4:
            approx_eps += 0.01  # 增大eps
        elif approx.shape[0] < 4:
            approx_eps -= 0.01  # 减小eps
        else:  # 逼近出四边形
            break
    if approx.shape[0] != 4:
        # 无法逼近到合适的四边形
        print("\033[1;31mFaild: Correct can't approximate rectangle of SolarPanel\033[0m")
        return False
    if debug:
        plt.imshow(opening, cmap=plt.get_cmap('gray'))
        plt.savefig("./saved_plot/Binarization.pdf", dpi=300, format="pdf", bbox_inches='tight',
                    pad_inches=0.0)
        plt.show()
        temp = image.copy()
        res = cv2.drawContours(temp, [approx], 0, (255, 0, 0), 30)
        plt.imshow(res)
        plt.savefig("./saved_plot/ContourApproximation.pdf", dpi=300, format="pdf", bbox_inches='tight',
                    pad_inches=0.0)
        plt.show()
    # approx = np.concatenate([approx[:, :, 1], approx[:, :, 0]], axis=-1)        # x,y 坐标顺序调整成张量行列顺序
    approx = np.reshape(approx, [4, 2])
    try:  # 如果图像姿态比较正
        # center_point = np.mean(approx, axis=0)  # 轮廓的近似中心点
        center_point = np.array([[(np.max(approx[:, 0]) + np.min(approx[:, 0])) / 2,  # x
                                  (np.max(approx[:, 1]) + np.min(approx[:, 1])) / 2]], np.float32)  # y
        local_flag = approx < center_point
        corner_points = np.array([approx[np.where(local_flag[:, 0] & local_flag[:, 1])],  # 左上
                                  approx[np.where(local_flag[:, 0] & (~local_flag[:, 1]))],  # 左下
                                  approx[np.where((~local_flag[:, 0]) & (~local_flag[:, 1]))],  # 右下
                                  approx[np.where((~local_flag[:, 0]) & local_flag[:, 1])]],  # 右上
                                 dtype=np.float32)
        corner_points = np.reshape(corner_points, [4, 2])
        # 简单估计板子姿态
        width, height = 3500, 2500  # 宽>高
        # 宽 < 高
        if np.abs(corner_points[3, 0] - corner_points[0, 0]) < np.abs(corner_points[1, 1] - corner_points[0, 1]):
            width, height = 2500, 3500
    except:  # 采用边长策略判断
        corner_points = approx.astype(np.float32)
        # 简单估计板子姿态
        # 第一条边 < 第二条边
        if np.linalg.norm(corner_points[1] - corner_points[0]) < np.linalg.norm(corner_points[2] - corner_points[1]):
            width, height = 2500, 3500  # 以第一条边为宽，第二条边为高
            corner_points = np.float32([corner_points[1],
                                        corner_points[2],
                                        corner_points[3],
                                        corner_points[0]])
        else:
            width, height = 3500, 2500  # 以第一条边为宽，第二条边为高
            corner_points = np.float32([corner_points[1],
                                        corner_points[2],
                                        corner_points[3],
                                        corner_points[0]])
    # 定义对应的像素点坐标
    corner_points_dst = np.float32([[0, 0],  # 左上
                                    [0, height],  # 左下
                                    [width, height],  # 右下
                                    [width, 0]])  # 右上
    matrix_K = cv2.getPerspectiveTransform(corner_points, corner_points_dst)  # 计算转换矩阵
    img_K = cv2.warpPerspective(image, matrix_K, (width, height))  # 进行透视变换
    if debug:
        temp = image.copy()
        res = cv2.drawContours(temp, [approx], 0, (255, 0, 0), 30)
        plt.subplot(121)
        plt.xlabel("DrawContours")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(res)
        plt.subplot(122)
        plt.xlabel("Corrected")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_K)
        plt.savefig("./saved_plot/EffectBeforeAndAfterCorrection.pdf", dpi=300, format="pdf", bbox_inches='tight',
                    pad_inches=0.0)
        plt.show()
    return img_K


def segment(image_corrected: np.ndarray, trough_th=200, seg_method=0, debug=False):
    '''
    电池板自动化分割
    :param image_corrected:     校正后的电池板图
    :param trough_th:           波谷判别阈值，用于确定分割位置(弃用)
    :param seg_method:          分割方法，0-纯波谷分割，1-平均分割， 2-均值间隔探测波谷， 3-均值间隔+波谷， 4-FFT频谱分析
    :param debug:               调试开关
    :return: segmentations      列表，每个元素格式为[[起始点坐标，终止点坐标], 分割图]
    '''
    img = np.array([])
    # 缩小到1/5
    if image_corrected.shape[0:2] == (2500, 3500):
        img = cv2.resize(image_corrected, (700, 500))
    elif image_corrected.shape[0:2] == (3500, 2500):
        img = cv2.resize(image_corrected, (500, 700))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))
    # 自适应滤波
    # gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11,
                                  2)  # 自适应二值化,最后一个参数越大黑色部分越少
    thres = cv2.bitwise_not(thres)
    opening_row = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 2)))  # 行形状闭运算
    opening_col = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 60)))  # 列形状闭运算
    wave_col = np.sum(opening_col, axis=0) / opening_col.shape[0]
    wave_row = np.sum(opening_row, axis=1) / opening_row.shape[1]
    if debug:
        plt.imshow(thres, plt.cm.get_cmap('gray'))
        plt.savefig("./saved_plot/AdaptiveBinarization.pdf", dpi=300, format="pdf", bbox_inches='tight',
                    pad_inches=0.0)
        plt.show()
        plt.subplot(121)
        plt.xlabel("Opening_col")
        plt.imshow(opening_col, plt.cm.gray)
        plt.subplot(122)
        plt.xlabel("Opening_row")
        plt.imshow(opening_row, plt.cm.gray)
        plt.savefig("./saved_plot/HorizontalAndVerticalClosureOperation.pdf", dpi=300, format="pdf", bbox_inches='tight',
                    pad_inches=0.0)
        plt.show()
        plt.title("Row and column waveform")
        plt.xlabel("Row and column index")
        plt.ylabel("Average gray scale")
        plt.plot(wave_col)
        plt.plot(wave_row)
        plt.plot(trough_th)
        plt.legend(["wave_col", "wave_row"])
        plt.savefig("./saved_plot/RowAndColumnWaveform.pdf", dpi=300, format="pdf", bbox_inches='tight',
                    pad_inches=0.0)
        plt.show()
    # 以100为阈值，获取波谷序号
    # trough_row = np.where(wave_row < trough_th)[0]
    # trough_col = np.where(wave_col < trough_th)[0]
    trough_row = np.where(wave_row < 128 + np.min(wave_row)/2.0)[0]
    trough_col = np.where(wave_col < 128 + np.min(wave_col)/2.0)[0]
    def trough_clustering(trough, th=10):
        '''
        波谷数据聚合，过于接近的序号合并为1个序号, 以左右距离阈值为10进行聚合
        :param trough:  波谷序号数组
        :param th:      同类点最大距离
        :return: trough_clustered 聚合后的波谷序号数组
        '''
        #
        trough_clustered = np.array([], np.float32)
        node_list = np.array([trough[0]], np.float32)  # 单类点集, 第一个点先入
        for i in range(trough.shape[0] - 1):
            if np.abs(trough[i + 1] - trough[i]) < th:  # 满足距离阈值，视为一类
                node_list = np.append(node_list, trough[i + 1])  # 属于当前类
            else:  # 到下一类了
                trough_clustered = np.append(trough_clustered, np.mean(node_list))  # 当前类点集构建完毕，以均值代表该点集
                node_list = np.array([trough[i + 1]], np.float32)  # 新建单类点集, 第一个点先入
        trough_clustered = np.append(trough_clustered, np.mean(node_list))  # 最后一个点

        return trough_clustered

    trough_row = trough_clustering(trough_row)  # 行波谷点
    trough_col = trough_clustering(trough_col)  # 列波谷点
    if seg_method == 0:
        '''
        纯波谷法
        '''
        # 补充边缘线
        trough_row = np.concatenate([[0], trough_row * 5, [image_corrected.shape[0] - 1]]).astype(np.int32)
        trough_col = np.concatenate([[0], trough_col * 5, [image_corrected.shape[1] - 1]]).astype(np.int32)
        trough_row = trough_clustering(trough_row).astype(np.int32)  # 行波谷点
        trough_col = trough_clustering(trough_col).astype(np.int32)  # 列波谷点
        segmentations = []
        copy = np.array([])
        if debug:
            copy = image_corrected.copy()
        for i in range(trough_row.shape[0] - 1):
            for j in range(trough_col.shape[0] - 1):
                if debug:
                    copy = cv2.rectangle(copy, (trough_col[j], trough_row[i]), (trough_col[j + 1], trough_row[i + 1]),
                                         (0, 255, 0), 10)
                img_segment = image_corrected[trough_row[i]:trough_row[i + 1], trough_col[j]:trough_col[j + 1], :]
                segmentations.append(
                    [np.array([[trough_col[j], trough_row[i]], [trough_col[j + 1], trough_row[i + 1]]], np.int32),
                     img_segment])
        if debug:
            plt.title("Debug demo")
            plt.imshow(copy)
            plt.show()
        return segmentations
    elif seg_method == 1:
        '''
        均值间隔
        '''
        # 以波谷间的最小间隔为相邻分界线间隔
        interval_row = trough_clustering(
            (np.concatenate([trough_row, [5000]]) - np.concatenate([[-5000], trough_row]))[1:-1])
        interval_col = trough_clustering(
            (np.concatenate([trough_col, [5000]]) - np.concatenate([[-5000], trough_col]))[1:-1])
        if interval_row.shape[0] == 2:  # 聚合出两个值
            interval_row = np.min(interval_row)  # 取最小值
        elif interval_row.shape[0] > 2:  # 聚合出两个以上值
            interval_row = np.median(interval_row)  # 取中值
        if interval_col.shape[0] == 2:  # 聚合出两个值
            interval_col = np.min(interval_col)  # 取最小值
        elif interval_col.shape[0] > 2:  # 聚合出两个以上值
            interval_col = np.median(interval_col)  # 取中值
        row_num = img.shape[0] / interval_row
        col_num = img.shape[1] / interval_col
        # 确保行列数均是偶数
        if row_num.astype(np.int32) % 2 != 0:
            row_num = (row_num + 1.0).astype(np.int32)
        else:
            row_num = row_num.astype(np.int32)
        if col_num.astype(np.int32) % 2 != 0:
            col_num = (col_num + 1.0).astype(np.int32)
        else:
            col_num = col_num.astype(np.int32)
        # 开始分割
        height = int(image_corrected.shape[0] / row_num)  # 单块晶片的高度
        width = int(image_corrected.shape[1] / col_num)  # 单块晶片的宽度
        copy = np.array([])
        if debug:
            copy = image_corrected.copy()
        segmentations = []
        for i in np.arange(row_num):
            for j in np.arange(col_num):
                start_height = i * height
                start_width = j * width
                end_height = min(start_height + height, image_corrected.shape[0] - 1)
                end_width = min(start_width + width, image_corrected.shape[1] - 1)
                if debug:
                    copy = cv2.rectangle(copy, (start_width, start_height), (end_width, end_height),
                                         (0, 255, 0), 10)
                img_segment = image_corrected[start_height:end_height, start_width:end_width, :]
                segmentations.append(
                    [np.array([[start_width, start_height], [end_width, end_height]], np.int32), img_segment])
        if debug:
            plt.title("Debug demo")
            plt.imshow(copy)
            plt.show()
        return segmentations
    elif seg_method == 2:
        '''
        均值间隔波谷检测
        '''
        # 以波谷间的最小间隔为相邻分界线间隔
        interval_row = trough_clustering(
            (np.concatenate([trough_row, [5000]]) - np.concatenate([[-5000], trough_row]))[1:-1])
        interval_col = trough_clustering(
            (np.concatenate([trough_col, [5000]]) - np.concatenate([[-5000], trough_col]))[1:-1])
        if interval_row.shape[0] == 2:  # 聚合出两个值
            interval_row = np.min(interval_row)  # 取最小值
        elif interval_row.shape[0] > 2:  # 聚合出两个以上值
            interval_row = np.median(interval_row)  # 取中值
        if interval_col.shape[0] == 2:  # 聚合出两个值
            interval_col = np.min(interval_col)  # 取最小值
        elif interval_col.shape[0] > 2:  # 聚合出两个以上值
            interval_col = np.median(interval_col)  # 取中值
        range_detect_row = int((interval_row+1))   # 探测步长
        range_detect_col = int((interval_col+1))   # 探测步长
        trough_row = np.int32([])
        trough_col = np.int32([])
        # 获取波谷
        i_row = 0
        i_col = 0
        while True:
            if i_row + range_detect_row > wave_row.shape[0]-1:   # 超过就结束
                break
            end = min(i_row+range_detect_row, wave_row.shape[0]-1)      # 不得超过wave_row长度
            trough_row = np.append(trough_row, i_row + np.argmin(wave_row[i_row:end]))
            i_row += range_detect_row
        while True:
            if i_col + range_detect_col > wave_col.shape[0]-1:   # 超过就结束
                break
            end = min(i_col+range_detect_col, wave_col.shape[0]-1)      # 不得超过wave_row长度
            trough_col = np.append(trough_col, i_col + np.argmin(wave_col[i_col:end]))
            i_col += range_detect_col
        # 补充边缘线
        trough_row = np.concatenate([[0], trough_row * 5, [image_corrected.shape[0] - 1]]).astype(np.int32)
        trough_col = np.concatenate([[0], trough_col * 5, [image_corrected.shape[1] - 1]]).astype(np.int32)
        # 聚合数据
        trough_row = trough_clustering(trough_row).astype(np.int32)  # 行波谷点
        trough_col = trough_clustering(trough_col).astype(np.int32)  # 列波谷点
        segmentations = []
        copy = np.array([])
        if debug:
            copy = image_corrected.copy()
        for i in range(trough_row.shape[0] - 1):
            for j in range(trough_col.shape[0] - 1):
                if debug:
                    copy = cv2.rectangle(copy, (trough_col[j], trough_row[i]), (trough_col[j + 1], trough_row[i + 1]),
                                         (0, 255, 0), 10)
                img_segment = image_corrected[trough_row[i]:trough_row[i + 1], trough_col[j]:trough_col[j + 1], :]
                segmentations.append(
                    [np.array([[trough_col[j], trough_row[i]], [trough_col[j + 1], trough_row[i + 1]]], np.int32),
                     img_segment])
        if debug:
            plt.title("Debug demo")
            plt.imshow(copy)
            plt.show()
        return segmentations
    elif seg_method == 3:
        '''
        均值间隔波谷检测+均值间隔，用于弥补波谷漏检
        '''
        # 以波谷间的最小间隔为相邻分界线间隔
        interval_row = trough_clustering(
            (np.concatenate([trough_row, [5000]]) - np.concatenate([[-5000], trough_row]))[1:-1])
        interval_col = trough_clustering(
            (np.concatenate([trough_col, [5000]]) - np.concatenate([[-5000], trough_col]))[1:-1])
        if interval_row.shape[0] == 2:  # 聚合出两个值
            interval_row = np.min(interval_row)  # 取最小值
        elif interval_row.shape[0] > 2:  # 聚合出两个以上值
            interval_row = np.median(interval_row)  # 取中值
        if interval_col.shape[0] == 2:  # 聚合出两个值
            interval_col = np.min(interval_col)  # 取最小值
        elif interval_col.shape[0] > 2:  # 聚合出两个以上值
            interval_col = np.median(interval_col)  # 取中值
        row_num = img.shape[0] / interval_row
        col_num = img.shape[1] / interval_col
        # 确保行列数均是偶数
        if row_num.astype(np.int32) % 2 != 0:
            row_num = (row_num + 1.0).astype(np.int32)
        else:
            row_num = row_num.astype(np.int32)
        if col_num.astype(np.int32) % 2 != 0:
            col_num = (col_num + 1.0).astype(np.int32)
        else:
            col_num = col_num.astype(np.int32)
        ############# 间距均值方法 ##########
        height = int(image_corrected.shape[0] / row_num)  # 单块晶片的高度
        width = int(image_corrected.shape[1] / col_num)  # 单块晶片的宽度
        trough_row1 = np.int32([])
        trough_col1 = np.int32([])
        for i in range(int(row_num-1)):
            trough_row1 = np.append(trough_row1, (i+1) * height)
        for i in range(int(col_num-1)):
            trough_col1 = np.append(trough_col1, (i+1) * width)
        ############# 均值间隔探测波谷法 ################
        range_detect_row = int((interval_row+1))   # 探测步长
        range_detect_col = int((interval_col+1))   # 探测步长
        trough_row = np.int32([])
        trough_col = np.int32([])
        # 获取波谷
        i_row = 0
        i_col = 0
        while True:
            if i_row + range_detect_row > wave_row.shape[0]-1:   # 超过就结束
                break
            end = min(i_row+range_detect_row, wave_row.shape[0]-1)      # 不得超过wave_row长度
            trough_row = np.append(trough_row, i_row + np.argmin(wave_row[i_row:end]))
            i_row += range_detect_row
        while True:
            if i_col + range_detect_col > wave_col.shape[0]-1:   # 超过就结束
                break
            end = min(i_col+range_detect_col, wave_col.shape[0]-1)      # 不得超过wave_row长度
            trough_col = np.append(trough_col, i_col + np.argmin(wave_col[i_col:end]))
            i_col += range_detect_col

        # 补充边缘线
        trough_row = np.concatenate([[0], trough_row * 5, [image_corrected.shape[0] - 1]]).astype(np.int32)
        trough_col = np.concatenate([[0], trough_col * 5, [image_corrected.shape[1] - 1]]).astype(np.int32)
        # 连接两数据
        trough_row = np.sort(np.concatenate([trough_row, trough_row1]))
        trough_col = np.sort(np.concatenate([trough_col, trough_col1]))
        # 聚合数据
        trough_row = trough_clustering(trough_row, th=100).astype(np.int32)  # 行波谷点
        trough_col = trough_clustering(trough_col, th=100).astype(np.int32)  # 列波谷点
        # 分割
        segmentations = []
        copy = np.array([])
        if debug:
            copy = image_corrected.copy()
        for i in range(trough_row.shape[0] - 1):
            for j in range(trough_col.shape[0] - 1):
                if debug:
                    copy = cv2.rectangle(copy, (trough_col[j], trough_row[i]), (trough_col[j + 1], trough_row[i + 1]),
                                         (0, 255, 0), 10)
                img_segment = image_corrected[trough_row[i]:trough_row[i + 1], trough_col[j]:trough_col[j + 1], :]
                segmentations.append(
                    [np.array([[trough_col[j], trough_row[i]], [trough_col[j + 1], trough_row[i + 1]]], np.int32),
                     img_segment])
        if debug:
            plt.title("Debug demo")
            plt.imshow(copy)
            plt.show()
        return segmentations
    elif seg_method == 4:
        '''
        频谱分析法获取行列数
        '''
        def fft_to_rc(wave, r=100):
            rfft = np.abs(np.fft.rfft(255 - wave)/wave.shape[0])    # 傅里叶变换幅频图
            # 波峰检测
            peak = signal.find_peaks(rfft[0:r], distance=6, height=np.max(rfft)/3.0)[0]
            # for i in range(r-1):
            #
            #     if rfft[i] < rfft[i+1] and rfft[i+2] < rfft[i+1]:
            #         peak = np.append(peak, i+1)
            peak_with_0 = np.concatenate([[0], peak])   # 波峰序列补0
            rc_num = peak - peak_with_0[0:-1]   # 求相邻波峰间隔
            counts = np.bincount(rc_num)    # 统计
            if debug:
                plt.plot(rfft)
            return np.argmax(counts)    # 取众数作为最终行列数
        row_num = fft_to_rc(wave_row)
        col_num = fft_to_rc(wave_col)
        if debug:
            plt.legend(["wave_row", "wave_col"])
            plt.title("FFT Spectrum analysis diagram")
            plt.xlabel("Row and column index")
            plt.ylabel("Amplitude")
            plt.savefig("./saved_plot/FFTSpectrumAnalysisDiagram.pdf", dpi=300, format="pdf", bbox_inches='tight',
                        pad_inches=0.0)
            plt.show()
        # 确保行列数均是偶数
        if row_num.astype(np.int32) % 2 != 0:
            row_num = (row_num + 1.0).astype(np.int32)
        else:
            row_num = row_num.astype(np.int32)
        if col_num.astype(np.int32) % 2 != 0:
            col_num = (col_num + 1.0).astype(np.int32)
        else:
            col_num = col_num.astype(np.int32)
        height = int(image_corrected.shape[0] / row_num)  # 单块晶片的高度
        width = int(image_corrected.shape[1] / col_num)  # 单块晶片的宽度
        trough_row = np.int32([])
        trough_col = np.int32([])
        for i in range(int(row_num-1)):
            trough_row = np.append(trough_row, (i+1) * height)    # 生成行分割点
        for i in range(int(col_num-1)):
            trough_col = np.append(trough_col, (i+1) * width)     # 生成列分割点
        # 补充边缘线
        trough_row = np.concatenate([[0], trough_row, [image_corrected.shape[0] - 1]]).astype(np.int32)
        trough_col = np.concatenate([[0], trough_col, [image_corrected.shape[1] - 1]]).astype(np.int32)
        # 分割
        segmentations = []
        copy = np.array([])
        if debug:
            copy = image_corrected.copy()
        for i in range(trough_row.shape[0] - 1):
            for j in range(trough_col.shape[0] - 1):
                if debug:
                    copy = cv2.rectangle(copy, (trough_col[j], trough_row[i]), (trough_col[j + 1], trough_row[i + 1]),
                                         (0, 255, 0), 10)
                img_segment = image_corrected[trough_row[i]:trough_row[i + 1], trough_col[j]:trough_col[j + 1], :]
                segmentations.append(
                    [np.array([[trough_col[j], trough_row[i]], [trough_col[j + 1], trough_row[i + 1]]], np.int32),
                     img_segment])
        if debug:
            plt.title("Segmentation diagram")
            plt.imshow(copy)
            plt.savefig("./saved_plot/SegmentationDiagram.pdf", dpi=300, format="pdf", bbox_inches='tight',
                        pad_inches=0.0)
            plt.show()
        return segmentations





if __name__ == '__main__':
    all = False
    data_root = pathlib.Path('./photos')
    all_image_names = sorted(
        item.name for item in data_root.glob('*.JPG'))  # 获取文件名
    if not all:
        # image = random.choice(all_image_names)
        image = '41191510617617.JPG'
        print("image name: "+image)
        img_src = cv2.imread("photos/" + image)
        image_corrected = correct(img_src, debug=True)
        seg = segment(image_corrected, seg_method=4, debug=True)
        copy = image_corrected.copy()
    elif all:
        i = 1
        for image in all_image_names:
            plt.subplot(2, 3, i)
            print("image name: " + image)
            img_src = cv2.imread("photos/" + image)
            image_corrected = correct(img_src, debug=True)
            seg = segment(image_corrected, seg_method=4, debug=True)
            copy = image_corrected.copy()
            for seg_ele in seg:
                copy = cv2.rectangle(copy, tuple(seg_ele[0][0]), tuple(seg_ele[0][1]), (0, 255, 0), 10)
            plt.xlabel(image)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(copy)
            if i == 6:
                plt.show()
                i = 1
            else:
                i += 1
        plt.show()
