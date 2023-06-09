import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os

# Function to compute the histograms
def get_hist(arr, bins):
    hist = dict()
    bin_vals = list()
    for bin in bins:
        count = (arr == bin).sum()
        bin_vals.append(count)
    hist['bins'] = bins
    hist['counts'] = bin_vals
    return hist 


#######################################################################################
############################### Histogram Equalization ################################
#######################################################################################
def hist_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_y = img_yuv[:, :, 0]
    flat_img = img_y.flatten()
    my_bins = list(range(0, 256))
    histogram = get_hist(flat_img, my_bins)
    norm_cum_sum = np.cumsum(histogram['counts'])/len(flat_img)
    norm_cum_sum = 255*norm_cum_sum
    transformed_pxls = np.array([norm_cum_sum[pxl] for pxl in flat_img], dtype="uint8").reshape(img_y.shape)
    img_yuv[:, :, 0] = transformed_pxls
    img_final = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_final


#######################################################################################
########################### Adaptive Histogram Equalization ###########################
#######################################################################################
def adap_hist_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_y = img_yuv[:, :, 0]
    img_width, img_height = img_y.shape[1], img_y.shape[0]
    blank_slate = np.zeros(img_y.shape, dtype="uint8")
    my_bins = list(range(0, 256))
    tile_width = [int(val) for val in np.linspace(0, img_width, num=8)]
    tile_height = [int(val) for val in np.linspace(0, img_height, num=8)]
    for h in range(len(tile_height)-1):
        for w in range(len(tile_width)-1):
            tile = img_y[tile_height[h]:tile_height[h+1], tile_width[w]:tile_width[w+1]]
            flat_tile = tile.flatten()
            histogram = get_hist(flat_tile, my_bins)
            norm_cum_sum = np.cumsum(histogram['counts'])/len(flat_tile)
            norm_cum_sum = 255*norm_cum_sum
            transformed_pxls = np.array([norm_cum_sum[pxl] for pxl in flat_tile], dtype="uint8").reshape(tile.shape)
            blank_slate[tile_height[h]:tile_height[h+1], tile_width[w]:tile_width[w+1]] = transformed_pxls
    img_yuv[:, :, 0] = blank_slate
    img_final = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InputFilePath', default='data/adaptive_hist_data/', help='Folder in which the input files are located')
    parser.add_argument('-t', '--HistType', default='norm', help='Type of histogram equalization to be used: adaptive or non-adaptive, default - non-adaptive')
    parser.add_argument('-s', '--SaveFileName', default='result1_norm.avi', help='Name of the output file')

    args = parser.parse_args()
    InputFilePath = args.InputFilePath
    hist_type = args.HistType
    save_path = args.SaveFileName
    images_to_stitch = list()
    for file_name in os.listdir(InputFilePath):
        # print(file_name)
        if ".png" or ".jpg" in file_name:
            image_path = os.path.join(InputFilePath, file_name) 
            img = cv2.imread(image_path)
            if hist_type == "norm":
                processed_img = hist_equalization(img)
            if hist_type == "adap":
                processed_img = adap_hist_equalization(img)
            images_to_stitch.append(processed_img)
    vid_resolution = (int(images_to_stitch[-1].shape[1]), int(images_to_stitch[-1].shape[0]))
    result = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 1.5, vid_resolution)
    for im in images_to_stitch:
        result.write(im)
        cv2.imshow('Histogram', im)
        if cv2.waitKey(100) == ord('q'):
            print("Saving video.......")
            break
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    result.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()




