import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import scipy.fft as ft


# Plot the selected trapezoid 
def plot_quadrilateral(frame, points):
    trapezoid = cv2.polylines(frame, np.int32([points]), True, (10,120,255), 3)
    while(1):
      cv2.imshow('Trapezoid', trapezoid) 
      if cv2.waitKey(0):
        break
    cv2.destroyAllWindows()


# Make a gray channel image 3-dimensional
def get3ChannelGray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im_wdth, im_hght = gray.shape[0], gray.shape[1]
    new_img = np.zeros((im_wdth, im_hght), dtype=np.uint8)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    new_img[int(im_wdth/2):im_wdth, 0:im_hght] = thresh[int(im_wdth/2):im_wdth, 0:im_hght]
    gray_3channel = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
    return gray_3channel


# Calculating the Radius of Curvature
def calculateROC(left_lane_pts, right_lane_pts, y_eval = 670):
    # Fitting polynomial curves to approximate real-world distances
    left_fit_cr = np.polyfit(left_lane_pts[:, 1], left_lane_pts[:, 0], 2)
    right_fit_cr = np.polyfit(right_lane_pts[:, 1], right_lane_pts[:, 0], 2)
                
    left_curvem = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0])
    right_curvem = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / (2*right_fit_cr[0])
    
    return left_curvem, right_curvem
        

# Displaying the curvature info
def showCurvature(image, left_curvem, right_curvem):
    image_size = image.shape[::-1][1:]
    img_width, img_height = image_size[0], image_size[1]
    cv2.putText(image,'Curve Radius: '+str((left_curvem+right_curvem)/2)[:7], (int((5/600)*img_width), int((20/338)*img_height)), 
                cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*img_width)),(255,255,255), 2, cv2.LINE_AA)
    if (left_curvem+right_curvem)/2 < -100:
        cv2.putText(image,"Turning left", (int((10/600)*img_width), int((40/338)*img_height)), 
                cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*img_width)),(255,255,255), 2, cv2.LINE_AA)
    if (left_curvem+right_curvem)/2 > -100 and (left_curvem+right_curvem)/2 < 100:
        cv2.putText(image,"Going straight", (int((10/600)*img_width), int((40/338)*img_height)), 
                cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*img_width)),(255,255,255), 2, cv2.LINE_AA)
    if (left_curvem+right_curvem)/2 > 100:
        cv2.putText(image,"Turning right", (int((10/600)*img_width), int((40/338)*img_height)), 
                cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*img_width)),(255,255,255), 2, cv2.LINE_AA)
    return image


# Warping the frame to get a bird's eye view
def getWarpedFrame(frame):
    image_size = frame.shape[::-1][1:]
    img_width = image_size[0]
    img_height = image_size[1]
    src_pts = np.float32([(585, 460), # Top-left
                        (260, 670), # Bottom-left
                        (1100, 670), # Bottom-right
                        (737, 460)]) # Top-right
    padding = int(0.25 * img_width)
    dst_pts = np.float32([[padding, 0], # Top-left
                        [padding, img_height], # Bottom-left
                        [img_width-padding, img_height], # Bottom-right
                        [img_width-padding, 0]]) # Top-right
    homography_mtrx = cv2.getPerspectiveTransform(src_pts, dst_pts) # Transformation between source and destination points
    warped_frame = cv2.warpPerspective(frame, homography_mtrx, image_size, flags=(cv2.INTER_LINEAR))
    # Convert image to binary
    _, binary_warped = cv2.threshold(warped_frame, 180, 255, cv2.THRESH_BINARY)  
    # binary_warped = cv2.polylines(binary_warped, np.int32([dst_pts]), True, (147,20,255), 3)
    return binary_warped


# Returns the lane indices
def getLaneIndices(frame):
    warped = getWarpedFrame(frame)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh_prime = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    row_split = np.array_split(thresh_prime, 45, axis=0)
    blocks = list()
    for l in row_split:
        b = np.array_split(l, 80, axis=1)
        blocks += b

    block_averages = [np.mean(block) for block in blocks]
    block_averages = np.reshape(block_averages, (45, 80)).astype(np.uint8)
    _, thresh_prime = cv2.threshold(block_averages, 150, 255, cv2.THRESH_BINARY)
    block_indices = list(zip(*np.where(thresh_prime == 255)))
    for idx1 in block_indices:
        for idx2 in block_indices:
            if idx1 != idx2:
                if idx1[0] == idx2[0] and idx2[1] == idx1[1]+1:
                    block_indices.remove(idx1)
    ##### Scaling up the block indices #####
    scaled_indices = list()
    for idx in block_indices:
        scaled_indices.append((16*idx[1] + 8, 16*idx[0] + 8))
    left_indices = list()
    right_indices = list()
    for idx in range(len(scaled_indices)):
        if scaled_indices[idx][0] > 500:
            right_indices.append(scaled_indices[idx])
        else:
            left_indices.append(scaled_indices[idx])

    left_indices = np.array(left_indices)
    right_indices = np.array(right_indices)

    return left_indices, right_indices


# Draw the lanes obtained onto the warped image
def drawLaneLines(frame, lparameter_hist, rparameter_hist, history_weight = 0.5):
    image_size = frame.shape[::-1][1:]
    img_width, img_height = image_size[0], image_size[1]
    left_lane_indices, right_lane_indices = getLaneIndices(frame)
    lfit_params = np.polyfit(left_lane_indices[:, 1], left_lane_indices[:, 0], deg=2)
    rfit_params = np.polyfit(right_lane_indices[:, 1], right_lane_indices[:, 0], deg=2)
    lparameter_hist = np.array(lparameter_hist)
    rparameter_hist = np.array(rparameter_hist)
    if len(lparameter_hist) != 0 and len(rparameter_hist) != 0:
        ldist = np.linalg.norm(lparameter_hist-lfit_params)
        rdist = np.linalg.norm(rparameter_hist-rfit_params)
        # print("Euclidean distance between parameters: ", dist)
        if ldist > 100 or rdist > 100:
            history_weight = 1.
        lfit_params = history_weight*lparameter_hist.mean(axis=0) + (1-history_weight)*lfit_params
        rfit_params = history_weight*rparameter_hist.mean(axis=0) + (1-history_weight)*rfit_params
    y_range = np.linspace(0, img_width-1, img_width)
    x_lfit = [lfit_params[0]*pt*pt + lfit_params[1]*pt + lfit_params[2] for pt in y_range]
    x_rfit = [rfit_params[0]*pt*pt + rfit_params[1]*pt + rfit_params[2] for pt in y_range]
    
    left_fit_idxs = list()
    for p in range(len(y_range)):
        left_fit_idxs.append((int(x_lfit[p]), y_range[p]))
    right_fit_idxs = list()
    for p in range(len(y_range)):
        right_fit_idxs.append((int(x_rfit[p]), y_range[p]))
    warp_copy = getWarpedFrame(frame).copy()
    cv2.polylines(warp_copy, np.int_([left_fit_idxs]), False, (0,255,0), thickness=5)
    cv2.polylines(warp_copy, np.int_([right_fit_idxs]), False, (0,255,0), thickness=5)
    
    return warp_copy, lfit_params, rfit_params, left_fit_idxs, right_fit_idxs


# Overlay the detected lane area over the image
def getOverlay(frame, left_fit, right_fit):
    image_size = frame.shape[::-1][1:]
    img_width, img_height = image_size[0], image_size[1]
    src_pts = np.float32([(585, 460), # Top-left
                        (260, 670), # Bottom-left
                        (1100, 670), # Bottom-right
                        (737, 460)]) # Top-right
    padding = int(0.25 * img_width)
    dst_pts = np.float32([[padding, 0], # Top-left
                        [padding, img_height], # Bottom-left
                        [img_width-padding, img_height], # Bottom-right
                        [img_width-padding, 0]]) # Top-right
    homography_mtrx = cv2.getPerspectiveTransform(src_pts, dst_pts) # Transformation between source and destination points
    inv_homography_mtrx = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
    warped_frame = cv2.warpPerspective(frame, homography_mtrx, image_size, flags=(cv2.INTER_LINEAR))
    img_width, img_height = warped_frame.shape[0], warped_frame.shape[1]
    warp_zero = np.zeros((img_width, img_height)).astype(np.uint8)
    
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))       
    
    left_fit = np.array(left_fit)
    right_fit = np.array(right_fit)
    pts_left = np.array([np.transpose(np.vstack([left_fit[:, 0], left_fit[:, 1]]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit[:, 0], right_fit[:, 1]])))])
    pts = np.hstack((pts_left, pts_right))
         
    # Draw lane on the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, inv_homography_mtrx, (img_height, img_width))
    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InputFilePath', default='Data/challenge.mp4', help='Path to the input file')
    parser.add_argument('-f', '--Flip', default=False, type=bool, help='Specify if the frame is to be flipped or not')
    parser.add_argument('-s', '--SaveFileName', default='result3.avi', help='Name of the output file')
    
    args = parser.parse_args()
    video_path = args.InputFilePath
    flip = args.Flip
    save_path = args.SaveFileName
    
    # Visualizing at the output generated
    cap = cv2.VideoCapture(video_path)
    vid_resolution = (1280, 720)
    result = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 10, vid_resolution)
    disp_duration = 1
    all_frames = list()
    if not cap.isOpened():
        print("Error opening video stream or file!")
    count = 0
    left_param_history = list()
    right_param_history = list()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if flip == True:
                frame = cv2.flip(frame, 1)       # Flipping and testing the video!
            count += 1
            # print("Evaluating frame ", count)
            all_frames.append(frame)
            try:
                gray_3channel = get3ChannelGray(frame)
                LineImg, lfit_params, rfit_params, lfit_indices, rfit_indices = drawLaneLines(frame, left_param_history, right_param_history)
                left_param_history.append(lfit_params)
                right_param_history.append(rfit_params)
                l_pts, r_pts = getLaneIndices(frame)
                l_ROC, r_ROC = calculateROC(l_pts, r_pts)
                overlay = getOverlay(frame, lfit_indices, rfit_indices)
                frame_with_curvature = showCurvature(overlay, l_ROC, r_ROC)
                
                small_1 = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)     # Rescaling images for fitting into one single frame
                small_2 = cv2.resize(gray_3channel, (0,0), fx=0.5, fy=0.5)
                small_3 = cv2.resize(frame_with_curvature, (0,0), fx=0.5, fy=0.5)
                small_4 = cv2.resize(LineImg, (0,0), fx=0.5, fy=0.5)
                numpy_horizontal_concat1 = np.concatenate((small_1, small_2), axis=1)
                numpy_horizontal_concat2 = np.concatenate((small_3, small_4), axis=1)
                numpy_vertical_concat = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2), axis=0)
            except:
                continue
            result.write(numpy_vertical_concat)
            cv2.imshow("frame", numpy_vertical_concat)
            if cv2.waitKey(disp_duration) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    result.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()