import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import scipy.fft as ft

# Generating a Gaussian Filter
def get_gaussian_kernel(kernel_size, sigma):
        kernel_u = np.linspace(-np.floor(kernel_size / 2), np.floor(kernel_size / 2), kernel_size)
        denominator = np.sqrt(2 * np.pi) * sigma  # using the separation principle
        for k in range(len(kernel_u)):
            numerator = np.exp(-(kernel_u[k]) ** 2 / (2 * (sigma ** 2)))
            kernel_u[k] = numerator / denominator
        gauss_kernel = np.outer(kernel_u.T, kernel_u.T)
        return gauss_kernel
    
# Creates a circular mask
def generate_mask(size, radius):
    height, width = size
    center = [int(height/2), int(width/2)]
    X, Y = np.ogrid[:height, :width]
    mask_area = (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius*radius
    mask = np.ones((height, width)) 
    mask[mask_area] = 0
    return mask

# Returns the edge-image using Fourier Transform  
def get_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thrshld_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # gaussian_kernel = get_gaussian_kernel(5, 5)
    # blur = cv2.filter2D(thrshld_img, -1, gaussian_kernel)
    blur = cv2.medianBlur(thrshld_img, 5)
    ft_img = ft.fft2(blur, axes=(0, 1))
    ft_shiftd = ft.fftshift(ft_img)
    magn_ft_shiftd = 20*np.log(np.abs(ft_shiftd))
    # masking the image8
    cmask = generate_mask(gray.shape, 275)
    ft_MaskingforEdges = ft_shiftd * cmask
    mag_masked = 20*np.log(np.abs(ft_MaskingforEdges))
    #retrieving the image
    rtrv_shft_img = ft.ifftshift(ft_MaskingforEdges)
    rtrv_img_raw = ft.ifft2(rtrv_shft_img)
    rtrv_img = np.uint8(np.abs(rtrv_img_raw))
    return rtrv_img

 
def drawHoughLines(image):
    """
    Computes and draws Hough lines atop a given frame
    Args:
        image (3D array): Three channel image
    Returns:
        3D array: Given image array with the identified lanes drawn on it
    """
    my_frame = image
    height, width = my_frame.shape[0], my_frame.shape[1]
    edge_img = get_edges(my_frame)
    et, edge_img = cv2.threshold(edge_img, 50, 255, cv2.THRESH_BINARY)
    minLineLength = 100
    maxLineGap = 5
    lines = cv2.HoughLinesP(edge_img, 2, np.pi/180, 40, minLineLength, maxLineGap)
    left_pts = list()
    right_pts = list()
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y1 >= height/2 and y2 >= height/2:
                slope = (y1-y2)/(x1-x2)
                if slope >= 0:
                    right_pts.append((x1, y1))
                    right_pts.append((x2, y2))
                else:
                    left_pts.append((x1, y1))
                    left_pts.append((x2, y2))               
    left_pts = np.array(left_pts)
    y_lmin = min(left_pts[:, 1])
    y_lmax = max(left_pts[:, 1])
    right_pts = np.array(right_pts)
    y_rmin = min(right_pts[:, 1])
    y_rmax = max(right_pts[:, 1])
    if len(left_pts) < len(right_pts):
        dashed_pts = left_pts
        solid_pts = right_pts
    else:
        if (y_lmax - y_lmin) < (y_rmax - y_rmin):
            dashed_pts = left_pts
            solid_pts = right_pts
        else:
            dashed_pts = right_pts
            solid_pts = left_pts
        
    ##### Fitting the dashed line
    x_dash = dashed_pts[:, 0] 
    y_dash = dashed_pts[:, 1]
    dash_params = np.polyfit(x_dash, y_dash, 1)  # slope and intercept
    x1 = np.min(x_dash).astype('int')
    y1 = (dash_params[0]*x1 + dash_params[1]).astype('int')
    x2 = np.max(x_dash).astype('int')
    y2 = (dash_params[0]*x2 + dash_params[1]).astype('int')
    cv2.line(my_frame, (x1,y1), (x2,y2), (0, 0, 255), 5)
    
    ##### Fitting the solid line
    x_solid = solid_pts[:, 0]
    y_solid = solid_pts[:, 1]
    solid_params = np.polyfit(x_solid, y_solid, 1)
    x1 = np.min(x_solid).astype('int')
    y1 = (solid_params[0]*x1 + solid_params[1]).astype('int')
    x2 = np.max(x_solid).astype('int')
    y2 = (solid_params[0]*x2 + solid_params[1]).astype('int')
    cv2.line(my_frame, (x1,y1), (x2,y2), (0, 255, 0), 5)
    
    return my_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InputFilePath', default='data/whiteline.mp4', help='Path to the input file')
    parser.add_argument('-s', '--SaveFileName', default='result2.avi', help='Name of the output file')
    
    args = parser.parse_args()
    video_path = args.InputFilePath
    save_path = args.SaveFileName
    
    # Visualizing at the output generated
    cap = cv2.VideoCapture(video_path)
    vid_resolution = (1280, 720)
    result = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 10, vid_resolution)
    disp_duration = 10
    all_frames = list()
    if not cap.isOpened():
        print("Error opening video stream or file!")
    count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            count += 1
            print("Evaluating frame ", count)
            all_frames.append(frame)
            try:
                hLineImg = drawHoughLines(frame)
            except:
                continue
            result.write(hLineImg)
            cv2.imshow("frame", hLineImg)
            if cv2.waitKey(disp_duration) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    result.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

