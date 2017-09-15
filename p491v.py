#
#C. D. Vijay, CarND-Term 1, Project 4

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import argparse

#Arrays to save object points (3D) and image points (2D)
obj_points = []
img_points = []

# prepare object points; count actual 4-way corners in a row and col. of a chessboard image from a specific cam
nx = 9
ny = 6

#fname = 'camera_cal/calibration2.jpg'
#image = mpimg.imread(fname)
#plt.imshow(image)
#plt.show()


def Show2Imgs (img1, img2, str1, str2):
    global obj_points, img_points
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    f.tight_layout()
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(str1, fontsize=15)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(str2, fontsize=15)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def FindDrawChessboardCorners (img, nx, ny):
    showImgs = 0
    
    # Create obj points based on nx, ny
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid [0:nx, 0:ny].T.reshape(-1,2)

    #Retain original img by making a copy
    ImgCopy = img.copy()
    img_size = (img.shape[1], img.shape[0])
    #print (img_size)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)
    
    if ret == True:
        img_points.append(corners)
        obj_points.append(objp)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
   
    return corners
    #if (showImgs):
    #    Show2Imgs(ImgCopy, img, 'Original', 'WithChessboardCornersDrawn')

def GradientDir (img, fn, thresh=(0, np.pi/2)):
    #gray = img
    kernel_size = 15
    gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    sx = np.absolute(cv2.Sobel (gray, cv2.CV_64F, 1, 0))
    sy = np.absolute(cv2.Sobel (gray, cv2.CV_64F, 0, 1))
    
    gradient = np.arctan2(sy, sx)
    
    gradient_binary = np.zeros_like(gradient)
    gradient_binary [(gradient >= thresh[0]) & (gradient <= thresh[1])] = 1
   
    return gradient_binary
    #Show2Imgs(img, gradient_binary, fn, 'Dir. of Gradient')
    
def ApplySobel (img, fn, xorient, thresh_min=0, thresh_max=255):

    #Experimenting with Sobel operators, gradients, etc.
    gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    
    #Sobel x or y depending on value of xorient(ation)
    if xorient == 1: #Sobel along x-axis
        sobel_gray = cv2.Sobel (gray, cv2.CV_64F, 1, 0)
        abs_sobel = np.absolute (sobel_gray)
        sobel_gray_scaled = np.uint8 (255*abs_sobel/np.max(abs_sobel))
        str1 = 'Sobel X - Gray, ' + fn
    
    elif xorient == 0 : #Sobel along y-axis
        sobel_gray = cv2.Sobel (gray, cv2.CV_64F, 0, 1)
        abs_sobel = np.absolute (sobel_gray)
        sobel_gray_scaled = np.uint8 (255*abs_sobel/np.max(abs_sobel))
        str1 = 'Sobel Y - Gray, ' + fn
    
    elif xorient == 2:  #Sobel along both
        sobel_grayx = cv2.Sobel (gray, cv2.CV_64F, 1, 0)
        sobel_grayy = cv2.Sobel (gray, cv2.CV_64F, 0, 1)
        mag_grad = np.sqrt (sobel_grayx**2 + sobel_grayy**2)
        scalefactor = np.max(mag_grad)/255
        sobel_gray_scaled = np.uint8 (mag_grad/scalefactor) #8-bit rescaling
        str1 = 'Sobel X & Y - Gray, ' + fn
        
    sx_binary = np.zeros_like (sobel_gray_scaled)
    sx_binary [(sobel_gray_scaled >= thresh_min) & (sobel_gray_scaled <= thresh_max)] = 1
        
    #Show2Imgs(sx_binary, sx_binary, str1, 'Same')
    return sx_binary

def ColorThreshold (img, fn, thresh=(170,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls [:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary [(s_channel>= thresh[0]) & (s_channel <= thresh[1])] = 1
    #Show2Imgs(img, s_binary, fn, 'S-channel')
    return s_binary
    
def WarpPerspectiveXform (thresh_img, corners):

    img = thresh_img
    img_size = (img.shape[1], img.shape[0])
        
    #Get source points for persp. xform
    src = np.float32([[190, 720], [548, 480], [740, 480], [1130, 720]])
    dst = np.float32([[190, 720], [190, 0],   [1130, 0],  [1130, 720]])
    
    #src = np.float32([[200, 720], [480, 510], [780, 510], [1130, 720]])
    #dst = np.float32([[200, 720], [200, 100], [1130, 100],[1130, 720]])
    #src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    #Get dest points to grab a fraction of the img for persp. xform
    #offset = 150
    #dst = np.float32([[offset, offset], [img_size[0]-offset, offset], \
    #                                 [img_size[0]-offset, img_size[1]-offset], \
    #                                 [offset, img_size[1]-offset]])
    #print ("src", src, "\n", "dest", dst)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return warped, Minv

def DrawImg(image, warped, left_fitx, right_fitx, ploty, Minv, undist):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    
    # Combine the result with the original image
    print ('Shapes -und, warped',  undist.shape, newwarp.shape, undist.size, newwarp.size)
    result = cv2.addWeighted(undist, 1, newwarp[:,:,0], 0.3, 0)
    #return result
    plt.imshow(result)
    plt.show()

#EXECUTION STARTS HERE
parser = argparse.ArgumentParser(description='Udacity-CarND-T1-P4-: Advanced Lane Detection')
parser.add_argument('-d', action="store", dest="debug" )
parser.add_argument ('-v', action="store", dest="video")
args=parser.parse_args()

if (args.debug == '1'):
      debug = 1
else :
      debug = 0
   
# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
    
#Read and calibrate camera chessboard images
print ('Calibrating camera with ', len(images), ' chessboard images...')
for filen in images:
    image = mpimg.imread(filen) 
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = FindDrawChessboardCorners (gray, nx, ny)  

#Calibrate camera now    
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
print ("Calibration done")

parser = argparse.ArgumentParser(description='Udacity-CarND-T1-P4-: Advanced Lane Detection')
parser.add_argument('-d', action="store", dest="debug" )
#parser.add_argument ('-v', action="store", dest="video")
args=parser.parse_args()

if (args.debug == '1'):
      debug = 1
else :
      debug = 0

if debug: 
    print ('\nDebug Mode ON...\n')
print ('\nPICK ONE: -OR- Enter to Exit')
print ('___________________________________________')
print ('1:  Project Video')
print ('2:  Challenge Video ')
print ('3:  Harder Challenge Video')
print ('4:  Test images')
choice = input('\nYour Choice (Enter to exit): ')

video = 0  
font = cv2.FONT_HERSHEY_SIMPLEX

# Read imgs or a video filename 
if   (choice == '1'): 	
    name = 'project_video.mp4'
    video = 1
elif (choice == '2'):  
    name = 'challenge_video.mp4'
    video = 1
elif (choice == '3'):  
    name = 'harder_challenge_video.mp4'
    video = 1
elif (choice == '4'):
    #Work on test images -- color and gradient and sobel x/y thresholding
    images = glob.glob('test_images/test*.jpg')
else: 
    exit()
    
if video:
    cap = cv2.VideoCapture(name)
    framenum = 1

imgindex = 0
    
while True:

    if video:
        ret, image  = cap.read()
        if ret == False: 
            break
        if debug: 
            print ("Frame: ", framenum)
        fn = str(framenum)
    else:
        image = mpimg.imread(images[imgindex])
        fn = str(images[imgindex])
        
    print ('Processing', fn, '...')
    
    #Apply Sobel
    sx_binary = ApplySobel (image, fn, 1, 30, 100) # Second arg. is orientation,: 0 (y-axis) or 1 (x-axis), 2 (both)
    #print (sx_binary.shape)
    
    #Apply gradients  - NOT USING AS RESULT LESS THAN SATISFACTORY
    #gradient = GradientDir (image, fn, thresh=(0.85, 1.1))
    #print (gradient.shape)
    
    #Apply color channel separation (doing HLS)
    s_channel = ColorThreshold (image, fn, thresh=(170,255))
    #print (s_channel.shape)
    
    combined_binary = np.dstack(( np.zeros_like(sx_binary), sx_binary, s_channel))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    #combined_binary[(s_channel == 1) | (sx_binary == 1)] = 1
    combined_binary[(s_channel == 1) | (sx_binary == 1)] = 1
    #Show2Imgs(image, combined_binary, 'Original', 'Sx+S_channel Combined')
    
    #Below not a good combo, all 3, so commented out
    #combined_binary1 = np.dstack(( np.zeros_like(combined_binary), combined_binary, gradient))
    #combined_binary1[((combined_binary == 1) | (gradient == 1))] = 1
    #Show2Imgs(image, combined_binary1, 'Original', 'All 3 Combined')
    
    #Undistort the thresholded image using the above calib matrix and vectors
    und = cv2.undistort (combined_binary, mtx, dist, None, mtx)
    
    #Calculate perspective transform matrix
    str1 = 'Thresholded Binary'+fn
    binary_warped, Minv =  WarpPerspectiveXform (und, corners)
    if (debug): Show2Imgs (image, binary_warped, str1, 'Undistorted and Xformed')
    
    
    #Search for lane lines using a sliding window and fit a polynomial to the lines

    # Taking a histogram of the bottom half of the "binary_warped" image
    index1 = binary_warped.shape[0]/2
    histogram = np.sum(binary_warped[int(index1):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    #DrawImg(image, binary_warped, left_fitx, right_fitx, ploty, Minv, und)
        
    #if cv2.waitKey(5) & 0xFF == ord('q'):
    #    break
    
    if debug:
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    
    if (video == 1): 
        framenum+=1
        
    elif (video == 0):
        if (imgindex < len(images)-1):
            imgindex+=1
        else:
            break


if video: 
    cap.release()
    cv2.destroyAllWindows()