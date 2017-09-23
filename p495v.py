#
#C. D. Vijay, CarND-Term 1, Project 4, Advanced Lane Detection

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import argparse

#Arrays to save object points (3D) and image points (2D)
obj_points = []
img_points = []

#Prepare object points; count actual 4-way corners in a row and col. of a chessboard image from a specific cam
nx = 9
ny = 6

def Show2Imgs (img1, img2, str1, str2, src=None, dst=None):
    global obj_points, img_points
    if src != None:
        print ('src, dst', src, dst)
        
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    f.tight_layout()
    ax1.imshow(img1, cmap='gray')
    if src != None:
        for i in np.arange(4):
            ax1.plot (src[i][0], src[i][1], 'rs')
            #ax1.add_line(Line2D(src[i][0], src[i][1], linewidth=2,'rs'))
            
    ax1.set_title(str1, fontsize=15)
    ax2.imshow(img2, cmap='gray')
    if dst != None:
        for i in np.arange(4):
            ax2.plot (dst[i][0], dst[i][1], 'rs')
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
    
    #Show2Imgs(ImgCopy, img, 'Original', 'WithChessboardCornersDrawn')
    return corners
    

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
    #corners -- not used
    img = thresh_img
    img_size = (img.shape[1], img.shape[0])
        
    #Demarcate source points for perspective xform, after experiementation
    src = np.float32([[190, 720], [548, 480], [740, 480], [1130, 720]])
    dst = np.float32([[190, 720], [190, 0],   [1180, 0],  [1180, 720]])
        
    #src = np.float32([[200, 720], [480, 510], [780, 510], [1130, 720]])
    #dst = np.float32([[200, 720], [200, 100], [1130, 100],[1130, 720]])
    #src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    #Get dest points to grab a fraction of the img for persp. xform
    #offset = 150
    #dst = np.float32([[offset, offset], [img_size[0]-offset, offset], \
    #                                 [img_size[0]-offset, img_size[1]-offset], \
    #                                 [offset, img_size[1]-offset]])
    #print ("src", src, "\n", "dest", dst)
    
    #Plot the points to check whether the ROI is parallel in the warped image
    #First, plot src points on orig. followed by dst on the warped
        
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    if debug: Show2Imgs(img, warped, 'src window points', 'dst window points', src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return warped, Minv

def CheckWhetherCurvesParallel(ploty, left_fitx, right_fitx, fn):
    #We trisect the height of the lane (curve) into top, middle and bottom (ok, bisect). 
    #And check for the distances b/w the corr. points on the left and the right curves.
    #compare the mod of that against a threshold to ascertain approx. parallelism.
    numdiv = 3
    distances=[]
    
    #Top of the lane (left pt, right pt)
    checkpoints = np.linspace(0, len(ploty), numdiv)
    if (debug): print ('Checking distance along these vertical points', checkpoints)
    for i in checkpoints:
        i = int(i)
        if (i >= len(ploty)):  
            i = len(ploty) -1
        if (debug): print ('at pt.', i, 'left x=', left_fitx[i], 'right x', right_fitx[i])
        #Use 2d distance formula b/w points, forget the (y2-y1) term as y is the same for every pair checked
        distances.append(np.sqrt((left_fitx[i] - right_fitx[i])**2))
    #print ('Frame', fn, "Distances", distances)
    #print ('\nMax. dist b/w lane lines=', np.max(distances), '\n')
    
    #Check differences, set tolerance to 25% of lane width (currently, 3.7m or 700 pix)
    tolerance = 0.25 * np.min (distances)
    if   (np.abs(distances[0] - distances[1])) > tolerance:
        print ('Lane line deviation > 25% of lane width! (in m)', np.abs(distances[0] - distances[1])*3.7/700)
    elif np.abs(distances[1] - distances[2]) > tolerance:
        print ('Lane line deviation > 25% of lane width! (in m)', np.abs(distances[1] - distances[2])*3.7/700)
    elif np.abs(distances[0] - distances[2]) > tolerance:
        print ('Lane line deviation > 25% of lane width! (in m)', np.abs(distances[0] - distances[2])*3.7/700)
    #print("Tolerance: ", tolerance)
  
def DetectLanes_FitPoly(binary_warped, debug, fn):
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
    #print ('midpt', midpoint, 'leftxbase', leftx_base, 'rightxbase', rightx_base)

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
    
    #Simple check for parallel curves (also called offset curves), 
    CheckWhetherCurvesParallel(ploty, left_fitx, right_fitx, fn)
        
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    if (debug):
        print ('Iter1 ploty', ploty, '\n', left_fitx, '\n', right_fitx)
        if ret == 0: #if curves are not parallel
            print ("We got a problem, doc! These two lanes curves may not form a lane at all!")
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    
    if ".jpg" not in fn: #meaning a video frame is being analyzed here
        return left_fit, right_fit, out_img
    else: #meaning an image file is being analyzed here
        return left_fitx, right_fitx, ploty
   
def SearchSubsequentFrames (binary_warped, left_fit, right_fit, out_img, debug, fn):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    if len(left_fit)==0 or len(right_fit) ==0:
        print ('left fit or right fit seems bad, exiting.') #Need to take better care of this exception...
        exit()
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    if (debug): print ('Iter2+ ploty', ploty, '\n', left_fitx, '\n', right_fitx)
    CheckWhetherCurvesParallel(ploty, left_fitx, right_fitx, fn)
    
    #if len(ploty) == 0: return 0, Failed to detect lane lines by searching around margin of the prev. frame
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    if (debug):
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    
    return ploty, left_fitx, right_fitx

def CalcRadOfCurvature(ploty, left_fit, right_fit, img_width): #Left fit and right fit are actual plottable x and y vals, not the 'other' polyfit returned values
    
    y_eval = np.max(ploty) 
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fit*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fit*xm_per_pix, 2)
    # Calculate the new radii of curvature in meters
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    #Calc. car / lane center offset
    #off_center = xm_per_pix * ((right_fitx[-1] - left_fitx[-1]) - img_width)/2 
    off_center = -(left_fitx[-1] + right_fit[-1]-1280)/2 * xm_per_pix
    return left_curverad, right_curverad, off_center
    #print(left_curverad, 'm', right_curverad, 'm')
    
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
    #plt.imshow(newwarp)  #This is the green color_warp being xformed to the perspective view (similar to a trapezoid)
    #plt.show()
    
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp[:,:], 1.0, 0)
    #plt.imshow(result)
    #plt.show()
    #print(result.shape) (720, 1280, 3)
    return result

#EXECUTION STARTS HERE
parser = argparse.ArgumentParser(description='Udacity-CarND-T1-P4-: Advanced Lane Detection')
parser.add_argument('-d', action="store", dest="debug" )
parser.add_argument ('-v', action="store", dest="video")
args=parser.parse_args()

if (args.debug == '1'):
      debug = 1
else :
      debug = 0
   
#CAMERA CALIBRATION: Make a list of calibration images first
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
#und = cv2.undistort (image, mtx, dist, None, mtx)
#Show2Imgs(image, und, 'Original', 'Undistorted')

print ("Calibration done")

if debug: 
    print ('\nDebug Mode ON...\n')
print ('Advanced Lane Detection, Car ND-P4, C D Vijay')
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
        fn = str(framenum)
    else:
        image = mpimg.imread(images[imgindex])
        fn = str(images[imgindex])
        
    print ('Processing ', fn, '...')
    
    und = cv2.undistort (image, mtx, dist, None, mtx)
    #Show2Imgs(image, und, 'Original', 'Undistorted')
    
    #Apply Sobel
    sx_binary = ApplySobel (und, fn, 1, 30, 100) # Second arg. is orientation,: 0 (y-axis) or 1 (x-axis), 2 (both)
    #print (sx_binary.shape)
    
    #Apply gradients  - NOT USING AS RESULT LESS THAN SATISFACTORY
    #gradient = GradientDir (image, fn, thresh=(0.85, 1.1))
    #print (gradient.shape)
    
    #Apply color channel separation (doing HLS), this range produces less noise than (90,255)
    s_channel = ColorThreshold (und, fn, thresh=(170,255))
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
    str1 = 'Thresholded Binary ' + fn
    binary_warped, Minv =  WarpPerspectiveXform (und, corners)
    if (debug): Show2Imgs (image, binary_warped, str1, 'Undistorted and Xformed')
    
    #Search for lane lines using a sliding window and fit a polynomial to the lines
   
    if (video == 1 and framenum == 1):
        #print('Calling lane detection via sliding windows  ', framenum)
        left_fit, right_fit, out_img = DetectLanes_FitPoly(binary_warped, debug, fn)
        #print ('left_fit',left_fit, '\nright fit', right_fit)
    
    elif (video == 1 and framenum > 1):
       
        #Next check if prev. procedure did not produce lane lines, then need to call Sliding windows routine
        ploty, left_fitx, right_fitx = SearchSubsequentFrames(binary_warped, left_fit, right_fit, out_img, debug, fn)
        #Have to watch for RankWarnings from the above polyfit which indicates a bad fit; not implem yet
        #print (left_fitx, '\n', right_fitx)
        l_radius, r_radius, off_center = CalcRadOfCurvature(ploty, left_fitx, right_fitx, image.shape[1])
        #if ret == 0:
        #    print ('Search did not yield any lane lines in frame # ', framenum)
        #    ploty, left_fitx, right_fitx = DetectLanes_FitPoly(binary_warped)
        #Converting ROC from m to KM
        l_radius /= 1000
        r_radius /= 1000
        
        res = DrawImg(image, binary_warped, left_fitx, right_fitx, ploty, Minv, und)
        radii_txt = "Lane Curvature: Left: {0:.2f} km - Right: {1:.2f} km".format(l_radius, r_radius)
        center_offset = "Center offset: {0:.2f} m".format(off_center)
        personal = "C. D. Vijay, Udacity CarND Program, 2017"
        cv2.putText(res, radii_txt, (10,15), font, 0.5, (255,255,255), 0, cv2.LINE_AA)
        cv2.putText(res, center_offset, (10,30), font, 0.5, (0,255,255), 0, cv2.LINE_AA)
        cv2.putText(res, personal, (930, 15), font, 0.5, (255,0,0), 0, cv2.LINE_AA)
        cv2.putText(res, name, (580, 710), font, 0.5, (255,0,255), 0, cv2.LINE_AA)
        cv2.imshow('Advanced Lane Detection',res)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    else:
        left_fitx, right_fitx, ploty = DetectLanes_FitPoly(binary_warped, debug, fn)
        l_radius, r_radius, off_center = CalcRadOfCurvature(ploty, left_fitx, right_fitx, image.shape[1])
        l_radius /= 1000
        r_radius /= 1000
        
        res = DrawImg(image, binary_warped, left_fitx, right_fitx, ploty, Minv, und)
        
        name = fn
        radii_txt = "Lane Curvature: Left: {0:.2f} km - Right: {1:.2f} km".format(l_radius, r_radius)
        center_offset = "Center offset: {0:.2f} m".format(off_center)
        personal = "C. D. Vijay, Udacity CarND Program, 2017"
        cv2.putText(res, radii_txt, (10,15), font, 0.5, (255,255,255), 0, cv2.LINE_AA)
        cv2.putText(res, center_offset, (10,30), font, 0.5, (0,255,255), 0, cv2.LINE_AA)
        cv2.putText(res, personal, (930, 15), font, 0.5, (255,0,0), 0, cv2.LINE_AA)
        cv2.putText(res, name, (580, 710), font, 0.5, (255,0,255), 0, cv2.LINE_AA)
        cv2.imshow('Advanced Lane Detection',res)
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        
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