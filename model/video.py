from __future__ import division 
import os
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")



from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.signal import resample, savgol_filter, butter, filtfilt
from PIL import Image, ImageDraw
import cv2

import tensorflow as tf

from keras import backend as K
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from custom_functions_video import IoU, sort_contours, contour_edge, intersection,  distFunc

# Function to detect mouse clicks for the purpose of image calibration
def mclick(event, x, y, flags, param):

    # grab references to the global variables
    global mlocs

    # if the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        mlocs.append(y)
        print("Mousclick")
        print(mlocs)

###############################################################################

# IMPORT THE TRAINED MODELS

# load the aponeurosis model
model_apo = load_model('./models/model-apo2-nc.h5', custom_objects={'IoU': IoU})

# load the fascicle model
modelF = load_model('./models/model-fascSnippets2-nc.h5', custom_objects={'IoU': IoU})

# DEFINE THE PATH OF YOUR VIDEO
vpath = r'C:\Gymlab\Python\image_tracking\data\20210520 181442_Timo_DJ_30_20kbps.mp4'
# vpath = 'D:/Unet annotations/UltratrackCompVideos/MG_MVC.avi'

# Video properties (do not edit)
cap = cv2.VideoCapture(vpath)
vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
count = 0
dataout = []
indices = [i for i, a in enumerate(vpath) if a == '/']
dots = [i for i, a in enumerate(vpath) if a == '.']
filename = './analysedVideos/' + vpath.split("\\")[-1] # indices
vid_out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'DIVX'), vid_fps, (vid_width, vid_height))
calibDist = []

# Define settings
apo_threshold = 0.3                    # Sensitivity threshold for detecting aponeuroses
fasc_threshold = 0.05                   # Sensitivity threshold for detecting fascicles
fasc_cont_thresh = 40                   # Minimum accepted contour length for fascicles (px)
flip = 0                                # If fascicles are oriented bottom-left to top-right, leave as 0. Otherwise set to 1
min_width = 40                          # Minimum acceptable distance between aponeuroses
curvature = 1                           # Set to 3 for curved fascicles or 1 for a straight line
min_pennation = 10                      # Minimum and maximum acceptable pennation angles
max_pennation = 40

# OPTIONAL
# Calibrate the analysis by clicking on 2 points in the image, followed by the 'q' key. These two points should be 1cm apart
# Alternatively, change the spacing setting below
# NOTE: Here we assume that the points are spaced apart in the y/vertical direction of the image
spacing = 10.0  # Space between the two calibration markers (mm)
mlocs = []
for cal in range(0, 1):
    _, frame = cap.read()

    # display the image and wait for a keypress
    cv2.imshow("image", frame)
    cv2.setMouseCallback("image", mclick)
    key = cv2.waitKey(0)

    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()

    calibDist.append(np.abs(mlocs[0] - mlocs[1]))
    print(str(spacing) + ' mm corresponds to ' + str(calibDist[0]) + ' pixels')

# Analyse whole video
fasc_l_all = []
pennation_all = []
x_lows_all = []
x_highs_all = []
thickness_all = []

for a in range(0, vid_len - 1):
    # for a in range(0, 10):

    # FORMAT EACH FRAME, RESHAPE AND COMPUTE NN PREDICTIONS
    _, frame = cap.read()
    img = img_to_array(frame)
    if flip == 1:
        img = np.fliplr(img)
    img_orig = img  # Make a copy
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    w = img.shape[1]
    img = np.reshape(img, [-1, h, w, 1])
    img = resize(img, (1, 512, 512, 1), mode='constant', preserve_range=True)
    img = img / 255.0

    pred_apo = model_apo.predict(img)
    pred_apo_t = (pred_apo > apo_threshold).astype(np.uint8)
    pred_fasc = modelF.predict(img)
    pred_fasc_t = (pred_fasc > fasc_threshold).astype(np.uint8)

    img = resize(img, (1, h, w, 1))
    img = np.reshape(img, (h, w))
    pred_apo = resize(pred_apo, (1, h, w, 1))
    pred_apo = np.reshape(pred_apo, (h, w))
    pred_apo_t = resize(pred_apo_t, (1, h, w, 1))
    pred_apo_t = np.reshape(pred_apo_t, (h, w))
    pred_fasc = resize(pred_fasc, (1, h, w, 1))
    pred_fasc = np.reshape(pred_fasc, (h, w))
    pred_fasc_t = resize(pred_fasc_t, (1, h, w, 1))
    pred_fasc_t = np.reshape(pred_fasc_t, (h, w))

    #############################################

    # COMPUTE CONTOURS TO IDENTIFY THE APONEUROSES
    _, thresh = cv2.threshold(pred_apo_t, 0, 255,
                              cv2.THRESH_BINARY)  # Or set lower threshold above and use pred_ind_t here
    thresh = thresh.astype('uint8')
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #     contours_re = []
    #     for contour in contours: # Remove any contours that are very small
    #         if len(contour) > 600:
    #             contours_re.append(contour)
    contours = [i for i in contours if len(i) > 600]
    #     contours = contours_re
    contours, _ = sort_contours(contours)  # Sort contours from top to bottom

    #     mask_apo = np.zeros(thresh.shape,np.uint8)
    contours_re2 = []
    for contour in contours:
        #         cv2.drawContours(mask_apo,[contour],0,255,-1)
        pts = list(contour)
        ptsT = sorted(pts, key=lambda k: [k[0][0], k[0][1]])  # Sort each contour based on x values
        allx = []
        ally = []
        for aa in range(0, len(ptsT)):
            allx.append(ptsT[aa][0, 0])
            ally.append(ptsT[aa][0, 1])
        app = np.array(list(zip(allx, ally)))
        contours_re2.append(app)

    # Merge nearby contours
    #     countU = 0
    xs1 = []
    xs2 = []
    ys1 = []
    ys2 = []
    maskT = np.zeros(thresh.shape, np.uint8)
    for cnt in contours_re2:
        ys1.append(cnt[0][1])
        ys2.append(cnt[-1][1])
        xs1.append(cnt[0][0])
        xs2.append(cnt[-1][0])
        cv2.drawContours(maskT, [cnt], 0, 255, -1)

    for countU in range(0, len(contours_re2) - 1):
        if xs1[countU + 1] > xs2[countU]:  # Check if x of contour2 is higher than x of contour 1
            y1 = ys2[countU]
            y2 = ys1[countU + 1]
            if y1 - 10 <= y2 <= y1 + 10:
                m = np.vstack((contours_re2[countU], contours_re2[countU + 1]))
                cv2.drawContours(maskT, [m], 0, 255, -1)
        countU += 1

    maskT[maskT > 0] = 1
    skeleton = skeletonize(maskT).astype(np.uint8)
    kernel = np.ones((3, 7), np.uint8)
    dilate = cv2.dilate(skeleton, kernel, iterations=15)
    erode = cv2.erode(dilate, kernel, iterations=10)

    contoursE, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_apoE = np.zeros(thresh.shape, np.uint8)
    #     contours_reE = []
    #     for contour in contoursE: # Remove any contours that are very small
    #         if len(contour) > 600:
    #             contours_reE.append(contour)
    #     contoursE = contours_reE
    contoursE = [i for i in contoursE if len(i) > 600]
    for contour in contoursE:
        cv2.drawContours(mask_apoE, [contour], 0, 255, -1)
    contoursE, _ = sort_contours(contoursE)

    if len(contoursE) >= 2:

        # Get the x,y coordinates of the upper/lower edge of the 2 aponeuroses
        # NOTE: THE APONEUROSES DISPLAYED IN THE IMAGES ARE FROM mask_apoE, NOT THE CONTOUR EDGES BELOW
        upp_x, upp_y = contour_edge('B', contoursE[0])
        if contoursE[1][0, 0, 1] > (contoursE[0][0, 0, 1] + min_width):
            low_x, low_y = contour_edge('T', contoursE[1])
        else:
            low_x, low_y = contour_edge('T', contoursE[2])

        upp_y_new = savgol_filter(upp_y, 81, 2)  # window size, polynomial order
        low_y_new = savgol_filter(low_y, 81, 2)

        # Make a binary mask to only include fascicles within the region between the 2 aponeuroses
        ex_mask = np.zeros(thresh.shape, np.uint8)
        ex_1 = 0
        ex_2 = np.minimum(len(low_x), len(upp_x))
        for ii in range(ex_1, ex_2):
            ymin = int(np.floor(upp_y_new[ii]))
            ymax = int(np.ceil(low_y_new[ii]))

            ex_mask[:ymin, ii] = 0
            ex_mask[ymax:, ii] = 0
            ex_mask[ymin:ymax, ii] = 255

        # Calculate slope of central portion of each aponeurosis & use this to compute muscle thickness
        Alist = list(set(upp_x).intersection(low_x))
        Alist = sorted(Alist)
        Alen = len(list(set(upp_x).intersection(low_x)))  # How many values overlap between x-axes
        A1 = int(Alist[0] + (.33 * Alen))
        A2 = int(Alist[0] + (.66 * Alen))
        mid = int((A2 - A1) / 2 + A1)
        mindist = 10000
        upp_ind = np.where(upp_x == mid)

        if upp_ind == len(upp_x):
            upp_ind -= 1

        for val in range(A1, A2):
            if val >= len(low_x):
                continue
            else:
                dist = distFunc(upp_x[upp_ind], upp_y_new[upp_ind], low_x[val], low_y_new[val])
                if dist < mindist:
                    mindist = dist

        # Add aponeuroses to a mask for display
        imgT = np.zeros((h, w, 3), np.uint8)

        # Compute functions to approximate the shape of the aponeuroses
        zUA = np.polyfit(upp_x, upp_y_new, 2)  # 2nd order polynomial
        g = np.poly1d(zUA)
        zLA = np.polyfit(low_x, low_y_new, 2)
        h = np.poly1d(zLA)

        mid = (low_x[-1] - low_x[0]) / 2 + low_x[0]  # Find middle of the aponeurosis
        x1 = np.linspace(low_x[0] - 700, low_x[-1] + 700,
                         10000)  # Extrapolate polynomial fits to either side of the mid-point
        y_UA = g(x1)
        y_LA = h(x1)

        new_X_UA = np.linspace(mid - 700, mid + 700, 5000)  # Extrapolate x,y data using f function
        new_Y_UA = g(new_X_UA)
        new_X_LA = np.linspace(mid - 700, mid + 700, 5000)  # Extrapolate x,y data using f function
        new_Y_LA = h(new_X_LA)

        #############################################

        # COMPUTE CONTOURS TO IDENTIFY FASCICLES/FASCICLE ORIENTATION
        _, threshF = cv2.threshold(pred_fasc_t, 0, 255, cv2.THRESH_BINARY)
        threshF = threshF.astype('uint8')
        contoursF, hierarchy = cv2.findContours(threshF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Remove any contours that are very small
        #         contours_re = []
        maskF = np.zeros(threshF.shape, np.uint8)
        for contour in contoursF:  # Remove any contours that are very small
            if len(contour) > fasc_cont_thresh:
                #                 contours_re.append(contour)
                cv2.drawContours(maskF, [contour], 0, 255, -1)

                # Only include fascicles within the region of the 2 aponeuroses
        mask_Fi = maskF & ex_mask
        contoursF2, hierarchy = cv2.findContours(mask_Fi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contoursF3 = []
        for contour in contoursF2:
            if len(contour) > fasc_cont_thresh:
                contoursF3.append(contour)

        xs = []
        ys = []
        fas_ext = []
        fasc_l = []
        pennation = []
        x_low1 = []
        x_high1 = []
        #         counter = 0

        for cnt in contoursF3:
            #         x,y = contour_edge('B', contoursF2[counter])
            x, y = contour_edge('B', cnt)
            if len(x) == 0:
                continue
            z = np.polyfit(np.array(x), np.array(y), 1)
            f = np.poly1d(z)
            newX = np.linspace(-400, w + 400, 5000)  # Extrapolate x,y data using f function
            newY = f(newX)

            # Find intersection between each fascicle and the aponeuroses.
            diffU = newY - new_Y_UA  # Find intersections
            locU = np.where(diffU == min(diffU, key=abs))[0]
            diffL = newY - new_Y_LA
            locL = np.where(diffL == min(diffL, key=abs))[0]

            coordsX = newX[int(locL):int(locU)]
            coordsY = newY[int(locL):int(locU)]

            if locL >= 4950:
                Apoangle = int(np.arctan(
                    (new_Y_LA[locL - 50] - new_Y_LA[locL - 50]) / (new_X_LA[locL] - new_X_LA[locL - 50])) * 180 / np.pi)
            else:
                Apoangle = int(np.arctan((new_Y_LA[locL] - new_Y_LA[locL + 50]) / (
                            new_X_LA[locL + 50] - new_X_LA[locL])) * 180 / np.pi)  # Angle relative to horizontal
            Apoangle = 90 + abs(Apoangle)

            # Don't include fascicles that are completely outside of the field of view or
            # those that don't pass through central 1/3 of the image
            #         if np.sum(coordsX) > 0 and coordsX[-1] > 0 and coordsX[0] < np.maximum(upp_x[-1],low_x[-1]) and coordsX[-1] - coordsX[0] < w:
            if np.sum(coordsX) > 0 and coordsX[-1] > 0 and coordsX[0] < np.maximum(upp_x[-1],
                                                                                   low_x[-1]) and Apoangle != float(
                    'nan'):
                FascAng = float(
                    np.arctan((coordsX[0] - coordsX[-1]) / (new_Y_LA[locL] - new_Y_UA[locU])) * 180 / np.pi) * -1
                ActualAng = Apoangle - FascAng

                if ActualAng <= max_pennation and ActualAng >= min_pennation:  # Don't include 'fascicles' beyond a range of pennation angles
                    length1 = np.sqrt((newX[locU] - newX[locL]) ** 2 + (y_UA[locU] - y_LA[locL]) ** 2)
                    fasc_l.append(length1[0])  # Calculate fascicle length
                    pennation.append(Apoangle - FascAng)
                    x_low1.append(coordsX[0].astype('int32'))
                    x_high1.append(coordsX[-1].astype('int32'))
                    coords = np.array(list(zip(coordsX.astype('int32'), coordsY.astype('int32'))))
                    cv2.polylines(imgT, [coords], False, (20, 15, 200), 3)
        #             counter += 1

        # Store the results for each frame and normalise using scale factor (if calibration was done above)
        try:
            midthick = mindist[0]  # Muscle thickness
        except:
            midthick = mindist

        if 'calibDist' in locals() and len(calibDist) > 0:
            fasc_l = fasc_l / (calibDist[0] / 10)
            midthick = midthick / (calibDist[0] / 10)

    else:
        fasc_l = []
        pennation = []
        x_low1 = []
        x_high1 = []
        imgT = np.zeros((h, w, 3), np.uint8)
        fasc_l.append(float("nan"))
        pennation.append(float("nan"))
        x_low1.append(float("nan"))
        x_high1.append(float("nan"))
        midthick = float("nan")

    fasc_l_all.append(fasc_l)
    pennation_all.append(pennation)
    x_lows_all.append(x_low1)
    x_highs_all.append(x_high1)

    thickness_all.append(midthick)

    #############################################

    # DISPLAY EACH PROCESSED IMAGE AND METRICS

    img_orig[mask_apoE > 0] = (235, 25, 42)

    comb = cv2.addWeighted(img_orig.astype(np.uint8), 1, imgT, 0.8, 0)
    vid_out.write(comb)  # Write each image to video file
    cv2.putText(comb, ('Frame: ' + str(a + 1) + ' of ' + str(vid_len)), (125, 350), cv2.FONT_HERSHEY_DUPLEX, 1,
                (249, 249, 249))
    cv2.putText(comb, ('Pennation angle: ' + str('%.1f' % np.median(pennation_all[-1])) + ' deg'), (125, 410),
                cv2.FONT_HERSHEY_DUPLEX, 1, (249, 249, 249))
    if calibDist:
        cv2.putText(comb, ('Fascicle length: ' + str('%.2f' % np.median(fasc_l_all[-1]) + ' mm')), (125, 380),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (249, 249, 249))
        cv2.putText(comb, ('Thickness at centre: ' + str('%.1f' % thickness_all[-1]) + ' mm'), (125, 440),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (249, 249, 249))
    else:
        cv2.putText(comb, ('Fascicle length: ' + str('%.2f' % np.median(fasc_l_all[-1]) + ' px')), (125, 380),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (249, 249, 249))
        cv2.putText(comb, ('Thickness at centre: ' + str('%.1f' % thickness_all[-1]) + ' px'), (125, 440),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (249, 249, 249))

    cv2.imshow('Analysed image', comb)

    #     count += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to stop the analysis
        break

cap.release()
vid_out.release()
cv2.destroyAllWindows()

fl = np.zeros([len(fasc_l_all),len(max(fasc_l_all, key = lambda x: len(x)))])
pe = np.zeros([len(pennation_all),len(max(pennation_all, key = lambda x: len(x)))])
xl = np.zeros([len(x_lows_all),len(max(x_lows_all, key = lambda x: len(x)))])
xh = np.zeros([len(x_highs_all),len(max(x_highs_all, key = lambda x: len(x)))])

for i,j in enumerate(fasc_l_all):
    fl[i][0:len(j)] = j
fl[fl==0] = np.nan
for i,j in enumerate(pennation_all):
    pe[i][0:len(j)] = j
pe[pe==0] = np.nan
for i,j in enumerate(x_lows_all):
    xl[i][0:len(j)] = j
xl[xl==0] = np.nan
for i,j in enumerate(x_highs_all):
    xh[i][0:len(j)] = j
xh[xh==0] = np.nan

df1 = pd.DataFrame(data=fl)
df2 = pd.DataFrame(data=pe)
df3 = pd.DataFrame(data=xl)
df4 = pd.DataFrame(data=xh)
df5 = pd.DataFrame(data=thickness_all)

# Create a Pandas Excel writer and your xlsx filename (same as the video filename)
# writer = ExcelWriter('./analysedVideos/' + os.path.splitext(filename)[0] + '.xlsx')
writer = ExcelWriter('./analysedVideos/' + vpath[indices[-1]+1:dots[-1]] + '.xlsx')

# Write each dataframe to a different worksheet.
df1.to_excel(writer, sheet_name='Fasc_length')
df2.to_excel(writer, sheet_name='Pennation')
df3.to_excel(writer, sheet_name='X_low')
df4.to_excel(writer, sheet_name='X_high')
df5.to_excel(writer, sheet_name='Thickness')

# Close the Pandas Excel writer and output the Excel file
writer.save()