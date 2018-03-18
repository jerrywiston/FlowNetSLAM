import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pyopt

# Load data
xd = np.load("XD.npz")
for key in xd:
    print(key)
    
# Camera parameter
kL = xd["matrixL"]
kR = xd["matrixR"]

fxL = kL[0,0]
fyL = kL[1,1]
cxL = kL[0,2]
cyL = kL[1,2]

fxR = kR[0,0]
fyR = kR[1,1]
cxR = kR[0,2]
cyR = kR[1,2]

flow = xd["flow"]
imgL = xd["imgL"]
imgR = xd["imgR"]

n = 18
print(flow.shape)

# ========================== Point Cloud Handling ==========================
# Depth to pointcloud
def GetPointCloud(img, depth, edgeMask, remapMask, cx, cy, f):
    pointCloud = []
    for i in range(depth.shape[1]):
        for j in range(depth.shape[0]):
            if(edgeMask[i,j]>100 or remapMask[i,j]>1.):
                continue
            y = -(i - cy) / f * depth[i,j]
            x = (j - cx) / f * depth[i,j]
            pointCloud.append([x, y, depth[i,j], img[i,j,2]/255., img[i,j,1]/255., img[i,j,0]/255.])

    pointCloud = np.asarray(pointCloud)
    return pointCloud

def WritePointCloud(filename, pointCloud):
    file = open(filename, "w")
    for j in range(pointCloud.shape[0]):
        file.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(\
            pointCloud[j,0], pointCloud[j,1], pointCloud[j,2],\
            pointCloud[j,3], pointCloud[j,4], pointCloud[j,5]))

# ========================== Calculate Mask ==========================
# Zero padding
def Padding(img, region):
    img_p = np.zeros((img.shape[0]+2*region, img.shape[1]+2*region, 3), dtype=np.float32)
    img_p[region:region+img.shape[0],region:region+img.shape[1]] = img
    return img_p/255.

# Calculate Patch difference
def PatchDiff(img1, img2, region):
    img1p = Padding(img1, region)
    img2p = Padding(img2, region)
    imgDiff = np.zeros(img1.shape[0:2], dtype=np.float32)
    for i in range(imgDiff.shape[0]):
        for j in range(imgDiff.shape[1]):
            d = img1p[i:i+region*2+1,j:j+region*2+1] - img2p[i:i+region*2+1,j:j+region*2+1]
            imgDiff[i,j] = np.sum(d**2)
    return imgDiff

# Calculate Remapping Mask
def GetRemapMask(img1, img2, flow):
    m = np.zeros((512,512,2), dtype=np.float32)
    for i in range(512):
        for j in range(512):
            m[j,i] = np.asarray([i,j])

    rmap = cv2.remap(img2, m[:,:,0] + flow[:,:,0], m[:,:,1] + flow[:,:,1], cv2.INTER_LINEAR)
    mapMask = PatchDiff(rmap, img1, 2)
    return mapMask

# Calculate Edge Mask
def GetEdgeMask(depth, dilated):
    depth_int = np.uint8(depth)
    canny = cv2.Canny(depth_int, 50, 10)
    edgeMask = cv2.dilate(canny, np.ones(dilated))
    return edgeMask
    
# ========================== Pose Estimation ==========================
# Get matching id
def GetRandMatch(flow, edgeMask, remapMask, size):
    count = 0
    pts1 = []
    pts2 = []
    while(True):
        rx = random.randint(0,flow.shape[0]-1)
        ry = random.randint(0,flow.shape[1]-1)
        if(edgeMask[ry,rx]>100 or remapMask[ry,rx]>0.5):
            continue
        pts1.append([rx,ry])
        pts2.append([rx+flow[ry,rx,0], ry+flow[ry,rx,1]])        
        count += 1
        if(count >= size):
            break
    return np.asarray(pts1), np.asarray(pts2)

def drawMatches(img1, pts1, img2, pts2, mask):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1, :cols1] = np.dstack([img1])
    out[:rows2, cols1:] = np.dstack([img2])

    for i in range(pts1.shape[0]):
        if(mask[i] == 0):
            continue
        (x1,y1) = pts1[i]
        (x2,y2) = pts2[i]
        color = tuple(np.random.randint(0,255,3).tolist())
        cv2.circle(out, (int(x1),int(y1)), 4, color, -1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, color, -1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color, 1)

    return out

# ========================== Disparity Optimize ==========================
# TODO

# ========================== Main Program ==========================
if __name__ == '__main__':
    for i in range(4):
        print("[Frame " + str(i) + "]")

        #Calculate Depth
        disp = flow[i,:,:,0]
        depth = fxL*n/abs(disp)

        #Calculate Mask
        edgeMask = GetEdgeMask(depth, (5,5))
        remapMask = GetRemapMask(imgL[i], imgR[i], flow[i])
        
        #Pose Estimation
        pts1, pts2 = GetRandMatch(flow[i,:,:], edgeMask, remapMask, 200)
        funMat, ransacMask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, param1=1)
        print(funMat)
        print(str(np.sum(ransacMask)) + "/200")
        match = drawMatches(imgL[i], pts1, imgR[i], pts2, ransacMask)
 
        #Pointcloud handle
        #pointCloud = GetPointCloud(imgL[i], depth, edgeMask, remapMask, cxL, cyL, fxL)
        #WritePointCloud("pc_" + str(i) + ".txt", pointCloud)

        #CV show
        cv2.imshow("Image", imgL[i])
        cv2.imshow("Remap Mask", remapMask)
        cv2.imshow("Edge Mask", edgeMask)
        cv2.imshow("Depth", depth/np.max(depth))
        cv2.imshow("Matches", match)
        cv2.waitKey(0)