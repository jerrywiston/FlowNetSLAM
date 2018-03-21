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
flow = xd["flow"]
imgL = xd["imgL"]
imgR = xd["imgR"]

fxL, fyL, cxL, cyL = kL[0,0], kL[1,1], kL[0,2], kL[1,2]
fxR, fyR, cxR, cyR = kR[0,0], kR[1,1], kR[0,2], kR[1,2]

n = 18
print(flow.shape)

# ========================== Point Cloud Handling ==========================
# Depth to pointcloud
def GetPointCloud(img, depth, mask, cx, cy, f):
    pointCloud = []
    for i in range(depth.shape[1]):
        for j in range(depth.shape[0]):
            if(mask[i,j] == 0):
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
def GetRemapMask(img1, img2, flow, threshold):
    m = np.zeros((512,512,2), dtype=np.float32)
    for i in range(512):
        for j in range(512):
            m[j,i] = np.asarray([i,j])

    rmap = cv2.remap(img2, m[:,:,0] + flow[:,:,0], m[:,:,1] + flow[:,:,1], cv2.INTER_LINEAR)
    mapMask = PatchDiff(rmap, img1, 2)
    mapMask = (mapMask < threshold).astype(np.float32)
    return mapMask

# Calculate Edge Mask
def GetEdgeMask(depth, dilated):
    depth_int = np.uint8(depth)
    canny = cv2.Canny(depth_int, 50, 10)
    edgeMask = cv2.dilate(canny, np.ones(dilated))
    edgeMask = (np.ones_like(edgeMask) - edgeMask/255).astype(np.float32)
    return edgeMask

# Mask Overlap
def OverlapMask(masks):
    totalMask = np.ones(masks[0].shape, dtype=np.float32)
    for i in range(len(masks)):
        totalMask = totalMask * masks[i]
    return totalMask
    
# ========================== Pose Estimation ==========================
# Get matching id
def GetRandMatch(flow, size, mask):
    count = 0
    pts1 = []
    pts2 = []
    while(True):
        rx = random.randint(0,flow.shape[0]-1)
        ry = random.randint(0,flow.shape[1]-1)
        if(mask[ry,rx] == 0):
            continue
        pts1.append([rx,ry])
        pts2.append([rx+flow[ry,rx,0], ry+flow[ry,rx,1]])        
        count += 1
        if(count >= size):
            break
    return np.asarray(pts1, dtype=np.float32), np.asarray(pts2, dtype=np.float32)

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

def GetExtrinsic(funMat, kL):
    w = np.asarray([[0,-1,0],[1,0,0],[0,0,1]], dtype=np.float32)
    essMat = np.matmul(np.matmul(np.transpose(kL), funMat), kL)
    u,s,vt = np.linalg.svd(essMat)
    R1 = np.matmul(np.matmul(u, w), vt)
    R2 = np.matmul(np.matmul(u, np.transpose(w)), vt)

    sigma = np.zeros((3,3), dtype=np.float32)
    sigma[0,0], sigma[1,1], sigma[2,2] = s[0], s[1], s[2]
    Tx1 = np.matmul(np.matmul(np.matmul(u, np.transpose(w)), sigma), np.transpose(u))
    Tx2 = np.matmul(np.matmul(np.matmul(u, w), sigma), np.transpose(u))

    T1 = np.asarray([-Tx1[2,1], Tx1[2,0], -Tx1[1,0]])
    T2 = np.asarray([-Tx2[2,1], Tx2[2,0], -Tx2[1,0]])

    return R1, R2, T1, T2

def CrossMatrix(v):
    vmat = np.zeros((3,3), dtype=np.float32)
    vmat[1,2], vmat[0,2], vmat[0,1] = -v[0], v[1], -v[2] 
    vmat[2,1], vmat[2,0], vmat[1,0] = v[0], -v[1], v[2] 
    return vmat

def DepthSolver(p1, p2, M1, M2, kL):
    pt1 = np.matmul(np.linalg.inv(kL), np.array([p1[0], p1[1], 1]))
    pt2 = np.matmul(np.linalg.inv(kL), np.array([p2[0], p2[1], 1]))
    A = np.concatenate([np.matmul(CrossMatrix(pt1), M1), np.matmul(CrossMatrix(pt2), M2)], axis=0)
    u,s,vt = np.linalg.svd(A)
    #print(s)
    p3d = vt[3,:] / vt[3,3]
    return p3d

def TransformMatrix(R, T):
    H = np.zeros((4,4), dtype=np.float32)
    H[0:3,0:3] = R
    H[0:3,3] = T
    H[3,3] = 1
    M1 = np.zeros((3,4), dtype=np.float32)
    M1[0,0], M1[1,1], M1[2,2] = 1,1,1
    M2 = np.linalg.inv(H)[0:3,0:4]
    return H, M1, M2

def SelectRT(R1, R2, T1, T2, pts1, pts2, kL, size):
    score = [0,0,0,0]
    for i in range(size):
        r = random.randint(0,pts1.shape[0]-1)
        p1 = pts1[r]
        p2 = pts2[r]
        H, M1, M2 = TransformMatrix(R1, T1)
        p3d1 = DepthSolver(p1, p2, M1, M2, kL)
        p3d2 = np.matmul(H,p3d1)
        if(p3d1[2]>0 and p3d2[2]>0):
            score[0] += 1
        
        H, M1, M2 = TransformMatrix(R2, T1)
        p3d1 = DepthSolver(p1, p2, M1, M2, kL)
        p3d2 = np.matmul(H,p3d1)
        if(p3d1[2]>0 and p3d2[2]>0):
            score[1] += 1
        
        H, M1, M2 = TransformMatrix(R1, T2)
        p3d1 = DepthSolver(p1, p2, M1, M2, kL)
        p3d2 = np.matmul(H,p3d1)
        if(p3d1[2]>0 and p3d2[2]>0):
            score[2] += 1
        
        H, M1, M2 = TransformMatrix(R2, T2)
        p3d1 = DepthSolver(p1, p2, M1, M2, kL)
        p3d2 = np.matmul(H,p3d1)
        if(p3d1[2]>0 and p3d2[2]>0):
            score[3] += 1

    print(score)
    winner = np.argmax(np.asarray(score))
    #print(winner)
    if(winner == 0):
        return R1, T1
    elif(winner == 1):
        return R2, T1
    elif(winner == 2):
        return R1, T2
    else:
        return R2, T2

def Get3dPoints(R, T, pts1, pts2, kL):
    H, M1, M2 = TransformMatrix(R,T)
    pts3d = []
    for i in range(pts1.shape[0]):
        p3d = DepthSolver(pts1[i]+10, pts2[i]+10, M1, M2, kL)
        pts3d.append(p3d)

    return np.asarray(pts3d)

def Get3dPointsImg(R, T, flow, kL, mask):
    H, M1, M2 = TransformMatrix(R,T)
    pimg = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
   
    for i in range(flow.shape[1]):
        for j in range(flow.shape[0]):
            if(mask[j,i] == 0):
                pimg[i,j] = 0
                continue
            print(i,j)
            p1 = np.asarray([j+10,i+10], dtype=np.float32)
            p2 = np.asarray([j+flow[i,j,0]+10, i+flow[i,j,1]+10], dtype=np.float32)
            pimg[i,j] = DepthSolver(p1, p2, M1, M2, kL)[0:3]

    return pimg

# ========================== Disparity Optimize ==========================
# TODO

# ========================== Main Program ==========================
if __name__ == '__main__':
    USE_MASK = False
    for i in range(4):
        print("[Frame " + str(i) + "]")
        img1 = imgL[i]
        img2 = imgR[i]

        #Calculate Depth
        disp = flow[i,:,:,0]
        depth = fxL*n/abs(disp)
        
        #Calculate Mask
        if(USE_MASK == True):
            edgeMask = GetEdgeMask(depth, (5,5))
            remapMask = GetRemapMask(img1, img2, flow[i], 1)
            totalMask = OverlapMask([edgeMask, remapMask])
        else:
            totalMask = np.ones((512,512), dtype=np.float32)
        
        #Pose Estimation
        rnum = 1000
        pts1, pts2 = GetRandMatch(flow[i,:,:], rnum, mask=totalMask)
        funMat, ransacMask = cv2.findFundamentalMat(pts1+10, pts2+10, cv2.FM_RANSAC, param1=1)
        funMask = (np.abs(funMat) > 1e-2).astype(np.float32)
        funMat = funMat * funMask

        match = drawMatches(img1, pts1[0:300], img2, pts2, ransacMask)
        print(str(np.sum(ransacMask)) + "/" + str(rnum))
        R1, R2, T1, T2 = GetExtrinsic(funMat, kL)

        scale = np.linalg.norm(T1)
        R, T = SelectRT(R1, R2, T1, T2, pts1+10, pts2+10, kL, 10)
        P = Get3dPoints(R, T, pts1, pts2, kL)
        print(np.mean(P[:,0:3]*18 / scale, axis=0))
        #pimg = Get3dPointsImg(R, T, flow[i], kL, totalMask)
        #print(np.mean(pimg[:,0:3]*18 / scale, axis=0))
        
        #Pointcloud handle
        #pointCloud = GetPointCloud(img1, depth, totalMask, cxL, cyL, fxL)
        #WritePointCloud("pc_" + str(i) + ".txt", pointCloud)

        #CV show
        cv2.imshow("Image", img1)
        #cv2.imshow("Remap Mask", remapMask)
        #cv2.imshow("Edge Mask", edgeMask)
        cv2.imshow("Total Mask", totalMask)
        cv2.imshow("Depth", depth/np.max(depth))
        cv2.imshow("Matches", match)
        print("< Press Enter to continue ... >")
        cv2.waitKey(0)
        #break