import torch
import torch.nn
from torch.autograd import Variable
import numpy as np

def normalize_axis(x,L):
    return (x-1-(L-1)/2)*2/(L-1)

def unnormalize_axis(x,L):
    return x*(L-1)/2+1+(L-1)/2


class PointTnf(object):
    """

    Class with functions for transforming a set of points with affine/tps transformations

    """

    def __init__(self, tps_grid_size=3, tps_reg_factor=0, use_cuda=True):
        self.use_cuda = use_cuda
        self.tpsTnf = TpsGridGen(grid_size=tps_grid_size,
                                 reg_factor=tps_reg_factor,
                                 use_cuda=self.use_cuda)

    def tpsPointTnf(self, theta, points):
        # points are expected in [B,2,N], where first row is X and second row is Y
        # reshape points for applying Tps transformation
        points = points.unsqueeze(3).transpose(1, 3)
        # apply transformation
        warped_points = self.tpsTnf.apply_transformation(theta, points)
        # undo reshaping
        warped_points = warped_points.transpose(3, 1).squeeze(3)
        return warped_points

    def affPointTnf(self, theta, points):
        theta_mat = theta.view(-1, 2, 3)
        warped_points = torch.bmm(theta_mat[:, :, :2], points)
        warped_points += theta_mat[:, :, 2].unsqueeze(2).expand_as(warped_points)
        return warped_points
def corr_to_matches(corr4d, delta4d=None, k_size=1, do_softmax=False, scale='centered', return_indices=False, invert_matching_direction=False):
    to_cuda = lambda x: x.cuda() if corr4d.is_cuda else x        
    batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()
    
    if scale=='centered':
        XA,YA=np.meshgrid(np.linspace(-1,1,fs2*k_size),np.linspace(-1,1,fs1*k_size))
        XB,YB=np.meshgrid(np.linspace(-1,1,fs4*k_size),np.linspace(-1,1,fs3*k_size))
    elif scale=='positive':
        XA,YA=np.meshgrid(np.linspace(0,1,fs2*k_size),np.linspace(0,1,fs1*k_size))
        XB,YB=np.meshgrid(np.linspace(0,1,fs4*k_size),np.linspace(0,1,fs3*k_size))

    JA,IA=(np.meshgrid(range(fs2), range(fs1)))
    JB,IB=(np.meshgrid(range(fs4), range(fs3)))
    
    XA,YA=Variable(to_cuda(torch.FloatTensor(XA))),Variable(to_cuda(torch.FloatTensor(YA)))
    XB,YB=Variable(to_cuda(torch.FloatTensor(XB))),Variable(to_cuda(torch.FloatTensor(YB)))

    JA,IA=Variable(to_cuda(torch.LongTensor(JA).view(1,-1))),Variable(to_cuda(torch.LongTensor(IA).view(1,-1)))
    JB,IB=Variable(to_cuda(torch.LongTensor(JB).view(1,-1))),Variable(to_cuda(torch.LongTensor(IB).view(1,-1)))
    
    if invert_matching_direction:
        nc_A_Bvec=corr4d.view(batch_size,fs1,fs2,fs3*fs4)

        if do_softmax:
            nc_A_Bvec=torch.nn.functional.softmax(nc_A_Bvec, dim=3)

        match_A_vals,idx_A_Bvec=torch.max(nc_A_Bvec, dim=3)
        score=match_A_vals.view(batch_size, -1)

        iB=IB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size,-1)
        jB=JB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size,-1)
        iA=IA.expand_as(iB)
        jA=JA.expand_as(jB)
        
    else:    
        nc_B_Avec=corr4d.view(batch_size,fs1*fs2,fs3,fs4)  # [batch_idx,k_A,i_B,j_B]
        if do_softmax:
            nc_B_Avec=torch.nn.functional.softmax(nc_B_Avec,dim=1)  # (1, 256, 16, 16) A中每一个点(256)对应一张score(16, 16)表
        # match_B_vals代表B中每一个点(16*16)在A中匹配到的最大分值(0~1)，idx_B_Avec相应位置存储A中该最佳匹配点的索引(0~255)
        match_B_vals,idx_B_Avec=torch.max(nc_B_Avec,dim=1)  # (1, 16, 16)
        score=match_B_vals.view(batch_size,-1)  # (1, 256)  for test, batch_size == 1

        iA=IA.view(-1)[idx_B_Avec.view(-1)].view(batch_size,-1)  # B中每个特征点匹配到的A特征点在A中的行索引(row)
        jA=JA.view(-1)[idx_B_Avec.view(-1)].view(batch_size,-1)  # B中每个特征点匹配到的A特征点在A中的列索引(col)
        iB=IB.expand_as(iA)  # B中每个点所在的行(0~15)
        jB=JB.expand_as(jA)

    if delta4d is not None: # relocalization
        delta_iA,delta_jA,delta_iB,delta_jB = delta4d

        diA=delta_iA.squeeze(0).squeeze(0)[iA.view(-1),jA.view(-1),iB.view(-1),jB.view(-1)]
        djA=delta_jA.squeeze(0).squeeze(0)[iA.view(-1),jA.view(-1),iB.view(-1),jB.view(-1)]        
        diB=delta_iB.squeeze(0).squeeze(0)[iA.view(-1),jA.view(-1),iB.view(-1),jB.view(-1)]
        djB=delta_jB.squeeze(0).squeeze(0)[iA.view(-1),jA.view(-1),iB.view(-1),jB.view(-1)]

        iA=iA*k_size+diA.expand_as(iA)
        jA=jA*k_size+djA.expand_as(jA)
        iB=iB*k_size+diB.expand_as(iB)
        jB=jB*k_size+djB.expand_as(jB)

    xA=XA[iA.view(-1),jA.view(-1)].view(batch_size,-1)  #该位置存着B中相应点匹配到的A特征点的X-col坐标(-1~1)
    yA=YA[iA.view(-1),jA.view(-1)].view(batch_size,-1)
    xB=XB[iB.view(-1),jB.view(-1)].view(batch_size,-1)  # B中每个点的坐标(-1~1)
    yB=YB[iB.view(-1),jB.view(-1)].view(batch_size,-1)
    
    if return_indices:
        return (xA,yA,xB,yB,score,iA,jA,iB,jB)
    else:
        return (xA,yA,xB,yB,score)
            
def nearestNeighPointTnf(matches,target_points_norm):
    xA,yA,xB,yB=matches
    
    # match target points to grid
    deltaX=target_points_norm[:,0,:].unsqueeze(1)-xB.unsqueeze(2)
    deltaY=target_points_norm[:,1,:].unsqueeze(1)-yB.unsqueeze(2)
    distB=torch.sqrt(torch.pow(deltaX,2)+torch.pow(deltaY,2))
    vals,idx=torch.min(distB,dim=1)

    warped_points_x = xA.view(-1)[idx.view(-1)].view(1,1,-1)
    warped_points_y = yA.view(-1)[idx.view(-1)].view(1,1,-1)
    warped_points_norm = torch.cat((warped_points_x,warped_points_y),dim=1)
    return warped_points_norm

def bilinearInterpPointTnf(matches,target_points_norm):
    xA,yA,xB,yB=matches
    
    feature_size=int(np.sqrt(xB.shape[-1]))
    
    b,_,N=target_points_norm.size()

    X_=xB.view(-1)
    Y_=yB.view(-1)

    grid = torch.FloatTensor(np.linspace(-1,1,feature_size)).unsqueeze(0).unsqueeze(2)
    if xB.is_cuda:
        grid=grid.cuda()
    if isinstance(xB,Variable):
        grid=Variable(grid)
        
    x_minus = torch.sum(((target_points_norm[:,0,:]-grid)>0).long(),dim=1,keepdim=True)-1  # index
    x_minus[x_minus<0]=0 # fix edge case
    x_plus = x_minus+1

    y_minus = torch.sum(((target_points_norm[:,1,:]-grid)>0).long(),dim=1,keepdim=True)-1
    y_minus[y_minus<0]=0 # fix edge case
    y_plus = y_minus+1
    # 找到B标注特征点所在网格的四个顶点索引
    toidx = lambda x,y,L: y*L+x

    m_m_idx = toidx(x_minus,y_minus,feature_size)  # coordinate index of B annotation points in (16*16)
    p_p_idx = toidx(x_plus,y_plus,feature_size)
    p_m_idx = toidx(x_plus,y_minus,feature_size)
    m_p_idx = toidx(x_minus,y_plus,feature_size)

    topoint = lambda idx, X, Y: torch.cat((X[idx.view(-1)].view(b,1,N).contiguous(),
                                     Y[idx.view(-1)].view(b,1,N).contiguous()),dim=1)

    P_m_m = topoint(m_m_idx,X_,Y_)
    P_p_p = topoint(p_p_idx,X_,Y_)
    P_p_m = topoint(p_m_idx,X_,Y_)
    P_m_p = topoint(m_p_idx,X_,Y_)
    # 找到B特征点所在网格的四个顶点坐标(-1~1)
    multrows = lambda x: x[:,0,:]*x[:,1,:]

    f_p_p=multrows(torch.abs(target_points_norm-P_m_m))
    f_m_m=multrows(torch.abs(target_points_norm-P_p_p))
    f_m_p=multrows(torch.abs(target_points_norm-P_p_m))
    f_p_m=multrows(torch.abs(target_points_norm-P_m_p))  # 双线性插值找到当前B标注特征点与周围四个网格顶点的权重关系

    Q_m_m = topoint(m_m_idx,xA.view(-1),yA.view(-1))
    Q_p_p = topoint(p_p_idx,xA.view(-1),yA.view(-1))
    Q_p_m = topoint(p_m_idx,xA.view(-1),yA.view(-1))
    Q_m_p = topoint(m_p_idx,xA.view(-1),yA.view(-1))

    warped_points_norm = (Q_m_m*f_m_m+Q_p_p*f_p_p+Q_m_p*f_m_p+Q_p_m*f_p_m)/(f_p_p+f_m_m+f_m_p+f_p_m)  # 权重置1
    return warped_points_norm
    # 详情可查阅浙大TIP2016多视点图像拼接论文中的双线性插值算法部分
    # 概括来讲，该文的标注点warp过程并非是建立了几何变换模型，而是根据其所在网格四个顶点及其相应的匹配点，采用双线性插值
    # 获取当前标注点的warp位置

def PointsToUnitCoords(P,im_size):
    h,w = im_size[:,0],im_size[:,1]
    P_norm = P.clone()
    # normalize Y
    P_norm[:,0,:] = normalize_axis(P[:,0,:],w.unsqueeze(1).expand_as(P[:,0,:]))
    # normalize X
    P_norm[:,1,:] = normalize_axis(P[:,1,:],h.unsqueeze(1).expand_as(P[:,1,:]))
    return P_norm

def PointsToPixelCoords(P,im_size):
    h,w = im_size[:,0],im_size[:,1]
    P_norm = P.clone()
    # normalize Y
    P_norm[:,0,:] = unnormalize_axis(P[:,0,:],w.unsqueeze(1).expand_as(P[:,0,:]))
    # normalize X
    P_norm[:,1,:] = unnormalize_axis(P[:,1,:],h.unsqueeze(1).expand_as(P[:,1,:]))
    return P_norm