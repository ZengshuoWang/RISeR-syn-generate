import os
import random
from pathlib import Path
import argparse
import cv2
import torch
import numpy as np
import copy
import PIL
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance


# -----------------------
# 图像颜色空间变换: 自动对比度
# -----------------------
def Autocontrast(img):
    img_transformed = PIL.ImageOps.autocontrast(img)

    return img_transformed


# ------------------
# 图像颜色空间变换: 反色
# ------------------
def Invert(img):
    img_transformed = PIL.ImageOps.invert(img)

    return img_transformed


# -------------------------
# 图像颜色空间变换: 直方图均衡化
# -------------------------
def Equalize(img):
    img_transformed = PIL.ImageOps.equalize(img)

    return img_transformed


# ------------------------------------
# 图像颜色空间变换: 反转指定阈值范围内的像素点
# ------------------------------------
def Solarize(img):
    value = random.randint(0, 256)
    img_transformed = PIL.ImageOps.solarize(img, value)

    return img_transformed, value


# ------------------------------------------------------------------------------------
# 图像颜色空间变换: 将每个颜色通道上变量bits对应的低(8-bits)个bit置0，变量bits的最大取值范围是[0, 8]
# ------------------------------------------------------------------------------------
def Posterize(img):
    value = random.randint(5, 7)
    img_transformed = PIL.ImageOps.posterize(img, value)

    return img_transformed, value


# --------------------------
# 图像颜色空间变换: 调整图像对比度
# --------------------------
def Contrast(img):
    value = random.choice([0.6, 0.7, 0.8, 0.9,
                           1.1, 1.2, 1.3, 1.4])
    img_transformed = PIL.ImageEnhance.Contrast(img).enhance(value)

    return img_transformed, value


# -----------------------------
# 图像颜色空间变换: 调整图像的色彩均衡
# -----------------------------
def Color(img):
    value = random.choice([0.6, 0.7, 0.8, 0.9,
                           1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    img_transformed = PIL.ImageEnhance.Color(img).enhance(value)

    return img_transformed, value


# --------------------------
# 图像颜色空间变换: 调整图像的亮度
# --------------------------
def Brightness(img):
    value = random.choice([0.6, 0.7, 0.8, 0.9,
                           1.1, 1.2, 1.3, 1.4])
    img_transformed = PIL.ImageEnhance.Brightness(img).enhance(value)

    return img_transformed, value


# --------------------------
# 图像颜色空间变换: 调整图像的锐度
# --------------------------
def Sharpness(img):
    value = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                           1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    img_transformed = PIL.ImageEnhance.Sharpness(img).enhance(value)

    return img_transformed, value


# -------------
# 生成图像的mask
# -------------
def generate_mask(img, bw_thresh=9):
    """
        img --->>> 灰度图
        bw_thresh --->>> 二值化的阈值，int类型
    """
    _, img_bw = cv2.threshold(img, bw_thresh, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = np.argmax(area)

    for j in range(len(contours)):
        if j != max_idx:
            cv2.fillPoly(img_bw, [contours[j]], 0)

    return img_bw


# ---------------
# 生成相机的内参矩阵
# ---------------
def get_IntrinsicCameraMatrix(r, u0, v0, location_camera=57.7, k=45.0, rou=12.0):
    """
        r表示图像中的圆形视网膜区域的半径，由图像的大小决定，单位是像素
        u0表示图像的中心在水平方向的坐标，单位是像素，可以理解为图像水平方向尺寸的一半
        v0表示图像的中心在竖直方向的坐标，单位是像素，可以理解为图像竖直方向尺寸的一半
        location_camera表示相机到世界坐标系原点（也就是眼球中心）的距离，单位是毫米
        k表示相机视野，单位是度，由相机规格而定
        rou表示初始化阶段球体眼睛模型的半径，单位是毫米
    """
    IntrinsicCameraMatrix = np.zeros((3, 3), dtype=float)
    IntrinsicCameraMatrix[0][0] = r*(location_camera+rou*np.cos(k*np.pi/(180*2)))/(rou*np.sin(k*np.pi/(180*2)))
    IntrinsicCameraMatrix[1][1] = IntrinsicCameraMatrix[0][0]
    IntrinsicCameraMatrix[2][2] = 1.0
    IntrinsicCameraMatrix[0][2] = u0
    IntrinsicCameraMatrix[1][2] = v0

    return IntrinsicCameraMatrix


# ---------------------------------------------------------------------
# 射线椭球相交，即建立由2d图像点到3d椭球上的视网膜点的映射模型（加入了四阶径向畸变模型）
# ---------------------------------------------------------------------
def map_2dTo3d(points_2d, K, rvec_camera, tvec_camera, location_camera, rvec_eye, a, b, c, k1, k2):
    """
        points_2d表示视网膜图像中的所有2D坐标点构成的矩阵，每一个2D点用行向量的形式表示，所以其尺寸为 n 行 2 列（必须是float类型的）
        K表示相机的内参矩阵
        rvec_camera表示相机位姿的旋转向量，以弧度为单位，是一个1*3的行向量或3*1的列向量
        tvec_camera表示相机位姿的平移向量，是一个3*1的列向量
        location_camera表示相机在世界坐标系中的位置坐标，是一个3*1的列向量
        rvec_eye表示椭球眼睛模型的旋转向量，即分别绕三个轴旋转的角度，以弧度为单位，是一个1*3的行向量或3*1的列向量，并且每个元素必须是float数据
        a、b、c分别为椭球的半轴长度
        k1、k2是四阶径向畸变模型的两个参数
    """
    points_2d = np.array([points_2d])
    points_2d_undistorted = cv2.undistortPoints(points_2d, K, np.array([k1, k2, 0, 0]), P=K)

    points_2d_undistorted_qici = np.r_[points_2d_undistorted[0].T, np.ones((1, points_2d_undistorted[0].shape[0]))]

    R, _ = cv2.Rodrigues(rvec_camera)
    Rt = np.c_[R, tvec_camera]

    P = np.dot(K, Rt)

    P_jia = np.dot(P.T, np.linalg.inv(np.dot(P, P.T)))

    X1 = np.dot(P_jia, points_2d_undistorted_qici)

    c11 = location_camera[0][0]
    c12 = location_camera[1][0]
    c13 = location_camera[2][0]

    Q, _ = cv2.Rodrigues(rvec_eye)
    A = np.zeros((3, 3), dtype=float)
    A[0][0] = a ** (-2)
    A[1][1] = b ** (-2)
    A[2][2] = c ** (-2)
    QAQ = np.dot(np.dot(Q.T, A), Q)
    y11 = QAQ[0][0]
    y12 = QAQ[0][1]
    y13 = QAQ[0][2]
    y21 = QAQ[1][0]
    y22 = QAQ[1][1]
    y23 = QAQ[1][2]
    y31 = QAQ[2][0]
    y32 = QAQ[2][1]
    y33 = QAQ[2][2]

    aa = (y11 * (c11 ** 2) + y21 * c11 * c12 + y12 * c11 * c12 + y31 * c11 * c13 + y13 * c11 * c13 +
          y22 * (c12 ** 2) + y32 * c12 * c13 + y23 * c12 * c13 + y33 * (c13 ** 2) - 1)
    bb = (2 * y11 * X1[0] * c11 + y21 * X1[0] * c12 + y21 * c11 * X1[1] + y12 * X1[0] * c12 +
          y12 * c11 * X1[1] + y31 * X1[0] * c13 + y31 * c11 * X1[2] + y13 * X1[0] * c13 +
          y13 * c11 * X1[2] + 2 * y22 * X1[1] * c12 + y32 * X1[1] * c13 + y32 * c12 * X1[2] +
          y23 * X1[1] * c13 + y23 * c12 * X1[2] + 2 * y33 * X1[2] * c13 - 2 * X1[3])
    cc = (y11 * (X1[0] * X1[0]) + y21 * X1[0] * X1[1] + y12 * X1[0] * X1[1] + y31 * X1[0] * X1[2] +
          y13 * X1[0] * X1[2] + y22 * (X1[1] * X1[1]) + y32 * X1[1] * X1[2] + y23 * X1[1] * X1[2] +
          y33 * (X1[2] * X1[2]) - X1[3] * X1[3])

    delt = bb * bb - 4 * aa * cc
    delt_jia = np.maximum(delt, 0)

    lam = (-bb - np.sqrt(delt_jia))/(2*aa)
    points_3d_x = (X1[0] + c11*lam)/(X1[3] + lam)
    points_3d_y = (X1[1] + c12*lam)/(X1[3] + lam)
    points_3d_z = (X1[2] + c13*lam)/(X1[3] + lam)

    points_3d_z[delt_jia == 0] = 0

    points_3d = np.r_[np.r_[[points_3d_x], [points_3d_y]], [points_3d_z]].T

    return points_3d


# -----------------------------------------------------
# 建立由3D空间点到2D像素点的映射模型（加入了四阶径向畸变模型）
# -----------------------------------------------------
def map_3dTo2d(points_3d, K, rvec_camera, tvec_camera, k1, k2):
    """
        points_3d表示眼球视网膜上的世界坐标系下的所有3D坐标点坐标构成的矩阵
        K表示相机的内参矩阵
        rvec_camera表示相机位姿的旋转向量，以弧度为单位，是一个1*3的行向量或3*1的列向量
        tvec_camera表示相机位姿的平移向量，是一个3*1的列向量
        k1、k2表示四阶径向畸变模型中的两个参数
    """
    points_3d_qici = np.r_[points_3d.T, np.ones((1, points_3d.shape[0]))]

    R_camera, _ = cv2.Rodrigues(rvec_camera)
    Rt_camera_qici = np.r_[np.c_[R_camera, tvec_camera], [[0, 0, 0, 1]]]

    P_C = np.dot(Rt_camera_qici, points_3d_qici)

    P_C = np.delete(P_C, -1, axis=0)
    P_C_1 = P_C * (1/P_C[2])

    r_2 = P_C_1[0] * P_C_1[0] + P_C_1[1] * P_C_1[1]
    distorted = 1 + k1 * r_2 + k2 * (r_2*r_2)
    P_CJ_1 = P_C_1 * distorted
    P_CJ_1 = np.delete(P_CJ_1, -1, axis=0)
    P_CJ_1 = np.r_[P_CJ_1, np.ones((1, P_CJ_1.shape[1]))]

    P_uv = np.dot(K, P_CJ_1)

    P_uv = np.delete(P_uv, -1, axis=0)
    points_2d = P_uv.T

    return points_2d


# -------------------------------------------------------------------
# 该函数的功能是将参考眼底曲面上的3D点映射到测试眼底曲面上
# -------------------------------------------------------------------
def pts_3D_from_reference_to_test(pts_3D_reference, a_Eye, b_Eye, c_Eye_test, Rvec_Eye):
    """
        pts_3D_reference表示参考眼底曲面上的所有3D点坐标，是一个 n 行 3 列的矩阵
        a_Eye表示眼球的a半轴长度
        b_Eye表示眼球的b半轴长度
        c_Eye_test表示测试眼球的c半轴长度
        Rvec_Eye表示眼球的姿态参数
    """
    vertex_rotate = np.array([0, 0, 0]).reshape((-1, 1))
    x1 = vertex_rotate[0][0]
    y1 = vertex_rotate[1][0]
    z1 = vertex_rotate[2][0]

    Q, _ = cv2.Rodrigues(Rvec_Eye)
    A = np.zeros((3, 3), dtype=float)
    A[0][0] = a_Eye ** (-2)
    A[1][1] = b_Eye ** (-2)
    A[2][2] = c_Eye_test ** (-2)
    QAQ = np.dot(np.dot(Q.T, A), Q)

    mnp = pts_3D_reference - vertex_rotate.T
    m = mnp[:, 0]
    n = mnp[:, 1]
    p = mnp[:, 2]

    t_a = (QAQ[0][0] * (m ** 2) + QAQ[1][1] * (n ** 2) + QAQ[2][2] * (p ** 2) + (QAQ[0][1] + QAQ[1][0]) * m * n +
           (QAQ[1][2] + QAQ[2][1]) * n * p + (QAQ[0][2] + QAQ[2][0]) * p * m)
    t_b = (QAQ[0][0] * 2 * x1 * m + QAQ[1][1] * 2 * y1 * n + QAQ[2][2] * 2 * z1 * p +
           (QAQ[0][1] + QAQ[1][0]) * (y1 * m + x1 * n) + (QAQ[1][2] + QAQ[2][1]) * (z1 * n + y1 * p) +
           (QAQ[0][2] + QAQ[2][0]) * (x1 * p + z1 * m))
    t_c = (QAQ[0][0] * (x1 ** 2) + QAQ[1][1] * (y1 ** 2) + QAQ[2][2] * (z1 ** 2) + (QAQ[0][1] + QAQ[1][0]) * x1 * y1 +
           (QAQ[1][2] + QAQ[2][1]) * y1 * z1 + (QAQ[0][2] + QAQ[2][0]) * z1 * x1 - 1)

    delt = t_b * t_b - 4 * t_a * t_c

    t = (-t_b + np.sqrt(delt)) / (2 * t_a)

    pts_3D_Test_x = m * t + x1
    pts_3D_Test_y = n * t + y1
    pts_3D_Test_z = p * t + z1
    assert pts_3D_Test_z.min() > 0
    pts_3D_Test = np.r_[np.r_[[pts_3D_Test_x], [pts_3D_Test_y]], [pts_3D_Test_z]].T

    return pts_3D_Test


# ---------------
# 回缩mask的边界
# ---------------
def reduce_mask_region(full_masks, reduce_size=1):
    """
        full_masks: (H, W), min 0, max 255
        reduce_size: the reduce size of mask region we want, default 1, generally do not more than 2
    """
    assert (np.min(full_masks) == 0 and np.max(full_masks) == 255)
    full_masks_reduced = copy.deepcopy(full_masks)
    for i in range(reduce_size):
        for u in range(1, full_masks.shape[1]-1):
            for v in range(1, full_masks.shape[0]-1):
                summation = int(full_masks[v-1, u-1]) + int(full_masks[v-1, u]) + int(full_masks[v-1, u+1]) + \
                            int(full_masks[v, u-1]) + int(full_masks[v, u]) + int(full_masks[v, u+1]) + \
                            int(full_masks[v+1, u-1]) + int(full_masks[v+1, u]) + int(full_masks[v+1, u+1])
                if (summation != 255 * 9) and (full_masks[v, u] == 255):
                    full_masks_reduced[v, u] = 0
        full_masks = copy.deepcopy(full_masks_reduced)
    return full_masks_reduced


# --------------------
# 设置待生成浮动图像的mask
# --------------------
def generate_float_image_mask(image_h, image_w, mask_r, delete_h_half=0):
    """
        image_h: 待生成浮动图像的高度
        image_w: 待生成浮动图像的宽度
        mask_r: 待生成浮动图像中mask区域的半径
        delete_h_half: 待生成浮动图像中mask区域距离图像上下边界的高度
    """
    float_image_mask_generated = np.zeros([image_h, image_w], np.uint8)
    cv2.circle(float_image_mask_generated, (int(image_w/2), int(image_h/2)), mask_r, 255, -1)

    if delete_h_half != 0:
        float_image_mask_generated[0:delete_h_half, :] = 0
        float_image_mask_generated[image_h-delete_h_half:image_h, :] = 0

    return float_image_mask_generated


# ---------------------------------
# 根据设置的模型参数和参考图像生成浮动图像
# ---------------------------------
def generate_float_image(float_image_mask, refer_image, K_refer, K_float, rvec_refer_camera, tvec_refer_camera,
                         rvec_eye, a_refer, a_float, b_refer, b_float, c_refer, c_float,
                         rvec_float_camera, tvec_float_camera, location_float_camera,
                         k1_refer, k2_refer, k1_float, k2_float, path_out):
    """
        float_image_mask: 设定的待生成的float图像的mask
        refer_image: 参考图像，即用于合成其它浮动图像的原始图像，颜色顺序为RGB，虽然用cv2进行读取的为BGR，但读取后进行了转换，转换成了RGB
        K_refer: 参考相机的内参矩阵
        K_float: 浮动相机的内参矩阵
        rvec_refer_camera: 参考相机位姿的旋转向量，以弧度为单位，是一个1*3的行向量或3*1的列向量
        tvec_refer_camera: 参考相机位姿的平移向量，是一个3*1的列向量
        rvec_eye: 椭球模型的旋转向量
        a, b, c分别为椭球的半轴长度
        rvec_float_camera: 浮动相机位姿的旋转向量
        tvec_float_camera: 浮动相机位姿的平移向量
        location_float_camera: 浮动相机在世界坐标系中的位置坐标，是一个3*1的列向量
        k1, k2: 相机四阶径向畸变模型中的两个参数
        path_out: 生成的float图像保存的地址
    """
    v_float = np.shape(float_image_mask)[0]
    u_float = np.shape(float_image_mask)[1]
    float_image_generated = np.zeros([v_float, u_float, 3], np.uint8)
    v_refer = np.shape(refer_image)[0]
    u_refer = np.shape(refer_image)[1]

    pts_in_mask = np.where(float_image_mask != 0)
    points_2d_float = np.append(pts_in_mask[1].reshape((-1, 1)), pts_in_mask[0].reshape((-1, 1)), axis=1).astype(float)

    points_3d_float = map_2dTo3d(points_2d_float, K_float, rvec_float_camera, tvec_float_camera,
                                 location_float_camera, rvec_eye, a_float, b_float, c_float,
                                 k1_float, k2_float)
    points_3d_refer = pts_3D_from_reference_to_test(points_3d_float, a_refer, b_refer, c_refer, rvec_eye)
    points_2d_refer = map_3dTo2d(points_3d_refer, K_refer, rvec_refer_camera, tvec_refer_camera, k1_refer, k2_refer)

    points1 = np.floor(points_2d_refer).astype(np.int_)
    points2 = np.c_[np.array([points1[:, 0]]).T, np.array([points1[:, 1] + 1]).T]
    points3 = np.c_[np.array([points1[:, 0] + 1]).T, np.array([points1[:, 1]]).T]
    points4 = np.c_[np.array([points1[:, 0] + 1]).T, np.array([points1[:, 1] + 1]).T]
    delt_u = points_2d_refer[:, 0] - points1[:, 0]
    delt_v = points_2d_refer[:, 1] - points1[:, 1]

    for j in range(points_2d_float.shape[0]):
        if (points1[j][0] < 0) or (points1[j][1] < 0):
            float_image_generated[int(points_2d_float[j][1]), int(points_2d_float[j][0]), :] = 0
        elif (points1[j][0] > u_refer-2) or (points1[j][1] > v_refer-2):
            float_image_generated[int(points_2d_float[j][1]), int(points_2d_float[j][0]), :] = 0
        else:
            float_image_generated[int(points_2d_float[j][1]), int(points_2d_float[j][0]), :] = \
                (1-delt_u[j])*(1-delt_v[j])*refer_image[points1[j][1], points1[j][0], :] + \
                (1-delt_u[j])*delt_v[j]*refer_image[points2[j][1], points2[j][0], :] + \
                (1-delt_v[j])*delt_u[j]*refer_image[points3[j][1], points3[j][0], :] + \
                delt_u[j]*delt_v[j]*refer_image[points4[j][1], points4[j][0], :]

    float_image_generated_BGR = cv2.cvtColor(float_image_generated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path_out, float_image_generated_BGR)

    return


torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--refer_image_dir', type=str, default='given-retinal-images/',
        help='Path to an image directory. '
    )
    parser.add_argument(
        '--refer_image_name', type=str, default='retinal-image-10.jpg',
        help='The reference image name we choose. '
    )
    parser.add_argument(
        '--output_dir', type=str, default='code-prints-for-generating-RISeR-syn',
        help='Directory where to write output images (If None, no output)'
    )
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.'
    )

    opt = parser.parse_args()
    print(opt)

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    # 读取参考图像
    refer_image_path = os.path.join(opt.refer_image_dir, opt.refer_image_name)
    refer_image_gray = cv2.imread(refer_image_path, 0)
    refer_image_BGR = cv2.imread(refer_image_path)
    refer_image_RGB = cv2.cvtColor(refer_image_BGR, cv2.COLOR_BGR2RGB)
    refer_image_mask = generate_mask(refer_image_gray)

    # 计算参考相机的内参矩阵
    v_refer, u_refer = np.shape(refer_image_mask)
    u0_refer = u_refer / 2
    v0_refer = v_refer / 2
    d = len(np.where(refer_image_mask[int(v0_refer), :] != 0)[0])
    r_refer = d / 2
    K_refer = get_IntrinsicCameraMatrix(r_refer, u0_refer, v0_refer)

    # 生成2D控制点（位于mask区域内的间隔20个像素的点阵列），尺寸为n行2列
    control_points_refer = None
    first = True
    refer_image_mask_reduced = reduce_mask_region(refer_image_mask, 4)
    for i in range(0, np.shape(refer_image_mask_reduced)[0], 20):
        points_in_mask_row = np.where(refer_image_mask_reduced[i, :] != 0)
        if len(points_in_mask_row[0]) != 0:
            control_points_refer_u = points_in_mask_row[0][0:len(points_in_mask_row[0]):20].reshape((-1, 1))
            control_points_refer_v = np.full(np.shape(control_points_refer_u)[0], i).reshape((-1, 1))
            control_points_refer_sub = np.append(control_points_refer_u, control_points_refer_v, axis=1).astype(float)
            if first:
                control_points_refer = copy.deepcopy(control_points_refer_sub)
                first = False
            else:
                control_points_refer = np.append(control_points_refer, control_points_refer_sub, axis=0)
    for i in range(np.shape(control_points_refer)[0]):
        cv2.circle(refer_image_BGR,
                   (round(control_points_refer[i, 0]), round(control_points_refer[i, 1])),
                   3, (0, 255, 0), -1)

    # 对模型中固定或已知不变的部分参数进行设置
    rvec_camera_refer = np.array([0.0, 0.0, 0.0]).reshape((1, -1))
    tvec_camera_refer = np.array([0.0, 0.0, 57.7]).reshape((-1, 1))
    R_camera_refer, _ = cv2.Rodrigues(rvec_camera_refer)
    location_camera_refer = -np.dot(np.linalg.inv(R_camera_refer), tvec_camera_refer)
    assert location_camera_refer[0, 0] == 0.0
    assert location_camera_refer[1, 0] == 0.0
    assert location_camera_refer[2, 0] == -57.7
    k1_camera_refer = -0.5623
    k2_camera_refer = 0.3317
    a_eye_refer = 11.9
    b_eye_refer = 11.2
    c_eye_refer = 11.9

    rvec_eye = np.array([0.0, 0.0, 0.0]).reshape((1, -1))

    # 循环随机设置其它模型参数，并据此生成相应的float图像
    control_points_float_list = []
    generate_float_image_number = 6
    refer_image_RGB_transformed = None
    img_transform_function = ['Autocontrast', 'Invert', 'Equalize',
                              'Solarize', 'Posterize', 'Contrast',
                              'Color', 'Brightness', 'Sharpness']
    img_transform_function_index = [5, 6, 7]  # 选择想要用到的颜色变换操作，对应上边的img_transform_function的操作索引序号
    refer_image_RGB_PIL = PIL.Image.fromarray(refer_image_RGB)
    refer_image_RGB_transformed_1_PIL = None
    refer_image_RGB_transformed_2_PIL = None
    # 几种可选择的float相机及图像参数
    float_camera_image_parameters = [
        {'float_image_h': 768, 'float_image_w': 960, 'float_image_mask_r': 460, 'float_mask_delete_h_half': 0},
        {'float_image_h': 768, 'float_image_w': 1152, 'float_image_mask_r': 460, 'float_mask_delete_h_half': 30},
        {'float_image_h': 768, 'float_image_w': 1280, 'float_image_mask_r': 480, 'float_mask_delete_h_half': 30},
        {'float_image_h': 640, 'float_image_w': 896, 'float_image_mask_r': 320, 'float_mask_delete_h_half': 0}
    ]
    float_camera_image_parameters_index = [0, 1, 2, 3]  # 对应上边的float_camera_image_parameters的索引序号
    generate_float_image_parameters = copy.deepcopy(float_camera_image_parameters_index)
    for i in range(generate_float_image_number - len(float_camera_image_parameters_index)):
        generate_float_image_parameters.append(random.choice(float_camera_image_parameters_index))
    random.shuffle(generate_float_image_parameters)
    float_image_h_list = []
    float_image_w_list = []
    float_image_mask_list = []
    for i in range(generate_float_image_number):
        float_image_h = float_camera_image_parameters[generate_float_image_parameters[i]].get('float_image_h')
        float_image_h_list.append(float_image_h)
        float_image_w = float_camera_image_parameters[generate_float_image_parameters[i]].get('float_image_w')
        float_image_w_list.append(float_image_w)
        float_image_mask_r = float_camera_image_parameters[generate_float_image_parameters[i]].get('float_image_mask_r')
        float_mask_delete_h_half = float_camera_image_parameters[
            generate_float_image_parameters[i]
        ].get('float_mask_delete_h_half')
        float_image_mask = generate_float_image_mask(float_image_h, float_image_w,
                                                     float_image_mask_r, float_mask_delete_h_half)
        float_image_mask_list.append(float_image_mask)

        # 计算float相机的内参矩阵
        v_float, u_float = np.shape(float_image_mask)
        u0_float = u_float / 2
        v0_float = v_float / 2
        d = len(np.where(float_image_mask[int(v0_float), :] != 0)[0])
        r_float = d / 2
        K_float = get_IntrinsicCameraMatrix(r_float, u0_float, v0_float)

        rvec_camera_float = np.array([random.uniform(0.0 - 0.05, 0.0 + 0.05),
                                      random.uniform(0.0 - 0.05, 0.0 + 0.05),
                                      random.uniform(0.0 - 0.05, 0.0 + 0.05)]).reshape((1, -1))
        tvec_camera_float = np.array([random.uniform(0.0 - 0.25, 0.0 + 0.25),
                                      random.uniform(0.0 - 0.25, 0.0 + 0.25),
                                      random.uniform(44.7 - 2.5, 44.7 + 0.5)]).reshape((-1, 1))
        print("\nrvec_camera_float_" + "{:02d}".format(i + 1) + ': ')
        print(*rvec_camera_float[0], sep=', ')
        print("\ntvec_camera_float_" + "{:02d}".format(i + 1) + ': ')
        print(*tvec_camera_float[:, 0], sep=', ')
        R_camera_float, _ = cv2.Rodrigues(rvec_camera_float)
        location_camera_float = -np.dot(np.linalg.inv(R_camera_float), tvec_camera_float)

        k1_camera_float = random.uniform(-0.5623 - 0.3, -0.5623 + 0.3)
        print("\nk1_camera_float_" + "{:02d}".format(i + 1) + ': ')
        print(k1_camera_float)
        k2_camera_float = random.uniform(0.3317 - 0.3, 0.3317 + 0.3)
        print("\nk2_camera_float_" + "{:02d}".format(i + 1) + ': ')
        print(k2_camera_float)

        a_eye_float = random.randint(120 - 10, 120 + 20) / 10
        print("\na_eye_float_" + "{:02d}".format(i + 1) + ': ')
        print(a_eye_float)
        b_eye_float = random.randint(120 - 10, 120 + 20) / 10
        print("\nb_eye_float_" + "{:02d}".format(i + 1) + ': ')
        print(b_eye_float)
        c_eye_float = random.randint(120 - 10, 120 + 20) / 10
        print("\nc_eye_float_" + "{:02d}".format(i + 1) + ': ')
        print(c_eye_float)

        if i == 0:
            refer_image_RGB_transformed = copy.deepcopy(refer_image_RGB)
        else:
            transform_index_1 = random.choice(img_transform_function_index)
            if transform_index_1 == 0:
                assert img_transform_function[transform_index_1] == 'Autocontrast'
                refer_image_RGB_transformed_1_PIL = Autocontrast(refer_image_RGB_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_1: ')
                print(img_transform_function[transform_index_1])
            elif transform_index_1 == 1:
                assert img_transform_function[transform_index_1] == 'Invert'
                refer_image_RGB_transformed_1_PIL = Invert(refer_image_RGB_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_1: ')
                print(img_transform_function[transform_index_1])
            elif transform_index_1 == 2:
                assert img_transform_function[transform_index_1] == 'Equalize'
                refer_image_RGB_transformed_1_PIL = Equalize(refer_image_RGB_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_1: ')
                print(img_transform_function[transform_index_1])
            elif transform_index_1 == 3:
                assert img_transform_function[transform_index_1] == 'Solarize'
                refer_image_RGB_transformed_1_PIL, value_1 = Solarize(refer_image_RGB_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_1: ')
                print(img_transform_function[transform_index_1])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_1: ')
                print(value_1)
            elif transform_index_1 == 4:
                assert img_transform_function[transform_index_1] == 'Posterize'
                refer_image_RGB_transformed_1_PIL, value_1 = Posterize(refer_image_RGB_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_1: ')
                print(img_transform_function[transform_index_1])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_1: ')
                print(value_1)
            elif transform_index_1 == 5:
                assert img_transform_function[transform_index_1] == 'Contrast'
                refer_image_RGB_transformed_1_PIL, value_1 = Contrast(refer_image_RGB_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_1: ')
                print(img_transform_function[transform_index_1])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_1: ')
                print(value_1)
            elif transform_index_1 == 6:
                assert img_transform_function[transform_index_1] == 'Color'
                refer_image_RGB_transformed_1_PIL, value_1 = Color(refer_image_RGB_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_1: ')
                print(img_transform_function[transform_index_1])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_1: ')
                print(value_1)
            elif transform_index_1 == 7:
                assert img_transform_function[transform_index_1] == 'Brightness'
                refer_image_RGB_transformed_1_PIL, value_1 = Brightness(refer_image_RGB_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_1: ')
                print(img_transform_function[transform_index_1])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_1: ')
                print(value_1)
            elif transform_index_1 == 8:
                assert img_transform_function[transform_index_1] == 'Sharpness'
                refer_image_RGB_transformed_1_PIL, value_1 = Sharpness(refer_image_RGB_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_1: ')
                print(img_transform_function[transform_index_1])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_1: ')
                print(value_1)

            transform_index_2 = random.choice(img_transform_function_index)
            while transform_index_2 == transform_index_1:
                transform_index_2 = random.choice(img_transform_function_index)
            if transform_index_2 == 0:
                assert img_transform_function[transform_index_2] == 'Autocontrast'
                refer_image_RGB_transformed_2_PIL = Autocontrast(refer_image_RGB_transformed_1_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_2: ')
                print(img_transform_function[transform_index_2])
            elif transform_index_2 == 1:
                assert img_transform_function[transform_index_2] == 'Invert'
                refer_image_RGB_transformed_2_PIL = Invert(refer_image_RGB_transformed_1_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_2: ')
                print(img_transform_function[transform_index_2])
            elif transform_index_2 == 2:
                assert img_transform_function[transform_index_2] == 'Equalize'
                refer_image_RGB_transformed_2_PIL = Equalize(refer_image_RGB_transformed_1_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_2: ')
                print(img_transform_function[transform_index_2])
            elif transform_index_2 == 3:
                assert img_transform_function[transform_index_2] == 'Solarize'
                refer_image_RGB_transformed_2_PIL, value_2 = Solarize(refer_image_RGB_transformed_1_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_2: ')
                print(img_transform_function[transform_index_2])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_2: ')
                print(value_2)
            elif transform_index_2 == 4:
                assert img_transform_function[transform_index_2] == 'Posterize'
                refer_image_RGB_transformed_2_PIL, value_2 = Posterize(refer_image_RGB_transformed_1_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_2: ')
                print(img_transform_function[transform_index_2])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_2: ')
                print(value_2)
            elif transform_index_2 == 5:
                assert img_transform_function[transform_index_2] == 'Contrast'
                refer_image_RGB_transformed_2_PIL, value_2 = Contrast(refer_image_RGB_transformed_1_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_2: ')
                print(img_transform_function[transform_index_2])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_2: ')
                print(value_2)
            elif transform_index_2 == 6:
                assert img_transform_function[transform_index_2] == 'Color'
                refer_image_RGB_transformed_2_PIL, value_2 = Color(refer_image_RGB_transformed_1_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_2: ')
                print(img_transform_function[transform_index_2])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_2: ')
                print(value_2)
            elif transform_index_2 == 7:
                assert img_transform_function[transform_index_2] == 'Brightness'
                refer_image_RGB_transformed_2_PIL, value_2 = Brightness(refer_image_RGB_transformed_1_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_2: ')
                print(img_transform_function[transform_index_2])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_2: ')
                print(value_2)
            elif transform_index_2 == 8:
                assert img_transform_function[transform_index_2] == 'Sharpness'
                refer_image_RGB_transformed_2_PIL, value_2 = Sharpness(refer_image_RGB_transformed_1_PIL)
                print('\nimage_' + "{:02d}".format(i + 1) + '_transform_function_2: ')
                print(img_transform_function[transform_index_2])
                print('image_' + "{:02d}".format(i + 1) + '_transform_function_value_2: ')
                print(value_2)

            refer_image_RGB_transformed = np.asarray(refer_image_RGB_transformed_2_PIL)

        path_float_image = str(Path(opt.output_dir, "{:02d}".format(i + 1) + '.jpg'))
        generate_float_image(float_image_mask, refer_image_RGB_transformed, K_refer, K_float,
                             rvec_camera_refer, tvec_camera_refer, rvec_eye, a_eye_refer, a_eye_float,
                             b_eye_refer, b_eye_float, c_eye_refer, c_eye_float,
                             rvec_camera_float, tvec_camera_float, location_camera_float,
                             k1_camera_refer, k2_camera_refer, k1_camera_float, k2_camera_float, path_float_image)

        # 利用构建好的模型对参考图像上的控制点进行变换
        control_points_3d_refer = map_2dTo3d(control_points_refer, K_refer, rvec_camera_refer, tvec_camera_refer,
                                             location_camera_refer, rvec_eye, a_eye_refer, b_eye_refer, c_eye_refer,
                                             k1_camera_refer, k2_camera_refer)
        control_points_3d_float = pts_3D_from_reference_to_test(control_points_3d_refer, a_eye_float, b_eye_float,
                                                                c_eye_float, rvec_eye)
        control_points_2d_float = map_3dTo2d(control_points_3d_float, K_float, rvec_camera_float, tvec_camera_float,
                                             k1_camera_float, k2_camera_float)
        control_points_float_list.append(control_points_2d_float)

    # 提取均在各自mask区域内的control_points，并保存
    assert len(control_points_float_list) == generate_float_image_number
    assert len(float_image_h_list) == generate_float_image_number
    assert len(float_image_w_list) == generate_float_image_number
    assert len(float_image_mask_list) == generate_float_image_number
    delete_r_index = []
    for i in range(np.shape(control_points_refer)[0]):
        for j in range(len(control_points_float_list)):
            if round(control_points_float_list[j][i, 0]) < 0 or round(control_points_float_list[j][i, 1]) < 0:
                delete_r_index.append(i)
                break
            elif ((round(control_points_float_list[j][i, 0]) >= float_image_w_list[j]) or
                  (round(control_points_float_list[j][i, 1]) >= float_image_h_list[j])):
                delete_r_index.append(i)
                break
            elif float_image_mask_list[j][
                round(control_points_float_list[j][i, 1]), round(control_points_float_list[j][i, 0])
            ] == 0:
                delete_r_index.append(i)
                break
    set_delete_r_index = set(delete_r_index)
    assert len(delete_r_index) == len(set_delete_r_index)

    control_points_refer_common = np.delete(control_points_refer, delete_r_index, 0)

    path_control_pts_refer_txt = str(Path(opt.output_dir, "control_points_refer" + '.txt'))
    np.savetxt(path_control_pts_refer_txt, control_points_refer_common, fmt='%.3f')

    for i in range(np.shape(control_points_refer_common)[0]):
        cv2.circle(refer_image_BGR,
                   (round(control_points_refer_common[i, 0]), round(control_points_refer_common[i, 1])),
                   3, (0, 0, 255), -1)
    path_refer_control_points_label = str(Path(opt.output_dir, "label_control_points_refer" + '.jpg'))
    cv2.imwrite(path_refer_control_points_label, refer_image_BGR)

    for i in range(len(control_points_float_list)):
        control_points_float_common = np.delete(control_points_float_list[i], delete_r_index, 0)

        path_control_pts_float_txt = str(Path(opt.output_dir, "control_points_" + "{:02d}".format(i+1) + '.txt'))
        np.savetxt(path_control_pts_float_txt, control_points_float_common, fmt='%.3f')

        path_float_image = str(Path(opt.output_dir, "{:02d}".format(i+1) + '.jpg'))
        float_image_BGR = cv2.imread(path_float_image)
        for j in range(np.shape(control_points_float_common)[0]):
            cv2.circle(float_image_BGR,
                       (round(control_points_float_common[j, 0]), round(control_points_float_common[j, 1])),
                       3, (0, 0, 255), -1)
        path_float_control_points_label = str(Path(opt.output_dir,
                                                   "label_control_points_" + "{:02d}".format(i+1) + '.jpg'))
        cv2.imwrite(path_float_control_points_label, float_image_BGR)
