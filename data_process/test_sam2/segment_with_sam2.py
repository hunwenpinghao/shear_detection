import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu

def preprocess_image(image_path):
    """
    预处理图像，提取白色带状区域和中心线
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 使用OTSU阈值分割提取白色区域
    thresh = threshold_otsu(image)
    binary_mask = image > thresh
    
    # 形态学操作清理噪声
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    return image, binary_mask.astype(bool)

def extract_centerline(binary_mask):
    """
    提取白色带状区域的中心线
    """
    # 使用骨架化提取中心线
    skeleton = skeletonize(binary_mask)
    
    # 找到骨架上的连续点
    skeleton_coords = np.column_stack(np.where(skeleton))
    
    # 按照y坐标排序（从上到下）
    skeleton_coords = skeleton_coords[np.argsort(skeleton_coords[:, 0])]
    
    return skeleton, skeleton_coords

def generate_prompt_points(skeleton_coords, binary_mask, num_points=10):
    """
    生成SAM2的提示点，用于分割左右两部分
    """
    left_points = []
    right_points = []
    
    # 沿着中心线生成左右两侧的提示点
    for i in range(0, len(skeleton_coords), len(skeleton_coords)//num_points):
        if i >= len(skeleton_coords):
            break
            
        center_y, center_x = skeleton_coords[i]
        
        # 在中心线左右两侧寻找白色区域的边界点
        row = binary_mask[center_y, :]
        white_indices = np.where(row)[0]
        
        if len(white_indices) > 0:
            left_boundary = white_indices[0]
            right_boundary = white_indices[-1]
            
            # 在左右边界和中心线之间选择点
            left_offset = int((center_x - left_boundary) * 0.3)
            right_offset = int((right_boundary - center_x) * 0.3)
            
            left_point = [center_x - left_offset, center_y]
            right_point = [center_x + right_offset, center_y]
            
            # 确保点在图像范围内且在白色区域内
            if (left_point[0] >= 0 and left_point[0] < binary_mask.shape[1] and 
                binary_mask[left_point[1], left_point[0]]):
                left_points.append(left_point)
                
            if (right_point[0] >= 0 and right_point[0] < binary_mask.shape[1] and 
                binary_mask[right_point[1], right_point[0]]):
                right_points.append(right_point)
    
    return np.array(left_points), np.array(right_points)

def segment_with_sam2(image_path, model_path="sam2_hiera_large.pt", config="sam2_hiera_l.yaml"):
    """
    使用SAM2进行图像分割
    """
    # 预处理图像
    image, binary_mask = preprocess_image(image_path)
    
    # 提取中心线
    skeleton, skeleton_coords = extract_centerline(binary_mask)
    
    # 生成提示点
    left_points, right_points = generate_prompt_points(skeleton_coords, binary_mask)
    
    # 加载SAM2模型
    sam2_model = build_sam2(config, model_path)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # 读取彩色图像用于SAM2
    image_rgb = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    
    # 设置图像
    predictor.set_image(image_rgb)
    
    # 分割左侧部分
    left_masks, left_scores, left_logits = predictor.predict(
        point_coords=left_points,
        point_labels=np.ones(len(left_points)),  # 所有点都是正样本
        multimask_output=False
    )
    
    # 分割右侧部分
    right_masks, right_scores, right_logits = predictor.predict(
        point_coords=right_points,
        point_labels=np.ones(len(right_points)),  # 所有点都是正样本
        multimask_output=False
    )
    
    return {
        'original_image': image_rgb,
        'binary_mask': binary_mask,
        'skeleton': skeleton,
        'left_mask': left_masks[0],
        'right_mask': right_masks[0],
        'left_points': left_points,
        'right_points': right_points
    }

def refine_masks_with_centerline(results):
    """
    使用中心线优化分割结果，确保左右分割准确
    """
    skeleton = results['skeleton']
    left_mask = results['left_mask'].copy()
    right_mask = results['right_mask'].copy()
    binary_mask = results['binary_mask']
    
    # 创建距离变换用于确定每个像素属于左侧还是右侧
    skeleton_coords = np.column_stack(np.where(skeleton))
    
    for y in range(binary_mask.shape[0]):
        for x in range(binary_mask.shape[1]):
            if binary_mask[y, x]:  # 只处理白色区域
                # 找到最近的骨架点
                distances = np.sqrt((skeleton_coords[:, 0] - y)**2 + (skeleton_coords[:, 1] - x)**2)
                nearest_skeleton_idx = np.argmin(distances)
                nearest_skeleton_point = skeleton_coords[nearest_skeleton_idx]
                
                # 判断当前点相对于最近骨架点的位置（左侧或右侧）
                if x < nearest_skeleton_point[1]:  # 左侧
                    left_mask[y, x] = True
                    right_mask[y, x] = False
                else:  # 右侧
                    left_mask[y, x] = False
                    right_mask[y, x] = True
    
    results['refined_left_mask'] = left_mask
    results['refined_right_mask'] = right_mask
    
    return results

def visualize_results(results, save_path=None):
    """
    可视化分割结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(results['original_image'])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 二值化结果
    axes[0, 1].imshow(results['binary_mask'], cmap='gray')
    axes[0, 1].set_title('Binary Mask')
    axes[0, 1].axis('off')
    
    # 骨架
    axes[0, 2].imshow(results['original_image'])
    axes[0, 2].imshow(results['skeleton'], alpha=0.7, cmap='Reds')
    axes[0, 2].scatter(results['left_points'][:, 0], results['left_points'][:, 1], 
                      c='blue', s=20, label='Left Points')
    axes[0, 2].scatter(results['right_points'][:, 0], results['right_points'][:, 1], 
                      c='green', s=20, label='Right Points')
    axes[0, 2].set_title('Skeleton & Prompt Points')
    axes[0, 2].axis('off')
    axes[0, 2].legend()
    
    # SAM2原始分割结果
    axes[1, 0].imshow(results['original_image'])
    axes[1, 0].imshow(results['left_mask'], alpha=0.5, cmap='Blues')
    axes[1, 0].imshow(results['right_mask'], alpha=0.5, cmap='Greens')
    axes[1, 0].set_title('SAM2 Raw Segmentation')
    axes[1, 0].axis('off')
    
    # 优化后的左侧分割
    axes[1, 1].imshow(results['original_image'])
    axes[1, 1].imshow(results.get('refined_left_mask', results['left_mask']), 
                     alpha=0.7, cmap='Blues')
    axes[1, 1].set_title('Left Segment')
    axes[1, 1].axis('off')
    
    # 优化后的右侧分割
    axes[1, 2].imshow(results['original_image'])
    axes[1, 2].imshow(results.get('refined_right_mask', results['right_mask']), 
                     alpha=0.7, cmap='Greens')
    axes[1, 2].set_title('Right Segment')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main(image_path):
    """
    主函数
    """
    try:
        print("开始处理图像...")
        
        # 执行分割
        results = segment_with_sam2(image_path)
        print("SAM2分割完成")
        
        # 优化分割结果
        results = refine_masks_with_centerline(results)
        print("分割结果优化完成")
        
        # 可视化结果
        visualize_results(results, save_path="segmentation_results.png")
        print("结果已保存到 segmentation_results.png")
        
        return results
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    image_path = "your_image_path.jpg"  # 替换为您的图像路径
    results = main(image_path)