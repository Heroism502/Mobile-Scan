import os
import sys

# 核心环境配置,防止 OpenGL 报错和 OpenMP 线程冲突
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OMP_NUM_THREADS"] = "8"

# 核心路径配置 
BASE_PATH = "/root/autodl-tmp"
sys.path.append(os.path.join(BASE_PATH, "TripoSR"))

import cv2
import numpy as np
import shutil
import zipfile
import uuid
import time
import json
import redis
import torch
import subprocess
from PIL import Image
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import datetime

# 导入 TripoSR 核心模块
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

app = FastAPI()

# 配置跨域权限（CORS），允许手机 WebView 下载模型
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# ==========================================
# 0. 配置与初始化
# ==========================================
# 连接 Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# 目录路径定义
SAVE_DIR_SINGLE = f"{BASE_PATH}/uploaded_singles"
SAVE_DIR_ZIP = f"{BASE_PATH}/uploaded_zips"
OUT_DIR = f"{BASE_PATH}/outputs"
WORKSPACE_3DGS = f"{BASE_PATH}/3dgs_workspace"

for d in [SAVE_DIR_SINGLE, SAVE_DIR_ZIP, OUT_DIR, WORKSPACE_3DGS]:
    os.makedirs(d, exist_ok=True)

# 挂载静态文件路由，方便手机端下载模型
app.mount("/models", StaticFiles(directory=OUT_DIR), name="models")
app.mount("/thumbs_single", StaticFiles(directory=SAVE_DIR_SINGLE), name="thumbs_single")
app.mount("/workspace", StaticFiles(directory=WORKSPACE_3DGS), name="workspace")

def update_redis_status(task_id, stage, progress, model_url=""):
    status_data = {
        "task_id": task_id,
        "stage": stage,
        "progress": progress,
        "model_url": model_url,
        "last_update": time.time()
    }
    r.set(f"task_status:{task_id}", json.dumps(status_data), ex=86400) # 有效期24小时

# ==========================================
#  基于拉普拉斯方差的图像预清洗模块 
# ==========================================
def filter_blur_images(image_dir: str, threshold: float = 50.0):
    """
    计算目录中所有图片的拉普拉斯方差，低于阈值的视为模糊废片并物理删除。
    同时能拦截并剔除损坏的图片，防止 OpenCV 后续报错。
    """
    print(f"开始进行拉普拉斯模糊检测 (阈值: {threshold})...")
    removed_count = 0
    
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, file)
            img = cv2.imread(img_path)
            
            # 跳过损坏的图片，防止 cvtColor 崩溃
            if img is None:
                print(f"⚠️ 警告：跳过并剔除损坏的图像文件 {file}")
                os.remove(img_path)
                removed_count += 1
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if score < threshold:
                print(f"剔除模糊废片: {file} (清晰度得分: {score:.2f})")
                os.remove(img_path)
                removed_count += 1
                
    print(f"✅ 模糊检测完毕，共剔除 {removed_count} 张废片/损坏图。")
    return removed_count
    
# ==========================================
# 1. 启动提示
# ==========================================
print("✅ 三维扫描重建准备就绪！")

# ==========================================
# 2. 单图生成流水线 
# ==========================================
def process_single_image_task(file_path: str, task_id: str):
    try:
        update_redis_status(task_id, "Waking up AI Engine...", 10)
        
        # 动态分配：任务进来时才占用显存
        print(f" [单图 {task_id}] 正在加载 TripoSR 模型到显卡...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tsr_model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        tsr_model.to(device)

        update_redis_status(task_id, "AI Image Processing...", 30)
        image = Image.open(file_path)
        
        # 预处理：去背景与缩放
        image = remove_background(image, rembg_session=None) 
        image = resize_foreground(image, 0.85)
        
        # 填充白色背景
        white_bg = Image.new("RGBA", image.size, "WHITE")
        white_bg.paste(image, (0, 0), mask=image) 
        image = white_bg.convert("RGB")
        
        update_redis_status(task_id, "Generating 3D Mesh...", 60)
        with torch.no_grad():
            scene_codes = tsr_model(image, device=device)
            mesh = tsr_model.extract_mesh(scene_codes, has_vertex_color=True)[0]
        
        # 坐标系转换 (调整模型朝向)
        mesh.apply_transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        out_filename = f"{task_id}.glb"
        out_path = os.path.join(OUT_DIR, out_filename)
        mesh.export(out_path)
        
        #  动态释放，强制清空显存给下一个任务让路
        print(f" [单图 {task_id}] 任务完成，正在强制释放显存...")
        del tsr_model
        del scene_codes
        import gc
        gc.collect()             # 触发 Python 垃圾回收
        torch.cuda.empty_cache() # 强制 PyTorch 释放底层的 CUDA 显存

        # 完成！
        model_url = f"/models/{out_filename}"
        update_redis_status(task_id, "Generation Success!", 100, model_url)
        print(f"✅ [单图 {task_id}] 处理完毕，显存已安全移交。")

    except Exception as e:
        if 'tsr_model' in locals():
            del tsr_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        update_redis_status(task_id, f"Failed: {str(e)}", -1)
        print(f"❌ [单图 {task_id}] 失败已退出，显存已兜底清空。")

# ==========================================
# 3. 整合 VGGT + Mobile-GS 的多图流水线
# ==========================================
def process_3dgs_pipeline(zip_path: str, task_id: str):
    try:
        workspace_dir = os.path.join(WORKSPACE_3DGS, task_id)
        input_dir = os.path.join(workspace_dir, "input")
        output_model_dir = os.path.join(OUT_DIR, f"3dgs_{task_id}")
        
        # 定义不同算法的虚拟环境 Python 路径
        vggt_python = "/root/miniconda3/envs/vggt/bin/python"
        mobile_gs_python = "/root/miniconda3/envs/mobile-gs/bin/python"
        
        # --- 步骤 1: 解压与清洗 ---
        update_redis_status(task_id, "Preparing Images...", 5)
        
        # 为 VGGT 创建一个 images 子文件夹
        images_dir = os.path.join(input_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 直接解压到 images 文件夹里
            zip_ref.extractall(images_dir) 
            
        # 捞取多层目录里的图片
        for root_path, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if root_path != images_dir:
                        shutil.move(os.path.join(root_path, file), os.path.join(images_dir, file))
                        
        # =================================================================
        # 图像预清洗 ，防止废片拖垮 VGGT
        # =================================================================
        update_redis_status(task_id, "Cleaning Blurry Images...", 15)
        filter_blur_images(images_dir, threshold=50.0)
        
        # 如果剔除废片后，剩下的好图不足 2 张，直接阻断报错
        valid_img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if valid_img_count < 2:
            raise Exception(f"有效图片不足({valid_img_count}张)！由于运动模糊过大，大部分图片已被剔除，请重新拍摄。")
            
        # 步骤 2: 运行 VGGT 获取顶级位姿 
        print(f"🚀 [1/3] 启动 VGGT 智能位姿解算...")
        update_redis_status(task_id, "VGGT Estimating Pose...", 25)
        
        # =================================================================
        # 自适应显存调度策略 (高、中、低三档)
        # =================================================================
        if valid_img_count <= 35:
            # 高精度模式
            max_pts = "2048"
            query_frames = "6"
            print(f"图片数({valid_img_count}张)较少，启动【高精度模式】 (2048pts, 6frames)")
            
        elif valid_img_count <= 50:
            # 均衡模式
            max_pts = "1024"
            query_frames = "4"
            print(f"图片数({valid_img_count}张)适中，启动【均衡模式】 (1024pts, 4frames)")
            
        else:
            # 低显存模式
            max_pts = "512"    # 进一步削减查询点数
            query_frames = "2" # 仅关联最近的2个参考帧
            print(f"⚠️ 图片数({valid_img_count}张)超载！触发【低显存极限模式】 (512pts, 2frames) 以防止系统崩溃！")
            
            if valid_img_count > 120:
                print("❌ 图片数量超过系统极限 (120张)，请精简后重新上传！")

        vggt_dir = os.path.join(BASE_PATH, "vggt")
        
        #  加入环境变量，优化 PyTorch 显存碎片，极大降低 OOM 概率
        env_vars = os.environ.copy()
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"       
        subprocess.run([
            vggt_python, "demo_colmap.py",
            "--scene_dir", input_dir,
            "--use_ba",               # 强制开启 Bundle Adjustment 算出 3D 点
            "--shared_camera",
            "--max_query_pts", max_pts,
            "--query_frame_num", query_frames,
        ], cwd=vggt_dir, env=env_vars, check=True)  # 这里多传了 env=env_vars

        # 步骤 3: 修正路径 
        sparse_src = os.path.join(input_dir, "sparse")
        sparse_zero_dir = os.path.join(input_dir, "sparse", "0")
        
        if os.path.exists(sparse_src) and not os.path.exists(sparse_zero_dir):
            os.makedirs(sparse_zero_dir, exist_ok=True)
            for f in ["cameras.bin", "images.bin", "points3D.bin"]:
                if os.path.exists(os.path.join(sparse_src, f)):
                    shutil.move(os.path.join(sparse_src, f), os.path.join(sparse_zero_dir, f))   
        
        # # 步骤 4: 启动 Mobile-GS 训练 
        mobile_gs_dir = os.path.join(BASE_PATH, "Mobile-GS")
        pretrain_iter = 30000
        train_iter = 30000 
        checkpoint_path = os.path.join(output_model_dir, f"chkpnt{train_iter}.pth")    
        
        # Stage A: Pretrain
        print("🚀 [2/3] 启动 Mobile-GS 预训练...")
        update_redis_status(task_id, "Mobile-GS Pre-training...", 60)
        subprocess.run([
            mobile_gs_python, "pretrain.py",
            "-s", input_dir,
            "-m", output_model_dir,
            "--iterations", str(pretrain_iter),
            "--checkpoint_iterations", str(pretrain_iter),
            "--imp_metric", "indoor",  # 🌟 已修改为 indoor
            "--eval"
        ], cwd=mobile_gs_dir, check=True) 
        
        # Stage B: Train 
        print("🚀 [3/3] 启动 Mobile-GS 压缩...")
        update_redis_status(task_id, "Mobile-GS Distillation...", 85)
        total_iter = pretrain_iter + train_iter
        subprocess.run([
            mobile_gs_python, "train.py",
            "-s", input_dir,
            "-m", output_model_dir,
            "--start_checkpoint", checkpoint_path,
            "--iterations", str(total_iter)
        ], cwd=mobile_gs_dir, check=True)

        # # --- 步骤 5: 交付模型 ---
        print("🚀 模型训练完成，交付模型")
        model_url = f"/models/3dgs_{task_id}/point_cloud/iteration_30000/point_cloud.ply"
        update_redis_status(task_id, "Success!", 100, model_url)
        return output_model_dir

    except Exception as e:
        print(f"❌ 任务失败: {e}")
        update_redis_status(task_id, f"Error: {str(e)}", -1)

# ==========================================
# 4. API 路由
# ==========================================
@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    data = r.get(f"task_status:{task_id}")
    return json.loads(data) if data else {"stage": "Not Found", "progress": 0}

@app.post("/upload_single")
async def upload_single(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = f"single_{uuid.uuid4().hex[:8]}"
    file_path = os.path.join(SAVE_DIR_SINGLE, f"{task_id}.jpg")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    update_redis_status(task_id, "Uploaded", 10)
    background_tasks.add_task(process_single_image_task, file_path, task_id)
    return {"status": "success", "task_id": task_id}

@app.post("/upload_zip")
async def upload_zip(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = file.filename.rsplit('.', 1)[0] + "_" + uuid.uuid4().hex[:4]
    zip_path = os.path.join(SAVE_DIR_ZIP, f"{task_id}.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    update_redis_status(task_id, "Zip Received", 5)
    background_tasks.add_task(process_3dgs_pipeline, zip_path, task_id)
    return {"status": "success", "task_id": task_id}

# ==========================================
#  获取带有真实封面的模型列表
# ==========================================
@app.get("/api/models")
async def get_all_models():
    models_list = []
    if not os.path.exists(OUT_DIR):
        return models_list
        
    for item in os.listdir(OUT_DIR):
        item_path = os.path.join(OUT_DIR, item)
        mtime = os.path.getmtime(item_path)
        date_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        
        # 单图 Mesh 模型
        if os.path.isfile(item_path) and item.startswith("single_") and item.endswith(".glb"):
            # 尝试去 uploaded_singles 里找对应的原图当封面
            thumb_file = item.replace(".glb", ".jpg")
            if os.path.exists(os.path.join(SAVE_DIR_SINGLE, thumb_file)):
                thumb_path = f"/thumbs_single/{thumb_file}"
            else:
                thumb_path = "" # 如果原图没了，留空，前端有渐变色兜底
                
            models_list.append({
                "id": item,
                "title": f"单图扫描_{item[7:13]}", 
                "type": "mesh",
                "date": date_str,
                "thumb": thumb_path,
                "modelUrl": f"/models/{item}"  
            })
            
        # 多图 3DGS 模型
        elif os.path.isdir(item_path) and item.startswith("3dgs_"):
            ply_path_6w = os.path.join(item_path, "point_cloud/iteration_30000/point_cloud.ply")
            ply_path_3w = os.path.join(item_path, "point_cloud/iteration_30000/point_cloud.ply")
            
            final_ply = None
            relative_path = ""
            
            if os.path.exists(ply_path_6w):
                final_ply = ply_path_6w
                relative_path = "point_cloud/iteration_30000/point_cloud.ply"
            elif os.path.exists(ply_path_3w):
                final_ply = ply_path_3w
                relative_path = "point_cloud/iteration_30000/point_cloud.ply"
                
            if final_ply:
                task_id_str = item.replace("3dgs_", "")
                workspace_img_dir = os.path.join(WORKSPACE_3DGS, task_id_str, "input", "images")
                thumb_path = ""
                # 去 workspace 里拿第一张照片当封面
                if os.path.exists(workspace_img_dir):
                    imgs = [f for f in os.listdir(workspace_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if imgs:
                        thumb_path = f"/workspace/{task_id_str}/input/images/{imgs[0]}"

                models_list.append({
                    "id": item,
                    "title": f"多图重建_{item[18:24]}", 
                    "type": "gs",
                    "date": date_str,
                    "thumb": thumb_path,
                    "modelUrl": f"/models/{item}/{relative_path}"
                })

    models_list.sort(key=lambda x: x["date"], reverse=True)
    return models_list

# ==========================================
# 新增彻底删除模型的 API
# ==========================================
@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    try:
        # 安全校验，防止跨目录攻击
        if ".." in model_id or "/" in model_id:
            return {"status": "error", "message": "Invalid ID"}
            
        # 删除 outputs 里的最终模型
        target_path = os.path.join(OUT_DIR, model_id)
        if os.path.exists(target_path):
            if os.path.isdir(target_path):
                shutil.rmtree(target_path) # 删除 3DGS 整个文件夹
            else:
                os.remove(target_path)     # 删除 GLB 单文件
                
        # 深度清理：释放 AutoDL 硬盘空间
        if model_id.startswith("single_"):
            # 清理单图原图
            thumb_path = os.path.join(SAVE_DIR_SINGLE, model_id.replace(".glb", ".jpg"))
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
                
        elif model_id.startswith("3dgs_"):
            # 清理多图的工作流文件夹 
            task_id_str = model_id.replace("3dgs_", "")
            workspace_dir = os.path.join(WORKSPACE_3DGS, task_id_str)
            if os.path.exists(workspace_dir):
                shutil.rmtree(workspace_dir)

        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)