# SAM2 Video Pre-Annotation — Label Studio ML Backend

基于 [SAM2 (Segment Anything Model 2)](https://github.com/facebookresearch/sam2) 的 Label Studio 视频追踪预标注服务。

用户在视频某一帧画追踪框，服务自动将目标追踪到后续帧，大幅提升视频标注效率。

![workflow](https://github.com/HumanSignal/label-studio-ml-backend/raw/master/label_studio_ml/examples/segment_anything_2_video/Sam2Video.gif)

---

## 工作原理

1. 在 Label Studio 中上传视频并打开任务
2. 在某帧用 **VideoRectangle** 追踪框圈住目标（支持多目标）
3. 点击 **Smart Annotation** → 后端自动将目标追踪到后续 N 帧
4. 人工检查并修正，继续向后追踪

> **说明：** 本服务基于官方 [segment_anything_2_video](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_video) 示例，
> 适配了 SAM2.1 Tiny 模型和跨平台部署。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.10 |
| CUDA | >= 11.8（推荐 12.1）|
| GPU | NVIDIA（SAM2 仅支持 GPU）|
| Label Studio | 已部署并可访问 |

---

## 快速开始

### 1. 获取代码

```bash
git clone https://github.com/Jiadezhende/SAM2-pre-annotation.git
cd SAM2-pre-annotation
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入以下两个必填项：

```ini
LABEL_STUDIO_URL=http://<你的LS地址>:8080
LABEL_STUDIO_API_KEY=<在 LS 账户设置中获取的 API Key>
```

---

## Windows 部署（本地开发）

### 前置条件

- Python 3.10+
- NVIDIA GPU + 对应版本的 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Git Bash 或 PowerShell

### 安装步骤

```bash
# 1. 创建并激活虚拟环境
python -m venv venv
source venv/Scripts/activate        # Git Bash
# 或：.\venv\Scripts\Activate.ps1   # PowerShell

# 2. 升级 pip
python -m pip install --upgrade pip

# 3. 安装 PyTorch（CUDA 12.1 版本，根据实际 CUDA 版本调整）
#    查询对应命令：https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. 安装 SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# 5. 安装其余依赖
pip install -r requirements-base.txt -r requirements.txt

# 6. 下载 SAM2.1 Tiny 模型权重（约 150 MB）
python download_models.py --model tiny
```

### 启动服务

```bash
python _wsgi.py
# 服务运行在 http://localhost:9090
```

> **注意：** `_wsgi.py` 会自动检测 Windows 环境并设置 `TORCH_COMPILE_DISABLE=1`，
> 无需手动配置（Windows 不支持 Triton，torch.compile 会报错）。

### 验证

```bash
curl http://localhost:9090/health
# 期望输出：{"status": "UP"}
```

---

## Linux 部署（Docker，推荐生产环境）

### 前置条件

- Docker + Docker Compose
- NVIDIA GPU + 驱动
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# 验证 NVIDIA Container Toolkit 是否正常
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 启动服务

```bash
# 编辑 docker-compose.yml，填入 LABEL_STUDIO_URL 和 LABEL_STUDIO_API_KEY
# 注意：不能使用 localhost，需填写宿主机的实际 IP 地址

docker-compose up --build
# 首次构建需下载依赖和模型权重，约需 5-10 分钟
```

后台运行：

```bash
docker-compose up --build -d
docker-compose logs -f    # 查看日志
docker-compose down       # 停止服务
```

### 验证

```bash
curl http://localhost:9090/health
# 期望输出：{"status": "UP"}
```

---

## 连接 Label Studio

1. 打开 Label Studio 项目 → **Settings → Machine Learning → Add Model**
2. 填写 URL：`http://<运行服务的机器IP>:9090`
   - Windows 本地：`http://localhost:9090`
   - Linux Docker：`http://<宿主机IP>:9090`（不能用 localhost）
3. 点击 **Validate and Save**

### 标注配置（Labeling Interface XML）

在项目 **Settings → Labeling Interface** 中粘贴以下配置（按需修改类别标签）：

```xml
<View>
    <Labels name="videoLabels" toName="video" allowEmpty="true">
        <Label value="Person" background="#11A39E"/>
        <Label value="Car"    background="#D4380D"/>
    </Labels>

    <!-- framerate 需与实际视频帧率一致 -->
    <Video name="video" value="$video" framerate="25.0"/>
    <VideoRectangle name="box" toName="video" smart="true"/>
</View>
```

---

## 模型权重选择

默认使用 **SAM2.1 Tiny**，可根据精度/速度需求切换：

| 模型 | 大小 | 速度 | 精度 | 配置名 |
|------|------|------|------|--------|
| Tiny | ~150 MB | 最快 | 较低 | `sam2.1_hiera_t.yaml` |
| Small | ~200 MB | 快 | 中等 | `sam2.1_hiera_s.yaml` |
| Base+ | ~350 MB | 中 | 较高 | `sam2.1_hiera_b+.yaml` |
| Large | ~900 MB | 慢 | 最高 | `sam2.1_hiera_l.yaml` |

切换方法：

```bash
# 1. 下载对应权重
python download_models.py --model large

# 2. 修改 .env（Windows）或 docker-compose.yml（Linux）
MODEL_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
MODEL_CHECKPOINT=sam2.1_hiera_large.pt
```

---

## 环境变量说明

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LABEL_STUDIO_URL` | — | Label Studio 地址（必填）|
| `LABEL_STUDIO_API_KEY` | — | Label Studio API Key（必填）|
| `MODEL_CONFIG` | `configs/sam2.1/sam2.1_hiera_t.yaml` | SAM2 模型配置文件 |
| `MODEL_CHECKPOINT` | `sam2.1_hiera_tiny.pt` | SAM2 模型权重文件名 |
| `DEVICE` | `cuda` | 推理设备（`cuda` 或 `cpu`）|
| `MAX_FRAMES_TO_TRACK` | `10` | 每次追踪的帧数 |
| `PORT` | `9090` | 服务端口 |
| `LOG_LEVEL` | `DEBUG` | 日志级别 |

---

## 已知限制

- 目前每次追踪仅支持**单个目标**（SAM2 本身支持多目标，后续可扩展）
- 返回 **bounding box**，不返回逐像素分割 mask
- 每次追踪 `MAX_FRAMES_TO_TRACK` 帧后需再次触发

---

## 参考资料

- [Label Studio ML Backend 官方文档](https://labelstud.io/guide/ml)
- [SAM2 官方仓库](https://github.com/facebookresearch/sam2)
- [官方 segment_anything_2_video 示例](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_video)
