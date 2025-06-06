# 使用现有镜像作为基础
FROM pytorch/pytorch:latest

# 卸载现有的torch相关包
RUN pip uninstall -y torch torchvision torchaudio

# 安装PyTorch 2.3.0和CUDA 11.8
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN apt-get update && apt-get install -y build-essential

# 安装PyG
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

# 安装系统依赖
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget \
    netcat-traditional \
    gnupg \
    curl \
    unzip \
    zip \
    xvfb \
    libgconf-2-4 \
    libxss1 \
    libnss3 \
    libnspr4 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    xdg-utils \
    fonts-liberation \
    dbus \
    xauth \
    xvfb \
    x11vnc \
    tigervnc-tools \
    supervisor \
    net-tools \
    procps \
    git \
    python3-numpy \
    fontconfig \
    fonts-dejavu \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    tmux \
    poppler-utils \
    wkhtmltopdf \
    antiword \
    unrtf \
    catdoc \
    grep \
    gawk \
    sed \
    file \
    jq \
    csvkit \
    xmlstarlet \
    less \
    vim \
    tree \
    rsync \
    lsof \
    iputils-ping \
    dnsutils \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Set platform for ARM64 compatibility
ARG TARGETPLATFORM=linux/amd64

RUN mkdir -p /workplace
# 设置工作目录
WORKDIR /workplace

# 复制requirements.txt并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app
# Copy server script
COPY . /app
COPY tcp_server.py /app/tcp_server.py


# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=false

# 设置supervisor配置
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN pip3 install --no-cache-dir requests fastapi uvicorn python-multipart httpx torch_geometric

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]