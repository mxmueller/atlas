# Docker, NVIDIA Drivers and CUDA Installation Guide for Ubuntu

## 1. NVIDIA Drivers and CUDA Installation

### 1.1 Install Ubuntu Drivers Utility
```bash
sudo apt update && sudo apt install -y ubuntu-drivers-common
```

### 1.2 Install NVIDIA Drivers
```bash
sudo ubuntu-drivers install
sudo reboot
```

### 1.3 Install CUDA Toolkit
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install -y ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-12-5
sudo reboot
```

### 1.4 Verify Installation
```bash
nvidia-smi
```

## 2. Docker Installation

### 2.1 Update System
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 2.2 Install Required Packages
```bash
sudo apt-get install -y ca-certificates curl gnupg
```

### 2.3 Add Docker GPG Key and Repository
```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 2.4 Install Docker
```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 2.5 Install NVIDIA Docker Support
```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

### 2.6 Restart Docker Service
```bash
sudo systemctl restart docker
```

### 2.7 Add User to Docker Group
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 2.8 Test Installation
```bash
docker --version
docker compose version
nvidia-docker version
docker run hello-world
```

## 3. Important Notes

- Ubuntu packages NVIDIA drivers directly from NVIDIA
- Installing drivers from other sources may break your system
- Driver installation on VMs with TrustedLaunch and Secure Boot requires additional steps
- Ubuntu drivers are signed by Canonical and work with Secure Boot
- For issues, consult the official NVIDIA documentation or Ubuntu community forums