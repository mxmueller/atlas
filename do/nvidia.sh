#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
    error "This script must be run with root privileges (sudo)."
    exit 1
fi

# Detect NVIDIA GPU
log "Detecting NVIDIA GPU..."
if ! lspci | grep -i nvidia > /dev/null; then
    error "No NVIDIA GPU detected. Please check your hardware."
    exit 1
fi

# Get GPU model
GPU_MODEL=$(lspci | grep -i nvidia | head -n 1)
log "Found NVIDIA GPU: $GPU_MODEL"

# Install tools for driver detection
log "Installing tools for driver detection..."
apt update
apt install -y ubuntu-drivers-common pciutils

# Get recommended driver
log "Checking recommended driver version..."
RECOMMENDED_DRIVER=$(ubuntu-drivers devices | grep "recommended" | awk '{print $3}')

if [ -z "$RECOMMENDED_DRIVER" ]; then
    warning "No recommended driver found, falling back to nvidia-driver-535"
    DRIVER_PACKAGE="nvidia-driver-535"
else
    log "Recommended driver: $RECOMMENDED_DRIVER"
    DRIVER_PACKAGE=$RECOMMENDED_DRIVER
fi

# Update package list
log "Updating package list..."
apt update || {
    error "Failed to update package list"
    exit 1
}

# Remove any existing NVIDIA drivers
log "Removing existing NVIDIA installations..."
apt remove --purge -y nvidia*
apt autoremove -y

# Add NVIDIA repository
log "Adding NVIDIA repository..."
apt install -y software-properties-common
add-apt-repository -y ppa:graphics-drivers/ppa
apt update

# Install recommended NVIDIA driver
log "Installing NVIDIA driver: $DRIVER_PACKAGE..."
apt install -y $DRIVER_PACKAGE || {
    error "Failed to install NVIDIA driver"
    exit 1
}

# Install CUDA toolkit
log "Installing CUDA toolkit..."
apt install -y nvidia-cuda-toolkit || {
    error "Failed to install CUDA toolkit"
    exit 1
}

# Install additional CUDA dependencies
log "Installing additional CUDA dependencies..."
apt install -y \
    nvidia-cuda-dev \
    nvidia-cuda-toolkit-gcc \
    || {
    error "Failed to install CUDA dependencies"
    exit 1
}

# Setup environment variables
log "Setting up environment variables..."
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /etc/profile.d/cuda.sh
chmod +x /etc/profile.d/cuda.sh

# Create diagnostic script
cat > /usr/local/bin/test-nvidia << 'EOL'
#!/bin/bash
echo "NVIDIA System Diagnostic Report"
echo "=============================="
echo -e "\n1. GPU Hardware Information:"
lspci | grep -i nvidia

echo -e "\n2. NVIDIA Driver Information:"
nvidia-smi

echo -e "\n3. CUDA Version Information:"
nvcc --version

echo -e "\n4. CUDA Device Check (Python):"
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('CUDA device count:', torch.cuda.device_count())
    print('Current CUDA device:', torch.cuda.current_device())
    print('CUDA device name:', torch.cuda.get_device_name(0))
"
EOL

chmod +x /usr/local/bin/test-nvidia

# Final instructions
log "Installation completed!"
log "Installed driver: $DRIVER_PACKAGE"
log "Please reboot your system to complete the installation:"
log "sudo reboot"
log "After reboot, run 'test-nvidia' to verify the installation"

exit 0