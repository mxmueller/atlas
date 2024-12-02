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

# Create installation directory
INSTALL_DIR="/opt/sam_project"
log "Creating installation directory: $INSTALL_DIR"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# System updates
log "Performing system updates..."
apt update || { error "apt update failed"; exit 1; }

# Install system dependencies including OpenGL libraries
log "Installing system dependencies..."
apt install -y \
    software-properties-common \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    || {
        error "System dependencies installation failed"
        exit 1
    }

# Create and activate virtual environment
log "Creating Python virtual environment..."
python3.10 -m venv $INSTALL_DIR/sam_env || {
    error "Virtual environment creation failed"
    exit 1
}

# Create activation script
cat > $INSTALL_DIR/activate_env.sh << 'EOL'
#!/bin/bash
source /opt/sam_project/sam_env/bin/activate
export PYTHONPATH=$PYTHONPATH:/opt/sam_project
EOL

chmod +x $INSTALL_DIR/activate_env.sh

# Activate virtual environment for installation
source $INSTALL_DIR/sam_env/bin/activate

# Upgrade pip
log "Upgrading pip..."
$INSTALL_DIR/sam_env/bin/pip install --upgrade pip || {
    error "Pip upgrade failed"
    exit 1
}

# Install packages one by one with error handling
install_package() {
    package=$1
    log "Installing $package..."
    $INSTALL_DIR/sam_env/bin/pip install $package || {
        error "$package installation failed"
        return 1
    }
    return 0
}

# Install numpy first
install_package "numpy>=1.24.0"

# Install PyTorch and dependencies
log "Installing PyTorch..."
$INSTALL_DIR/sam_env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 || {
    error "PyTorch installation failed"
    exit 1
}

# Install other packages
packages=(
    "opencv-python"
    "tqdm"
    "segment-anything"
)

for package in "${packages[@]}"; do
    install_package "$package" || exit 1
done

# Create Python test script
cat > $INSTALL_DIR/test_installation.py << 'EOL'
import torch
import cv2
from segment_anything import sam_model_registry
import numpy as np

def test_installation():
    print("Python Package Test:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"OpenCV Version: {cv2.__version__}")
    print("SAM import successful")
    print("Numpy Version:", np.__version__)

if __name__ == "__main__":
    test_installation()
EOL

# Run test
log "Testing installation..."
$INSTALL_DIR/sam_env/bin/python $INSTALL_DIR/test_installation.py || {
    error "Installation test failed"
    exit 1
}

# Set permissions
log "Setting permissions..."
chown -R root:root $INSTALL_DIR
chmod -R 755 $INSTALL_DIR

# Create symbolic links
log "Creating symbolic links..."
ln -sf $INSTALL_DIR/activate_env.sh /usr/local/bin/activate_sam

log "Installation completed successfully!"
log "Use 'source activate_sam' to activate the environment"

# Add information to .bashrc
echo "# SAM Project" >> /root/.bashrc
echo "alias activate_sam='source /opt/sam_project/activate_env.sh'" >> /root/.bashrc

exit 0