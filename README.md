# NVIDIA Drivers and CUDA Installation Guide for Ubuntu

This guide provides step-by-step instructions for installing NVIDIA proprietary drivers and CUDA toolkit on Ubuntu systems.

## Important Notes

Ubuntu packages NVIDIA proprietary drivers directly from NVIDIA. These drivers are packaged by Ubuntu for automatic system management. Please note:

- Downloading and installing drivers from other sources may break your system
- Installing third-party drivers on VMs with TrustedLaunch and Secure Boot requires additional steps
- A new Machine Owner Key must be added for the system to boot
- Drivers from Ubuntu are signed by Canonical and will work with Secure Boot

## Installation Steps

### 1. Install Ubuntu Drivers Utility

```bash
sudo apt update && sudo apt install -y ubuntu-drivers-common
```

### 2. Install NVIDIA Drivers

```bash
sudo ubuntu-drivers install
```

**Important**: Reboot the VM after the GPU driver installation is complete.

### 3. Install CUDA Toolkit

> **Note**: The following example shows the CUDA package path for Ubuntu 24.04 LTS. Replace the path specific to your Ubuntu version.
>
> Visit the [NVIDIA Download Center](https://developer.nvidia.com/cuda-downloads) or [NVIDIA CUDA Resources](https://developer.nvidia.com/cuda-toolkit) page for the full path specific to your version.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install -y ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-12-5
```

The installation process can take several minutes to complete.

### 4. System Reboot

After installation completes, reboot your system:

```bash
sudo reboot
```

### 5. Verify Installation

After the system reboots, verify that the GPU is correctly recognized:

```bash
nvidia-smi
```

This command should display information about your NVIDIA GPU if the installation was successful.

## Support

If you encounter any issues during installation, please consult the official NVIDIA documentation or Ubuntu community forums for additional support.