# Configured for machine:
# Linux llm-opt 6.5.0-1023-azure #24~22.04.1-Ubuntu SMP Wed Jun 12 19:55:26 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux

distro="ubuntu2204"
arch="x86_64"

clean() {
  # remove cuda toolkit
  sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"  

  # remove nvidia drivers
  sudo apt-get --purge remove "*nvidia*" "libxnvctrl*"
  sudo apt-get autoremove -y


  rm -rf cuda-keyring_*_all.deb*
  rm -rf cuda-repo*
  rm -rf cuda_*.run*
}

main() {
  clean

  sudo apt-get install --verbose-versions nvidia-kernel-source-550-open cuda-drivers-550 -y
  wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_555.42.06_linux.run
  # sudo sh cuda_12.5.1_555.42.06_linux.run -m=kernel-open ## NOTE: this will cause erronous terminal output, hence run it manually after the script completes

}



main 2>&1 | tee "install-cuda-toolkit.log" || exit 1
sudo sh cuda_12.5.1_555.42.06_linux.run -m=kernel-open


sudo apt autoremove -y