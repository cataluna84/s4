.PHONY: autoformat lint dev lib torchtext cu126 pt260 py312

.ONESHELL:

autoformat:
	black src/ checkpoints/ models/
	isort --atomic src/ checkpoints/ models/
	docformatter --in-place --recursive src checkpoints models

lint:
	isort -c src/ checkpoints/ models/
	black src/ checkpoints/ models/ --check
	flake8 src/ checkpoints/ models/

dev:
	pip install -r requirements-dev.txt

lib:
	pip install -r requirements.txt

torchtext: lib
	git clone https://github.com/pytorch/text build/torchtext
	cd build/torchtext
	git submodule update --init --recursive
	python setup.py clean install

cu126:
	#nvcc -V  # See original CUDA versions
	sudo apt-get update
	#sudo dpkg -l | grep -E "nvidia|cuda"  # See related packages
	sudo apt-get --allow-change-held-packages --purge remove "nvidia*" "libnvidia-*" "*cublas*" "cuda*" "nsight*"
	sudo rm -rf /usr/bin/nvidia*
	sudo rm -rf /usr/lib64-nvidia/*  # This path is Colab's weird choice and the folder is locked by other processes
	sudo rm -rf /opt/nvidia
	sudo rm -rf /opt/bin/nvidia-*
	sudo apt-get autoremove -y
	sudo apt-get autoclean
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i cuda-keyring_1.1-1_all.deb
	sudo apt-get update
	echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections
	sudo apt-get -y install cuda-12-6
	sudo apt-get autoremove -y
	sudo apt-get autoclean
	#nvcc -V  # See updated CUDA versions
	sudo update-alternatives --display cuda

pt260: cu126
	pip uninstall -y torch torchaudio torchvision
	pip install "torch>=2.6.0,<2.7" torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126

py312:
	sudo add-apt-repository -y ppa:deadsnakes/ppa
	DEBIAN_FRONTEND=noninteractive sudo apt-get install -y python3.12-full
	sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 3
	python -m ensurepip
