
FROM nvidia/cuda:11.3.1-base

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated --no-install-recommends \
    build-essential apt-utils cmake git curl vim ca-certificates \
    libjpeg-dev libpng-dev \
    libgtk-3-dev libsm6 cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev \
    libboost-python-dev libtinyxml-dev bash \
    wget unzip libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev openssh-server \
    libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev patchelf libglfw3 tmux ffmpeg

RUN rm -rf /var/lib/apt/lists/*

ARG UID
RUN useradd -u $UID --create-home user
USER user
WORKDIR /home/user

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH /home/user/miniconda3/bin:$PATH

RUN mkdir -p .mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d .mujoco \
    && rm mujoco.zip
RUN wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d .mujoco \
    && rm mujoco.zip

# Make sure you have a license, otherwise comment this line out
# Of course you then cannot use Mujoco and DM Control, but Roboschool is still available
#COPY ./mjkey.txt .mujoco/mjkey.txt

#ENV LD_LIBRARY_PATH /home/user/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
#ENV LD_LIBRARY_PATH /home/user/.mujoco/mjpro200_linux/bin:${LD_LIBRARY_PATH}

RUN conda install -y python=3.7
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN conda install mpi4py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install glfw Cython imageio lockfile
#RUN pip install mujoco-py==1.50.1.68
RUN pip install tensorflow==1.14
RUN pip install git+git://github.com/deepmind/dm_control.git@103834
RUN pip install git+https://github.com/ShangtongZhang/dm_control2gym.git@scalar_fix
RUN pip install git+git://github.com/openai/baselines.git@8e56dd#egg=baselines
RUN pip3 install -U scikit-learn
RUN conda install -c conda-forge faiss-gpu cudatoolkit=10.2
RUN pip3 install gym-minigrid
RUN pip3 install pygad
WORKDIR /home/user/deep_rl