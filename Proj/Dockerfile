FROM ubuntu:20.04

# Set non-interactive mode to prevent tzdata from hanging
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.6 python3.6-dev python3-pip \
    && apt-get clean

# Clone the noise2noise repository
RUN git clone https://github.com/NVlabs/noise2noise.git

# Set working directory
WORKDIR /noise2noise

# # Upgrade pip and install dependencies globally for Python 3.6
# RUN python3.6 -m pip install --upgrade pip
# RUN python3.6 -m pip install --no-cache-dir -r requirements.txt

# Start in the bash shell, in the noise2noise directory.
CMD ["/bin/bash"]
