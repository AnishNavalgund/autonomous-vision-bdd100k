FROM jupyter/base-notebook:latest

# Install system dependencies for OpenCV and other image processing
USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better Docker layer caching
COPY docker_requirements.txt /tmp/docker_requirements.txt

# Install Python dependencies from requirements file
RUN pip install --no-cache-dir -r /tmp/docker_requirements.txt

# Copy notebooks and data analysis code
COPY notebooks/DataAnalysis/ /home/jovyan/work/notebooks/

# Set working directory
WORKDIR /home/jovyan/work

# Switch back to jovyan user
USER jovyan

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
