# last updated Mar 25 2025, 11:00am
FROM python:3.12-slim

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless), procps (for 'ps') and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark's scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh

# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Airflow home directory (shared via named volume in docker-compose)
ENV AIRFLOW_HOME=/opt/airflow

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Apache Airflow with official constraints to avoid dependency conflicts
ARG AIRFLOW_VERSION=2.9.3
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"

# Expose JupyterLab port and Airflow webserver port
EXPOSE 8888 8080

# Create a volume mount point for notebooks
VOLUME /app

# Enable JupyterLab via environment variable
ENV JUPYTER_ENABLE_LAB=yes

# Default command starts JupyterLab; Airflow services override this via docker-compose
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
