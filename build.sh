#!/bin/bash

# This script runs in the Vercel build environment

echo "--- Installing system dependencies (libgomp) ---"
yum install -y libgomp

echo "--- Installing Python dependencies ---"
# Use the python3.12 executable to run pip
python3.12 -m pip install -r requirements.txt

echo "--- Build finished ---"