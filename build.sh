#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Installing system dependencies (libgomp) ---"
yum install -y libgomp

echo "--- Installing Python dependencies ---"
python3.12 -m pip install -r requirements.txt

echo "--- Locating and copying libgomp ---"
# Find the installed libgomp.so.1 file
LIB_PATH=$(find / -name "libgomp.so.1" -type f 2>/dev/null | head -n 1)

if [ -z "$LIB_PATH" ]; then
  echo "Error: libgomp.so.1 not found after installation."
  exit 1
fi

echo "Found libgomp.so.1 at: $LIB_PATH"

# Copy the library to the _vendor directory where pip packages are installed
# This ensures it's included in the final deployment package.
VENDOR_DIR="./_vendor"
if [ -d "$VENDOR_DIR" ]; then
  echo "Copying $LIB_PATH to $VENDOR_DIR"
  cp "$LIB_PATH" "$VENDOR_DIR/"
else
  echo "Warning: _vendor directory not found. Cannot copy libgomp."
fi

echo "--- Build finished ---"