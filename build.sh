
echo "--- Installing system dependencies (libgomp) ---"
yum install -y libgomp

echo "--- Installing Python dependencies ---"
pip install -r requirements.txt

echo "--- Build finished ---"