#!/bin/bash

# =============================================
# Auto Lock Screen - Setup Script
# =============================================

echo "╔═══════════════════════════════════════╗"
echo "║   🔐 Auto Lock Screen - Setup         ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 chưa được cài đặt!"
    echo "Vui lòng cài đặt Python từ https://python.org hoặc dùng Homebrew:"
    echo "  brew install python3"
    exit 1
fi

echo "✅ Python 3 đã cài đặt: $(python3 --version)"

# Tạo virtual environment
echo ""
echo "📦 Tạo virtual environment..."
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Cài đặt dependencies
echo ""
echo "📥 Cài đặt dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""

echo "✅ Cài đặt hoàn tất!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Để chạy ứng dụng:"
echo ""
echo "  source venv/bin/activate"
echo "  python auto_lock.py"
echo ""
echo "Hoặc với debug mode:"
echo ""
echo "  python auto_lock.py -d"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
