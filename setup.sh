#!/bin/bash
# WAF Setup Script
# Automatically detects OS and installs dependencies

set -e

echo "🛡️  Transformer WAF - Automated Setup"
echo "======================================"
echo ""

# Check Python version
echo "✓ Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "  Python version: $PYTHON_VERSION"

# Check Docker
echo "✓ Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed."
    echo "   Install from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
echo "  Docker version: $DOCKER_VERSION"

# Check Docker Compose
echo "✓ Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed."
    echo "   Install from: https://docs.docker.com/compose/install/"
    exit 1
fi

COMPOSE_VERSION=$(docker-compose --version | awk '{print $3}')
echo "  Docker Compose version: $COMPOSE_VERSION"

# Check Git
echo "✓ Checking Git..."
if ! command -v git &> /dev/null; then
    echo "⚠️  Git is not installed (optional, but recommended)"
fi

# Install Python dependencies
echo ""
echo "✓ Installing Python dependencies..."
python3 -m pip install --quiet requests 2>/dev/null || true

# Create nginx logs directory
echo "✓ Creating directories..."
mkdir -p nginx/logs

# Verify setup
echo ""
echo "======================================"
echo "✅ SETUP COMPLETE"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Start the system:"
echo "   docker-compose up -d --build"
echo ""
echo "2. Wait 40 seconds for initialization:"
echo "   sleep 40"
echo ""
echo "3. Run the dashboard:"
echo "   python3 waf_dashboard.py"
echo ""
echo "4. Visit the website:"
echo "   open http://localhost:8080/"
echo ""
