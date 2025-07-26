#!/bin/bash

# =============================================================================
# SSH Key Setup Script for A100 Pod Access
# =============================================================================
# This script automates SSH key generation and deployment to A100 pod
# Usage: ./setup_ssh_keys.sh [pod_ip] [username]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
POD_IP="${1}"
POD_USER="${2:-root}"
KEY_NAME="a100_pod_key"
KEY_PATH="$HOME/.ssh/$KEY_NAME"

echo -e "${BLUE}🔑 SSH Key Setup for A100 Pod${NC}"
echo "================================"

# =============================================================================
# Validation
# =============================================================================
if [ -z "$POD_IP" ]; then
    echo -e "${RED}❌ Error: Pod IP address required${NC}"
    echo "Usage: $0 <pod_ip> [username]"
    echo "Example: $0 192.168.1.100 root"
    exit 1
fi

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "   Pod IP: $POD_IP"
echo "   Username: $POD_USER"
echo "   Key name: $KEY_NAME"
echo "   Key path: $KEY_PATH"
echo ""

# =============================================================================
# Check if SSH key already exists
# =============================================================================
if [ -f "$KEY_PATH" ]; then
    echo -e "${YELLOW}⚠️  SSH key already exists at $KEY_PATH${NC}"
    read -p "Do you want to use the existing key? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🔄 Generating new key...${NC}"
        rm -f "$KEY_PATH" "$KEY_PATH.pub"
    else
        echo -e "${GREEN}✅ Using existing key${NC}"
        SKIP_GENERATION=true
    fi
fi

# =============================================================================
# Generate SSH Key Pair
# =============================================================================
if [ "$SKIP_GENERATION" != "true" ]; then
    echo -e "${YELLOW}🔐 Generating SSH key pair...${NC}"
    
    # Try ed25519 first, fallback to RSA if not supported
    if ssh-keygen -t ed25519 -f "$KEY_PATH" -C "$(whoami)@$(hostname)-a100-pod" -N ""; then
        echo -e "${GREEN}✅ Generated ed25519 key${NC}"
    elif ssh-keygen -t rsa -b 4096 -f "$KEY_PATH" -C "$(whoami)@$(hostname)-a100-pod" -N ""; then
        echo -e "${GREEN}✅ Generated RSA key${NC}"
    else
        echo -e "${RED}❌ Failed to generate SSH key${NC}"
        exit 1
    fi
fi

# =============================================================================
# Display Public Key
# =============================================================================
echo -e "${YELLOW}🔍 Your public key:${NC}"
echo "----------------------------------------"
cat "$KEY_PATH.pub"
echo "----------------------------------------"
echo ""

# =============================================================================
# Test Pod Connectivity
# =============================================================================
echo -e "${YELLOW}🔍 Testing pod connectivity...${NC}"

if timeout 10 nc -z "$POD_IP" 22 2>/dev/null; then
    echo -e "${GREEN}✅ Pod is reachable on port 22${NC}"
else
    echo -e "${RED}❌ Cannot reach pod on port 22${NC}"
    echo "Please check:"
    echo "1. Pod IP address is correct"
    echo "2. Pod is running and accessible"
    echo "3. SSH service is running on the pod"
    exit 1
fi

# =============================================================================
# Copy Key to Pod
# =============================================================================
echo -e "${YELLOW}📤 Copying public key to pod...${NC}"

# Method 1: Try ssh-copy-id first
if command -v ssh-copy-id >/dev/null 2>&1; then
    echo "Using ssh-copy-id..."
    if ssh-copy-id -i "$KEY_PATH.pub" "$POD_USER@$POD_IP"; then
        echo -e "${GREEN}✅ Key copied successfully using ssh-copy-id${NC}"
        KEY_COPIED=true
    else
        echo -e "${YELLOW}⚠️  ssh-copy-id failed, trying manual method...${NC}"
    fi
fi

# Method 2: Manual copy if ssh-copy-id failed
if [ "$KEY_COPIED" != "true" ]; then
    echo "Using manual copy method..."
    echo "You will be prompted for the pod password..."
    
    # Create the command to run on the remote pod
    REMOTE_CMD="mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
    
    if cat "$KEY_PATH.pub" | ssh "$POD_USER@$POD_IP" "$REMOTE_CMD"; then
        echo -e "${GREEN}✅ Key copied successfully using manual method${NC}"
        KEY_COPIED=true
    else
        echo -e "${RED}❌ Failed to copy key to pod${NC}"
        exit 1
    fi
fi

# =============================================================================
# Test SSH Connection
# =============================================================================
echo -e "${YELLOW}🔧 Testing SSH connection...${NC}"

if ssh -i "$KEY_PATH" -o ConnectTimeout=10 -o BatchMode=yes "$POD_USER@$POD_IP" "echo 'SSH key authentication successful'" 2>/dev/null; then
    echo -e "${GREEN}✅ SSH key authentication working!${NC}"
else
    echo -e "${RED}❌ SSH key authentication failed${NC}"
    echo "Troubleshooting steps:"
    echo "1. Check pod permissions: ssh $POD_USER@$POD_IP 'ls -la ~/.ssh/'"
    echo "2. Try manual connection: ssh -i $KEY_PATH $POD_USER@$POD_IP"
    echo "3. Use verbose mode: ssh -i $KEY_PATH -v $POD_USER@$POD_IP"
    exit 1
fi

# =============================================================================
# Create SSH Config Entry
# =============================================================================
echo -e "${YELLOW}⚙️  Creating SSH config entry...${NC}"

SSH_CONFIG="$HOME/.ssh/config"
CONFIG_ENTRY="
# A100 Pod Configuration
Host a100-pod
    HostName $POD_IP
    User $POD_USER
    IdentityFile $KEY_PATH
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
"

# Check if config already has this entry
if [ -f "$SSH_CONFIG" ] && grep -q "Host a100-pod" "$SSH_CONFIG"; then
    echo -e "${YELLOW}⚠️  SSH config entry already exists${NC}"
    read -p "Update existing entry? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove existing entry and add new one
        sed -i '/# A100 Pod Configuration/,/ServerAliveCountMax 3/d' "$SSH_CONFIG"
        echo "$CONFIG_ENTRY" >> "$SSH_CONFIG"
        echo -e "${GREEN}✅ SSH config updated${NC}"
    fi
else
    echo "$CONFIG_ENTRY" >> "$SSH_CONFIG"
    echo -e "${GREEN}✅ SSH config entry added${NC}"
fi

# =============================================================================
# Test SSH Config
# =============================================================================
echo -e "${YELLOW}🧪 Testing SSH config...${NC}"

if ssh -o ConnectTimeout=10 a100-pod "echo 'SSH config working'" 2>/dev/null; then
    echo -e "${GREEN}✅ SSH config working! You can now use: ssh a100-pod${NC}"
else
    echo -e "${YELLOW}⚠️  SSH config test failed, but direct connection should work${NC}"
fi

# =============================================================================
# Add Key to SSH Agent
# =============================================================================
echo -e "${YELLOW}🔐 Adding key to SSH agent...${NC}"

# Start SSH agent if not running
if [ -z "$SSH_AUTH_SOCK" ]; then
    eval "$(ssh-agent -s)"
fi

# Add key to agent
if ssh-add "$KEY_PATH" 2>/dev/null; then
    echo -e "${GREEN}✅ Key added to SSH agent${NC}"
else
    echo -e "${YELLOW}⚠️  Could not add key to SSH agent (this is usually fine)${NC}"
fi

# =============================================================================
# Create Helper Scripts
# =============================================================================
echo -e "${YELLOW}📝 Creating helper scripts...${NC}"

# Create connection script
cat > "connect_a100.sh" << EOF
#!/bin/bash
# Quick connection script for A100 pod
echo "🚀 Connecting to A100 pod..."
ssh -i "$KEY_PATH" "$POD_USER@$POD_IP" "\$@"
EOF
chmod +x "connect_a100.sh"

# Create file transfer script
cat > "transfer_to_a100.sh" << EOF
#!/bin/bash
# File transfer script for A100 pod
if [ \$# -eq 0 ]; then
    echo "Usage: \$0 <local_file_or_dir> [remote_path]"
    echo "Example: \$0 ./my_files/ /home/training/"
    exit 1
fi

LOCAL_PATH="\$1"
REMOTE_PATH="\${2:-~/}"

echo "📤 Transferring \$LOCAL_PATH to $POD_USER@$POD_IP:\$REMOTE_PATH"
rsync -avz -e "ssh -i $KEY_PATH" "\$LOCAL_PATH" "$POD_USER@$POD_IP:\$REMOTE_PATH"
EOF
chmod +x "transfer_to_a100.sh"

echo -e "${GREEN}✅ Helper scripts created:${NC}"
echo "   - connect_a100.sh: Quick connection"
echo "   - transfer_to_a100.sh: File transfer"

# =============================================================================
# Final Instructions
# =============================================================================
echo ""
echo -e "${GREEN}🎉 SSH Setup Complete!${NC}"
echo "======================="
echo ""
echo -e "${BLUE}🔗 Connection Methods:${NC}"
echo "   Direct: ssh -i $KEY_PATH $POD_USER@$POD_IP"
echo "   Config: ssh a100-pod"
echo "   Script: ./connect_a100.sh"
echo ""
echo -e "${BLUE}📤 File Transfer:${NC}"
echo "   Manual: scp -i $KEY_PATH file.txt $POD_USER@$POD_IP:~/"
echo "   Rsync:  rsync -avz -e 'ssh -i $KEY_PATH' local/ $POD_USER@$POD_IP:remote/"
echo "   Script: ./transfer_to_a100.sh local_file remote_path"
echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo "   1. Test connection: ssh a100-pod"
echo "   2. Transfer training files: ./transfer_to_a100.sh uav_landing_project/"
echo "   3. Setup training environment: ssh a100-pod 'cd uav_landing_project && ./setup_a100_pod.sh'"
echo ""
echo -e "${BLUE}🛠️ Troubleshooting:${NC}"
echo "   Verbose SSH: ssh -v a100-pod"
echo "   Check permissions: ssh a100-pod 'ls -la ~/.ssh/'"
echo "   Manual connection: ssh -i $KEY_PATH $POD_USER@$POD_IP"
echo ""
echo -e "${GREEN}✅ Ready for A100 GPU training!${NC}" 