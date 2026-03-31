#!/usr/bin/env bash
# ============================================================
# SAM3 Server Setup Script
# ─────────────────────────────────────────────────────────────
# One-time server setup script. Run this on a fresh Ubuntu
# server before your first deployment.
#
# Usage (as root):
#   curl -fsSL https://your-ci-server/scripts/deploy/setup-server.sh | bash
# ============================================================

set -euo pipefail

echo "══════════════════════════════════════════════════════════"
echo "  SAM3 Server Setup"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Detect OS ─────────────────────────────────────────────────
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS="$ID"
else
    OS="unknown"
fi

echo "[INFO] Detected OS: $OS"

# ── Update system ──────────────────────────────────────────────
echo ""
echo "[INFO] Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

# ── Install Docker ─────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[INFO] Installing Docker..."
    apt-get install -y -qq ca-certificates curl gnupg lsb-release

    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/${OS:-ubuntu}/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg 2>/dev/null || \
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/${OS:-ubuntu} $(lsb_release -cs) stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin

    # Enable Docker
    systemctl enable docker
    systemctl start docker

    # Add docker group
    groupadd -f docker
    usermod -aG docker "$SUDO_USER"
    echo "[OK] Docker installed"
else
    echo "[OK] Docker already installed: $(docker --version)"
fi

# ── Install Docker Compose standalone (if not using plugin) ─────
if ! docker compose version &>/dev/null; then
    echo "[INFO] Installing Docker Compose..."
    DOCKER_COMPOSE_VERSION=$(curl -fsSL https://api.github.com/repos/docker/compose/releases/latest | \
        grep '"tag_name"' | cut -d'"' -f4 2>/dev/null || echo "v2.24.0")
    curl -fsSL "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64" \
        -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "[OK] Docker Compose installed"
fi

# ── Install utilities ─────────────────────────────────────────
echo "[INFO] Installing utilities..."
apt-get install -y -qq curl git jq ufw fail2ban htop vim net-tools

# ── Install Tailscale ─────────────────────────────────────────
echo "[INFO] Setting up Tailscale..."
if ! command -v tailscale &>/dev/null; then
    echo "[INFO] Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
    echo "[OK] Tailscale installed"
else
    echo "[OK] Tailscale already installed"
fi

echo ""
echo "[INFO] To connect this server to your Tailscale network:"
echo "       Run the following command on this server:"
echo ""
echo "       sudo tailscale up --accept-routes --accept-dns \\"
echo "         --operator=\$USER \\"
echo "         --authkey=<YOUR_TAILSCALE_AUTHKEY>"
echo ""
echo "       After connecting, note the Tailscale IP (100.x.x.x) —"
echo "       use it as DEPLOY_HOST in GitHub Actions."

# ── Configure firewall ─────────────────────────────────────────
echo "[INFO] Configuring firewall (UFW)..."
ufw --force enable
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8001/tcp    # SAM3 API
ufw allow 9000/tcp   # MinIO API
ufw allow 9001/tcp   # MinIO Console
ufw allow 5432/tcp   # PostgreSQL (restrict in production!)
ufw allow 19530/tcp  # Milvus
echo "[OK] Firewall configured"

# ── Create deploy user ─────────────────────────────────────────
if ! id "deploy" &>/dev/null; then
    echo "[INFO] Creating deploy user..."
    useradd -m -s /bin/bash -G docker deploy
    echo "deploy ALL=(ALL) NOPASSWD: /usr/bin/docker" >> /etc/sudoers.d/deploy
    echo "[OK] Deploy user created"
fi

# ── Create deployment directory ──────────────────────────────────
echo "[INFO] Creating deployment directory..."
mkdir -p /opt/sam3
chown deploy:deploy /opt/sam3
echo "[OK] Deployment directory: /opt/sam3"

# ── Harden SSH ─────────────────────────────────────────────────
echo "[INFO] Hardening SSH (disable password auth)..."
sed -i 's/^#*PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#*PubkeyAuthentication no/PubkeyAuthentication yes/' /etc/ssh/sshd_config
systemctl reload sshd 2>/dev/null || true
echo "[OK] SSH hardened"

# ── Enable automatic security updates ──────────────────────────
echo "[INFO] Setting up automatic security updates..."
apt-get install -y -qq unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades || true
echo "[OK] Automatic security updates configured"

# ── Install Fail2Ban ───────────────────────────────────────────
echo "[INFO] Configuring Fail2Ban..."
cat > /etc/fail2ban/jail.local << 'EOF'
[sshd]
enabled = true
port = ssh
maxretry = 5
bantime = 3600
findtime = 600
EOF
systemctl enable fail2ban
systemctl start fail2ban
echo "[OK] Fail2Ban configured"

# ── Setup log rotation ─────────────────────────────────────────
echo "[INFO] Setting up log rotation..."
cat > /etc/logrotate.d/sam3 << 'EOF'
/opt/sam3/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 deploy deploy
    sharedscripts
    postrotate
        docker compose -f /opt/sam3/docker-compose.prod.yml reload sam3-api > /dev/null 2>&1 || true
    endscript
}
EOF
echo "[OK] Log rotation configured"

# ── Final message ─────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ✅ Server Setup Complete!"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Add SSH public key for 'deploy' user:"
echo "     mkdir -p /home/deploy/.ssh && chmod 700 /home/deploy/.ssh"
echo "     echo 'YOUR_PUBLIC_KEY' >> /home/deploy/.ssh/authorized_keys"
echo ""
echo "  2. Copy your files to /opt/sam3:"
echo "     scp docker-compose.prod.yml .env.prod deploy@SERVER:/opt/sam3/"
echo ""
echo "  3. Set up GitHub Actions secrets in your repo:"
echo "     - DEPLOY_SSH_KEY: Private key for deploy user"
echo "     - DEPLOY_HOST: Your server hostname/IP"
echo ""
echo "  4. Push to main branch to trigger deployment!"
echo ""
