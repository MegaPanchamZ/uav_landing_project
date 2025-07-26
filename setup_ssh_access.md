# SSH Access Setup for A100 Pod

Complete guide to set up secure SSH access to your A100 GPU pod using public key authentication.

## 🔑 Quick SSH Setup

### Step 1: Generate SSH Key Pair (if you don't have one)

**On your local machine:**

```bash
# Generate a new SSH key pair
ssh-keygen -t ed25519 -C "your_email@example.com"

# Or if your system doesn't support ed25519, use RSA
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

When prompted:
- **File location**: Press Enter for default (`~/.ssh/id_ed25519` or `~/.ssh/id_rsa`)
- **Passphrase**: Enter a secure passphrase (recommended) or press Enter for no passphrase

### Step 2: Copy Public Key to A100 Pod

**Method 1: Using ssh-copy-id (easiest)**
```bash
# Replace with your pod's IP and username
ssh-copy-id root@your_pod_ip

# Or if using a specific key file
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@your_pod_ip
```

**Method 2: Manual copy (if ssh-copy-id doesn't work)**
```bash
# Display your public key
cat ~/.ssh/id_ed25519.pub
# Copy the entire output

# Then SSH into your pod with password and paste it
ssh root@your_pod_ip
mkdir -p ~/.ssh
echo "paste_your_public_key_here" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
exit
```

**Method 3: Using SCP**
```bash
# Copy public key file to pod
scp ~/.ssh/id_ed25519.pub root@your_pod_ip:/tmp/

# SSH into pod and install key
ssh root@your_pod_ip
mkdir -p ~/.ssh
cat /tmp/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
rm /tmp/id_ed25519.pub
exit
```

### Step 3: Test SSH Connection

```bash
# Test connection (should not ask for password)
ssh root@your_pod_ip

# If it works, you should be logged in without password prompt
```

## 🛠️ Platform-Specific Instructions

### Windows Users

**Using Windows Subsystem for Linux (WSL)**
```bash
# Open WSL terminal
wsl

# Follow the Linux instructions above
ssh-keygen -t ed25519 -C "your_email@example.com"
ssh-copy-id root@your_pod_ip
```

**Using PowerShell**
```powershell
# Generate key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Display public key
Get-Content ~\.ssh\id_ed25519.pub

# Copy the output and manually add to pod (Method 2 above)
```

**Using PuTTY**
1. Download PuTTYgen
2. Generate key pair
3. Save private key (.ppk format)
4. Copy public key text to pod's ~/.ssh/authorized_keys

### macOS/Linux Users

```bash
# Standard process
ssh-keygen -t ed25519 -C "your_email@example.com"
ssh-copy-id root@your_pod_ip
```

## 🔧 Troubleshooting SSH Issues

### Common Problems and Solutions

**1. Permission Denied (publickey)**
```bash
# Check if your key is loaded in SSH agent
ssh-add -l

# If not loaded, add it
ssh-add ~/.ssh/id_ed25519

# Try connecting with verbose output to debug
ssh -v root@your_pod_ip
```

**2. Wrong Permissions on Pod**
```bash
# SSH into pod and fix permissions
ssh root@your_pod_ip
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
chown -R root:root ~/.ssh
```

**3. SSH Agent Issues**
```bash
# Start SSH agent
eval "$(ssh-agent -s)"

# Add your key
ssh-add ~/.ssh/id_ed25519

# Test connection
ssh root@your_pod_ip
```

**4. Firewall/Network Issues**
```bash
# Test if SSH port (22) is open
telnet your_pod_ip 22

# Or use nmap
nmap -p 22 your_pod_ip

# Try connecting with different port if needed
ssh -p 2222 root@your_pod_ip
```

**5. Key Format Issues**
```bash
# Convert between key formats if needed
ssh-keygen -p -f ~/.ssh/id_rsa -m pem

# Or generate a new RSA key if ed25519 isn't supported
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

## 🔐 Advanced SSH Configuration

### Create SSH Config File

Create `~/.ssh/config` for easier connection:

```bash
# Edit SSH config
nano ~/.ssh/config

# Add this content:
Host a100-pod
    HostName your_pod_ip
    User root
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Now you can connect simply with:
```bash
ssh a100-pod
```

### Multiple Keys Management

If you have multiple SSH keys:
```bash
# Specify which key to use
ssh -i ~/.ssh/specific_key root@your_pod_ip

# Or add to SSH config
Host a100-pod
    HostName your_pod_ip
    User root
    IdentityFile ~/.ssh/specific_key
```

## 🚀 Quick Setup Script

Here's a one-liner to set up SSH access:

```bash
# Generate key and copy to pod (replace YOUR_POD_IP)
ssh-keygen -t ed25519 -C "$(whoami)@$(hostname)" -f ~/.ssh/a100_key -N "" && ssh-copy-id -i ~/.ssh/a100_key.pub root@YOUR_POD_IP
```

## 📋 Verification Checklist

After setup, verify:

- [ ] SSH connection works without password: `ssh root@your_pod_ip`
- [ ] You can run commands: `ssh root@your_pod_ip "nvidia-smi"`
- [ ] File transfer works: `scp test_file root@your_pod_ip:/tmp/`
- [ ] Rsync works: `rsync -av local_dir/ root@your_pod_ip:/remote_dir/`

## 🔄 Using SSH with Training Scripts

Once SSH is set up, you can use the training scripts:

```bash
# Copy training files to pod
scp -r uav_landing_project/ root@your_pod_ip:~/

# SSH and start setup
ssh root@your_pod_ip
cd uav_landing_project
chmod +x *.sh
./setup_a100_pod.sh
```

## 🛡️ Security Best Practices

1. **Use strong passphrases** for SSH keys
2. **Disable password authentication** on the pod after key setup:
   ```bash
   # On the pod, edit SSH config
   sudo nano /etc/ssh/sshd_config
   
   # Set these values:
   PasswordAuthentication no
   PubkeyAuthentication yes
   PermitRootLogin prohibit-password
   
   # Restart SSH service
   sudo systemctl restart sshd
   ```

3. **Use SSH agent forwarding** for git operations:
   ```bash
   ssh -A root@your_pod_ip
   ```

4. **Consider using a jump host** if accessing through a gateway

## 🔗 Integration with Training Workflow

With SSH set up, your complete workflow becomes:

```bash
# 1. Transfer files to pod
rsync -av uav_landing_project/ root@your_pod_ip:~/uav_landing_system/

# 2. Setup environment
ssh root@your_pod_ip "cd uav_landing_system && ./setup_a100_pod.sh"

# 3. Upload Kaggle credentials
scp kaggle.json root@your_pod_ip:~/.kaggle/

# 4. Start training
ssh root@your_pod_ip "cd uav_landing_system && tmux new -d -s training './download_datasets.sh && python train_a100.py'"

# 5. Monitor training
ssh root@your_pod_ip "tmux attach -t training"

# 6. Sync results back
./sync_results.sh your_pod_ip root ./results
```

---

**Need help?** If you're still having issues, provide:
- Your operating system
- Error messages you're seeing  
- Output of `ssh -v root@your_pod_ip` 