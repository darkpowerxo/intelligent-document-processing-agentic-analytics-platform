# 🚀 Cloud Deployment Guide - AWS & Azure Terraform

This directory contains comprehensive Terraform configurations for deploying the AI Architecture Demo on both AWS and Azure cloud platforms. The infrastructure includes all necessary services for a production-ready AI system with multi-agent architecture, streaming capabilities, and MLOps pipeline.

## 📋 **Quick Start**

### **Prerequisites**
- [Terraform](https://terraform.io) >= 1.0
- [Docker](https://docker.com) for building images
- **AWS**: AWS CLI configured with appropriate permissions
- **Azure**: Azure CLI logged in with appropriate permissions

### **Deploy to AWS**
```bash
cd terraform/aws
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
chmod +x deploy.sh
./deploy.sh
```

### **Deploy to Azure**
```bash
cd terraform/azure
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
chmod +x deploy.sh
./deploy.sh
```

## 🏗️ **Architecture Overview**

### **AWS Architecture**
- **Compute**: ECS Fargate with auto-scaling
- **Database**: RDS PostgreSQL with Multi-AZ
- **Cache**: ElastiCache Redis with failover
- **Streaming**: Amazon MSK (Kafka)
- **Storage**: S3 for MLflow artifacts
- **Load Balancer**: Application Load Balancer
- **Container Registry**: ECR
- **Monitoring**: CloudWatch + SNS alerts
- **Security**: VPC, Security Groups, IAM roles

### **Azure Architecture**
- **Compute**: Container Instances with auto-restart
- **Database**: PostgreSQL Flexible Server
- **Cache**: Azure Cache for Redis
- **Streaming**: Event Hubs (Kafka-compatible)
- **Storage**: Blob Storage for MLflow artifacts
- **Load Balancer**: Application Gateway
- **Container Registry**: Azure Container Registry
- **Monitoring**: Azure Monitor (optional)
- **Security**: VNet, NSGs, Key Vault, Managed Identity

## 📁 **Directory Structure**

```
terraform/
├── aws/                          # AWS Terraform configuration
│   ├── main.tf                   # VPC, networking, databases
│   ├── ecs.tf                    # ECS cluster and services
│   ├── iam.tf                    # IAM roles and policies
│   ├── variables.tf              # Input variables
│   ├── outputs.tf                # Output values
│   ├── terraform.tfvars.example  # Example configuration
│   └── deploy.sh                 # Deployment script
├── azure/                        # Azure Terraform configuration
│   ├── main.tf                   # Resource group, VNet, App Gateway
│   ├── storage.tf                # Database, Redis, Event Hubs, Storage
│   ├── containers.tf             # Container Instances
│   ├── variables.tf              # Input variables
│   ├── outputs.tf                # Output values
│   ├── terraform.tfvars.example  # Example configuration
│   └── deploy.sh                 # Deployment script
└── README.md                     # This file
```

## ⚙️ **Configuration**

### **Required Variables**

Both platforms require these core variables:

```hcl
# Project Configuration
project_name = "ai-architect-demo"
environment  = "dev"
owner        = "ai-team"

# Database Configuration
db_name     = "ai_demo"
db_username = "postgres"
db_password = "YourSecurePassword123!"  # CHANGE THIS!

# Network Configuration
vpc_cidr = "10.0.0.0/16"  # AWS
vnet_cidr = "10.0.0.0/16" # Azure
```

### **AWS-Specific Variables**
```hcl
aws_region          = "us-east-1"
rds_instance_class  = "db.t3.micro"
redis_node_type     = "cache.t3.micro"
kafka_instance_type = "kafka.t3.small"
api_task_cpu        = 512
api_task_memory     = 1024
```

### **Azure-Specific Variables**
```hcl
azure_location           = "East US"
postgres_sku_name        = "B_Standard_B1ms"
redis_sku_name          = "Basic"
eventhub_sku            = "Standard"
api_container_cpu       = 1
api_container_memory    = 2
```

## 🚀 **Deployment Process**

### **Step 1: Setup Configuration**
1. Copy the example tfvars file:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Edit `terraform.tfvars` with your specific values:
   - **Change default passwords**
   - Adjust instance sizes for your needs
   - Set your region/location
   - Configure alert email (optional)

### **Step 2: Deploy Infrastructure**
The deployment scripts handle the complete process:

```bash
# AWS
./deploy.sh

# Azure  
./deploy.sh

# Other commands
./deploy.sh plan     # Show deployment plan
./deploy.sh destroy  # Destroy all resources
./deploy.sh help     # Show help
```

### **Step 3: Build and Push Images**
The scripts automatically:
1. Create container registries
2. Build Docker images from your local codebase
3. Push images to the registries
4. Deploy container services

### **Step 4: Access Services**
After deployment, access your services:
- **API**: `http://<load-balancer-dns>`
- **Dashboard**: `http://<load-balancer-dns>/dashboard`
- **MLflow**: `http://<load-balancer-dns>/mlflow`

## 🔒 **Security Best Practices**

### **Secrets Management**
- **AWS**: Uses SSM Parameter Store for secrets
- **Azure**: Uses Key Vault for secrets
- Never commit passwords to version control
- Rotate secrets regularly in production

### **Network Security**
- Services deployed in private subnets
- Load balancers in public subnets
- Security groups/NSGs restrict access
- Database access only from application tier

### **Access Control**
- **AWS**: IAM roles with least privilege
- **Azure**: Managed Identity with RBAC
- Container registries require authentication
- Encrypted storage and transit

## 📊 **Resource Sizing**

### **Development Environment**
```hcl
# AWS
rds_instance_class = "db.t3.micro"
redis_node_type   = "cache.t3.micro"
api_task_cpu      = 512
api_task_memory   = 1024

# Azure
postgres_sku_name      = "B_Standard_B1ms"
redis_sku_name        = "Basic"
api_container_cpu     = 1
api_container_memory  = 2
```

### **Production Environment**
```hcl
# AWS
rds_instance_class = "db.r6g.large"
redis_node_type   = "cache.r6g.large"
api_task_cpu      = 2048
api_task_memory   = 4096

# Azure
postgres_sku_name      = "GP_Standard_D2s_v3"
redis_sku_name        = "Premium"
api_container_cpu     = 2
api_container_memory  = 4
```

## 💰 **Cost Management**

### **Development Costs (Estimated Monthly)**
- **AWS**: $50-100/month
- **Azure**: $60-120/month

### **Production Costs (Estimated Monthly)**
- **AWS**: $300-600/month
- **Azure**: $350-700/month

### **Cost Optimization Tips**
1. Use smaller instances for development
2. Enable auto-scaling to handle load
3. Set up billing alerts
4. Use reserved instances for production
5. Schedule non-production environments

## 🔧 **Troubleshooting**

### **Common Issues**

**1. Terraform Init Fails**
```bash
# AWS
export AWS_DEFAULT_REGION=us-east-1

# Azure
az login
az account set --subscription "your-subscription-id"
```

**2. Container Images Not Found**
```bash
# Rebuild and push images
BUILD_IMAGES=true ./deploy.sh
```

**3. Services Not Healthy**
```bash
# Check logs (AWS)
aws logs describe-log-groups --log-group-name-prefix "/ecs/ai-architect-demo"

# Check logs (Azure)  
az container logs --resource-group ai-architect-demo-rg --name ai-architect-demo-api-cg
```

**4. Load Balancer Not Responding**
- Check security group/NSG rules
- Verify backend health checks
- Ensure containers are running

### **Debugging Commands**

```bash
# Show Terraform outputs
terraform output

# Check resource status
terraform show

# Validate configuration
terraform validate

# Format code
terraform fmt -recursive
```

## 🔄 **CI/CD Integration**

### **GitHub Actions Example**

```yaml
name: Deploy to AWS
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
          
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        
      - name: Deploy Infrastructure
        run: |
          cd terraform/aws
          ./deploy.sh
```

## 📚 **Additional Resources**

- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Azure Provider](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)

## 🆘 **Support**

For deployment issues:
1. Check the troubleshooting section above
2. Review Terraform and container logs
3. Verify cloud provider permissions
4. Ensure all prerequisites are installed

## 🔄 **Updates and Maintenance**

### **Regular Tasks**
- Update Terraform providers monthly
- Patch container images regularly  
- Monitor resource usage and costs
- Review security configurations
- Backup important data

### **Scaling Considerations**
- Monitor application performance
- Adjust auto-scaling policies
- Upgrade database instances as needed
- Consider multi-region deployment
- Implement CDN for global users

---

**🎯 This infrastructure demonstrates enterprise-grade cloud architecture patterns suitable for AI/ML workloads with production-ready security, monitoring, and scalability features.**