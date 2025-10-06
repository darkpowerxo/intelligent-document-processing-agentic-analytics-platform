#!/bin/bash

# =============================================================================
# AWS Deployment Script for AI Architecture Demo
# =============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGION=${AWS_DEFAULT_REGION:-"us-east-1"}
PROJECT_NAME="ai-architect-demo"
BUILD_IMAGES=${BUILD_IMAGES:-"true"}

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_prerequisites() {
    print_section "Checking Prerequisites"
    
    # Check if required tools are installed
    command -v aws >/dev/null 2>&1 || { print_error "AWS CLI is required but not installed."; exit 1; }
    command -v terraform >/dev/null 2>&1 || { print_error "Terraform is required but not installed."; exit 1; }
    command -v docker >/dev/null 2>&1 || { print_error "Docker is required but not installed."; exit 1; }
    
    # Check AWS credentials
    aws sts get-caller-identity >/dev/null 2>&1 || { print_error "AWS credentials not configured."; exit 1; }
    
    # Check if terraform.tfvars exists
    if [ ! -f "terraform.tfvars" ]; then
        print_error "terraform.tfvars not found. Copy from terraform.tfvars.example and customize."
        exit 1
    fi
    
    print_success "All prerequisites met"
}

build_and_push_images() {
    if [ "$BUILD_IMAGES" = "true" ]; then
        print_section "Building and Pushing Container Images"
        
        # Get ECR login token
        aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$REGION.amazonaws.com
        
        # Get repository URLs from Terraform output (after apply)
        API_REPO=$(terraform output -raw ecr_api_repository_url 2>/dev/null || echo "")
        DASHBOARD_REPO=$(terraform output -raw ecr_dashboard_repository_url 2>/dev/null || echo "")
        MLFLOW_REPO=$(terraform output -raw ecr_mlflow_repository_url 2>/dev/null || echo "")
        
        if [ -n "$API_REPO" ]; then
            print_warning "Building and pushing API image..."
            docker build -t $API_REPO:latest -f ../../docker/api/Dockerfile ../../
            docker push $API_REPO:latest
            
            print_warning "Building and pushing Dashboard image..."
            docker build -t $DASHBOARD_REPO:latest -f ../../docker/dashboard/Dockerfile ../../
            docker push $DASHBOARD_REPO:latest
            
            print_warning "Building and pushing MLflow image..."
            docker build -t $MLFLOW_REPO:latest -f ../../docker/mlflow/Dockerfile ../../
            docker push $MLFLOW_REPO:latest
            
            print_success "Images built and pushed successfully"
        else
            print_warning "ECR repositories not yet created. Images will be built after infrastructure deployment."
        fi
    fi
}

deploy_infrastructure() {
    print_section "Deploying Infrastructure with Terraform"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -out=tfplan
    
    # Apply deployment
    terraform apply tfplan
    
    print_success "Infrastructure deployed successfully"
}

post_deployment_tasks() {
    print_section "Post-Deployment Tasks"
    
    # Build and push images if not done earlier
    if [ "$BUILD_IMAGES" = "true" ] && [ -z "$API_REPO" ]; then
        build_and_push_images
        
        # Force ECS service update to use new images
        print_warning "Updating ECS services with new images..."
        CLUSTER_NAME=$(terraform output -raw ecs_cluster_id)
        aws ecs update-service --cluster $CLUSTER_NAME --service ${PROJECT_NAME}-api --force-new-deployment --region $REGION
        aws ecs update-service --cluster $CLUSTER_NAME --service ${PROJECT_NAME}-dashboard --force-new-deployment --region $REGION
    fi
    
    # Wait for services to be healthy
    print_warning "Waiting for services to become healthy..."
    sleep 60
    
    # Display deployment information
    print_section "Deployment Complete"
    
    ALB_DNS=$(terraform output -raw alb_dns_name)
    echo -e "\n${GREEN}🎉 Deployment successful!${NC}\n"
    echo -e "API URL:       http://$ALB_DNS"
    echo -e "Dashboard URL: http://$ALB_DNS/dashboard"
    echo -e "MLflow URL:    http://$ALB_DNS/mlflow"
    echo -e "\nNote: It may take a few minutes for the load balancer to route traffic properly."
    
    # Show resource summary
    terraform output deployment_summary 2>/dev/null || true
}

cleanup() {
    print_section "Cleanup"
    print_warning "This will destroy all AWS resources. Are you sure? (yes/no)"
    read -r response
    if [ "$response" = "yes" ]; then
        terraform destroy -auto-approve
        print_success "Resources destroyed successfully"
    else
        print_warning "Cleanup cancelled"
    fi
}

show_help() {
    echo "AWS Deployment Script for AI Architecture Demo"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy infrastructure and applications (default)"
    echo "  destroy    Destroy all resources"
    echo "  plan       Show deployment plan"
    echo "  help       Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  AWS_DEFAULT_REGION  AWS region (default: us-east-1)"
    echo "  BUILD_IMAGES        Build and push images (default: true)"
}

# Main execution
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            deploy_infrastructure
            build_and_push_images
            post_deployment_tasks
            ;;
        "destroy")
            cleanup
            ;;
        "plan")
            terraform plan
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"