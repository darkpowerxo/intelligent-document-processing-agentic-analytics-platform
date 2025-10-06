#!/bin/bash

# =============================================================================
# Azure Deployment Script for AI Architecture Demo
# =============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LOCATION=${AZURE_LOCATION:-"East US"}
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
    command -v az >/dev/null 2>&1 || { print_error "Azure CLI is required but not installed."; exit 1; }
    command -v terraform >/dev/null 2>&1 || { print_error "Terraform is required but not installed."; exit 1; }
    command -v docker >/dev/null 2>&1 || { print_error "Docker is required but not installed."; exit 1; }
    
    # Check Azure login
    az account show >/dev/null 2>&1 || { print_error "Not logged in to Azure. Run 'az login'"; exit 1; }
    
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
        
        # Get ACR login server from Terraform output (after apply)
        ACR_SERVER=$(terraform output -raw acr_login_server 2>/dev/null || echo "")
        ACR_USERNAME=$(terraform output -raw acr_admin_username 2>/dev/null || echo "")
        ACR_PASSWORD=$(terraform output -raw acr_admin_password 2>/dev/null || echo "")
        
        if [ -n "$ACR_SERVER" ]; then
            # Login to ACR
            echo "$ACR_PASSWORD" | docker login $ACR_SERVER --username $ACR_USERNAME --password-stdin
            
            print_warning "Building and pushing API image..."
            docker build -t $ACR_SERVER/${PROJECT_NAME}/api:latest -f ../../docker/api/Dockerfile ../../
            docker push $ACR_SERVER/${PROJECT_NAME}/api:latest
            
            print_warning "Building and pushing Dashboard image..."
            docker build -t $ACR_SERVER/${PROJECT_NAME}/dashboard:latest -f ../../docker/dashboard/Dockerfile ../../
            docker push $ACR_SERVER/${PROJECT_NAME}/dashboard:latest
            
            print_warning "Building and pushing MLflow image..."
            docker build -t $ACR_SERVER/${PROJECT_NAME}/mlflow:latest -f ../../docker/mlflow/Dockerfile ../../
            docker push $ACR_SERVER/${PROJECT_NAME}/mlflow:latest
            
            print_success "Images built and pushed successfully"
        else
            print_warning "ACR not yet created. Images will be built after infrastructure deployment."
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
    if [ "$BUILD_IMAGES" = "true" ] && [ -z "$ACR_SERVER" ]; then
        build_and_push_images
        
        # Restart container groups to use new images
        print_warning "Restarting container groups with new images..."
        RG_NAME=$(terraform output -raw resource_group_name)
        az container restart --resource-group $RG_NAME --name ${PROJECT_NAME}-api-cg
        az container restart --resource-group $RG_NAME --name ${PROJECT_NAME}-dashboard-cg
        az container restart --resource-group $RG_NAME --name ${PROJECT_NAME}-mlflow-cg
    fi
    
    # Wait for services to be healthy
    print_warning "Waiting for services to become healthy..."
    sleep 60
    
    # Display deployment information
    print_section "Deployment Complete"
    
    PUBLIC_IP=$(terraform output -raw app_gateway_public_ip)
    echo -e "\n${GREEN}🎉 Deployment successful!${NC}\n"
    echo -e "API URL:       http://$PUBLIC_IP"
    echo -e "Dashboard URL: http://$PUBLIC_IP/dashboard"
    echo -e "MLflow URL:    http://$PUBLIC_IP/mlflow"
    echo -e "\nNote: It may take a few minutes for the Application Gateway to route traffic properly."
    
    # Show resource summary
    terraform output deployment_summary 2>/dev/null || true
}

setup_monitoring() {
    print_section "Setting up Azure Monitor (Optional)"
    print_warning "Would you like to enable Azure Monitor Application Insights? (yes/no)"
    read -r response
    if [ "$response" = "yes" ]; then
        RG_NAME=$(terraform output -raw resource_group_name)
        
        # Create Application Insights
        az monitor app-insights component create \
            --resource-group $RG_NAME \
            --app ${PROJECT_NAME}-insights \
            --location "$LOCATION" \
            --kind web \
            --retention-time 90
            
        print_success "Application Insights enabled"
        
        # Get instrumentation key
        INSTRUMENTATION_KEY=$(az monitor app-insights component show \
            --resource-group $RG_NAME \
            --app ${PROJECT_NAME}-insights \
            --query instrumentationKey \
            --output tsv)
            
        echo -e "Instrumentation Key: $INSTRUMENTATION_KEY"
        echo -e "Add this to your application configuration for monitoring."
    fi
}

cleanup() {
    print_section "Cleanup"
    print_warning "This will destroy all Azure resources. Are you sure? (yes/no)"
    read -r response
    if [ "$response" = "yes" ]; then
        terraform destroy -auto-approve
        print_success "Resources destroyed successfully"
    else
        print_warning "Cleanup cancelled"
    fi
}

show_help() {
    echo "Azure Deployment Script for AI Architecture Demo"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy infrastructure and applications (default)"
    echo "  destroy    Destroy all resources"
    echo "  plan       Show deployment plan"
    echo "  monitor    Setup Azure Monitor (optional)"
    echo "  help       Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  AZURE_LOCATION   Azure location (default: East US)"
    echo "  BUILD_IMAGES     Build and push images (default: true)"
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
        "monitor")
            setup_monitoring
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