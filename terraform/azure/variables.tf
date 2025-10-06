# =============================================================================
# Azure Variables
# =============================================================================

variable "azure_location" {
  description = "Azure region"
  type        = string
  default     = "East US"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "ai-architect-demo"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "ai-team"
}

# =============================================================================
# Network Variables
# =============================================================================

variable "vnet_cidr" {
  description = "CIDR block for VNet"
  type        = string
  default     = "10.0.0.0/16"
}

# =============================================================================
# Database Variables
# =============================================================================

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "ai_demo"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "postgres"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
  
  validation {
    condition     = length(var.db_password) >= 8
    error_message = "Database password must be at least 8 characters long."
  }
}

variable "postgres_sku_name" {
  description = "PostgreSQL SKU name"
  type        = string
  default     = "B_Standard_B1ms"
}

variable "postgres_storage_mb" {
  description = "PostgreSQL storage in MB"
  type        = number
  default     = 32768
}

variable "postgres_storage_tier" {
  description = "PostgreSQL storage tier"
  type        = string
  default     = "P4"
}

variable "postgres_ha_mode" {
  description = "PostgreSQL high availability mode"
  type        = string
  default     = "Disabled"
  
  validation {
    condition     = contains(["Disabled", "ZoneRedundant", "SameZone"], var.postgres_ha_mode)
    error_message = "PostgreSQL HA mode must be one of: Disabled, ZoneRedundant, SameZone."
  }
}

# =============================================================================
# Redis Variables
# =============================================================================

variable "redis_capacity" {
  description = "Redis cache capacity"
  type        = number
  default     = 0
}

variable "redis_family" {
  description = "Redis cache family"
  type        = string
  default     = "C"
}

variable "redis_sku_name" {
  description = "Redis cache SKU name"
  type        = string
  default     = "Basic"
}

# =============================================================================
# Event Hub Variables
# =============================================================================

variable "eventhub_sku" {
  description = "Event Hub namespace SKU"
  type        = string
  default     = "Standard"
}

variable "eventhub_capacity" {
  description = "Event Hub namespace capacity (throughput units)"
  type        = number
  default     = 1
}

# =============================================================================
# Container Registry Variables
# =============================================================================

variable "acr_sku" {
  description = "Azure Container Registry SKU"
  type        = string
  default     = "Standard"
}

# =============================================================================
# Storage Variables
# =============================================================================

variable "storage_replication_type" {
  description = "Storage account replication type"
  type        = string
  default     = "LRS"
}

# =============================================================================
# Application Gateway Variables
# =============================================================================

variable "app_gateway_sku_name" {
  description = "Application Gateway SKU name"
  type        = string
  default     = "Standard_v2"
}

variable "app_gateway_sku_tier" {
  description = "Application Gateway SKU tier"
  type        = string
  default     = "Standard_v2"
}

variable "app_gateway_capacity" {
  description = "Application Gateway capacity"
  type        = number
  default     = 2
}

# =============================================================================
# Container Instance Variables
# =============================================================================

variable "api_container_cpu" {
  description = "CPU cores for API container"
  type        = number
  default     = 1
}

variable "api_container_memory" {
  description = "Memory in GB for API container"
  type        = number
  default     = 2
}

variable "dashboard_container_cpu" {
  description = "CPU cores for Dashboard container"
  type        = number
  default     = 0.5
}

variable "dashboard_container_memory" {
  description = "Memory in GB for Dashboard container"
  type        = number
  default     = 1
}

variable "mlflow_container_cpu" {
  description = "CPU cores for MLflow container"
  type        = number
  default     = 1
}

variable "mlflow_container_memory" {
  description = "Memory in GB for MLflow container"
  type        = number
  default     = 2
}

# =============================================================================
# Monitoring Variables
# =============================================================================

variable "alert_email" {
  description = "Email address for alerts"
  type        = string
  default     = ""
}