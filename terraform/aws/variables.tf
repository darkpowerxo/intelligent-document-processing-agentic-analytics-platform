# =============================================================================
# AWS Variables
# =============================================================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
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

variable "vpc_cidr" {
  description = "CIDR block for VPC"
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

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 100
}

# =============================================================================
# Redis Variables
# =============================================================================

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_auth_token" {
  description = "Redis authentication token"
  type        = string
  sensitive   = true
  
  validation {
    condition     = length(var.redis_auth_token) >= 16
    error_message = "Redis auth token must be at least 16 characters long."
  }
}

# =============================================================================
# Kafka Variables
# =============================================================================

variable "kafka_instance_type" {
  description = "Kafka instance type"
  type        = string
  default     = "kafka.t3.small"
}

variable "kafka_volume_size" {
  description = "Kafka EBS volume size in GB"
  type        = number
  default     = 100
}

# =============================================================================
# ECS Variables
# =============================================================================

variable "api_task_cpu" {
  description = "CPU units for API task (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "api_task_memory" {
  description = "Memory for API task in MiB"
  type        = number
  default     = 1024
}

variable "api_desired_count" {
  description = "Desired number of API tasks"
  type        = number
  default     = 2
}

variable "api_min_capacity" {
  description = "Minimum number of API tasks for auto scaling"
  type        = number
  default     = 1
}

variable "api_max_capacity" {
  description = "Maximum number of API tasks for auto scaling"
  type        = number
  default     = 10
}

variable "dashboard_task_cpu" {
  description = "CPU units for Dashboard task (1024 = 1 vCPU)"
  type        = number
  default     = 256
}

variable "dashboard_task_memory" {
  description = "Memory for Dashboard task in MiB"
  type        = number
  default     = 512
}

variable "dashboard_desired_count" {
  description = "Desired number of Dashboard tasks"
  type        = number
  default     = 1
}

# =============================================================================
# Monitoring Variables
# =============================================================================

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

variable "alert_email" {
  description = "Email address for alerts"
  type        = string
  default     = ""
}