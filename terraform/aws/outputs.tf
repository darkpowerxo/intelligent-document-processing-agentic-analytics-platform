# =============================================================================
# AWS Outputs
# =============================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

# =============================================================================
# Database Outputs
# =============================================================================

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.postgres.port
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = aws_elasticache_replication_group.redis.port
}

# =============================================================================
# Kafka Outputs
# =============================================================================

output "kafka_bootstrap_brokers" {
  description = "MSK Kafka bootstrap brokers"
  value       = aws_msk_cluster.kafka.bootstrap_brokers_tls
  sensitive   = true
}

output "kafka_zookeeper_connect" {
  description = "MSK Zookeeper connection string"
  value       = aws_msk_cluster.kafka.zookeeper_connect_string
  sensitive   = true
}

# =============================================================================
# ECS Outputs
# =============================================================================

output "ecs_cluster_id" {
  description = "ID of the ECS cluster"
  value       = aws_ecs_cluster.main.id
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

# =============================================================================
# ECR Outputs
# =============================================================================

output "ecr_api_repository_url" {
  description = "ECR repository URL for API"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_dashboard_repository_url" {
  description = "ECR repository URL for Dashboard"
  value       = aws_ecr_repository.dashboard.repository_url
}

output "ecr_mlflow_repository_url" {
  description = "ECR repository URL for MLflow"
  value       = aws_ecr_repository.mlflow.repository_url
}

# =============================================================================
# Load Balancer Outputs
# =============================================================================

output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.main.zone_id
}

output "api_url" {
  description = "URL of the API service"
  value       = "http://${aws_lb.main.dns_name}"
}

output "dashboard_url" {
  description = "URL of the Dashboard service"
  value       = "http://${aws_lb.main.dns_name}/dashboard"
}

output "mlflow_url" {
  description = "URL of the MLflow service"
  value       = "http://${aws_lb.main.dns_name}/mlflow"
}

# =============================================================================
# S3 Outputs
# =============================================================================

output "mlflow_artifacts_bucket" {
  description = "S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "mlflow_artifacts_bucket_arn" {
  description = "ARN of the MLflow artifacts S3 bucket"
  value       = aws_s3_bucket.mlflow_artifacts.arn
}

# =============================================================================
# Monitoring Outputs
# =============================================================================

output "sns_alerts_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = aws_sns_topic.alerts.arn
}

# =============================================================================
# Connection Information
# =============================================================================

output "deployment_summary" {
  description = "Summary of deployed resources"
  value = {
    region              = var.aws_region
    environment         = var.environment
    vpc_id              = aws_vpc.main.id
    ecs_cluster         = aws_ecs_cluster.main.name
    api_url             = "http://${aws_lb.main.dns_name}"
    dashboard_url       = "http://${aws_lb.main.dns_name}/dashboard"
    mlflow_url          = "http://${aws_lb.main.dns_name}/mlflow"
    rds_endpoint        = aws_db_instance.postgres.endpoint
    redis_endpoint      = aws_elasticache_replication_group.redis.primary_endpoint_address
    kafka_brokers       = aws_msk_cluster.kafka.bootstrap_brokers_tls
    s3_bucket           = aws_s3_bucket.mlflow_artifacts.bucket
  }
  sensitive = true
}