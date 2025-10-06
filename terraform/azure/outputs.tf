# =============================================================================
# Azure Outputs
# =============================================================================

output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "resource_group_location" {
  description = "Location of the resource group"
  value       = azurerm_resource_group.main.location
}

output "vnet_id" {
  description = "ID of the virtual network"
  value       = azurerm_virtual_network.main.id
}

output "vnet_name" {
  description = "Name of the virtual network"
  value       = azurerm_virtual_network.main.name
}

# =============================================================================
# Database Outputs
# =============================================================================

output "postgres_fqdn" {
  description = "PostgreSQL server FQDN"
  value       = azurerm_postgresql_flexible_server.main.fqdn
  sensitive   = true
}

output "postgres_server_name" {
  description = "PostgreSQL server name"
  value       = azurerm_postgresql_flexible_server.main.name
}

output "redis_hostname" {
  description = "Redis cache hostname"
  value       = azurerm_redis_cache.main.hostname
  sensitive   = true
}

output "redis_ssl_port" {
  description = "Redis cache SSL port"
  value       = azurerm_redis_cache.main.ssl_port
}

# =============================================================================
# Event Hub Outputs
# =============================================================================

output "eventhub_namespace_name" {
  description = "Event Hub namespace name"
  value       = azurerm_eventhub_namespace.main.name
}

output "eventhub_connection_string" {
  description = "Event Hub connection string"
  value       = azurerm_eventhub_namespace.main.default_primary_connection_string
  sensitive   = true
}

output "eventhub_kafka_endpoint" {
  description = "Event Hub Kafka endpoint"
  value       = "${azurerm_eventhub_namespace.main.name}.servicebus.windows.net:9093"
  sensitive   = true
}

# =============================================================================
# Container Registry Outputs
# =============================================================================

output "acr_login_server" {
  description = "Azure Container Registry login server"
  value       = azurerm_container_registry.main.login_server
}

output "acr_admin_username" {
  description = "Azure Container Registry admin username"
  value       = azurerm_container_registry.main.admin_username
  sensitive   = true
}

output "acr_admin_password" {
  description = "Azure Container Registry admin password"
  value       = azurerm_container_registry.main.admin_password
  sensitive   = true
}

# =============================================================================
# Storage Outputs
# =============================================================================

output "storage_account_name" {
  description = "Storage account name for MLflow artifacts"
  value       = azurerm_storage_account.mlflow.name
}

output "storage_account_primary_endpoint" {
  description = "Storage account primary blob endpoint"
  value       = azurerm_storage_account.mlflow.primary_blob_endpoint
}

output "mlflow_container_name" {
  description = "Storage container name for MLflow artifacts"
  value       = azurerm_storage_container.mlflow_artifacts.name
}

# =============================================================================
# Key Vault Outputs
# =============================================================================

output "key_vault_uri" {
  description = "Key Vault URI"
  value       = azurerm_key_vault.main.vault_uri
  sensitive   = true
}

output "key_vault_name" {
  description = "Key Vault name"
  value       = azurerm_key_vault.main.name
}

# =============================================================================
# Application Gateway Outputs
# =============================================================================

output "app_gateway_public_ip" {
  description = "Application Gateway public IP address"
  value       = azurerm_public_ip.main.ip_address
}

output "app_gateway_fqdn" {
  description = "Application Gateway FQDN"
  value       = azurerm_public_ip.main.fqdn
}

# =============================================================================
# Container Instances Outputs
# =============================================================================

output "api_container_ip" {
  description = "API container group private IP"
  value       = azurerm_container_group.api.ip_address
  sensitive   = true
}

output "dashboard_container_ip" {
  description = "Dashboard container group private IP"
  value       = azurerm_container_group.dashboard.ip_address
  sensitive   = true
}

output "mlflow_container_ip" {
  description = "MLflow container group private IP"
  value       = azurerm_container_group.mlflow.ip_address
  sensitive   = true
}

# =============================================================================
# Application URLs
# =============================================================================

output "api_url" {
  description = "URL of the API service"
  value       = "http://${azurerm_public_ip.main.ip_address}"
}

output "dashboard_url" {
  description = "URL of the Dashboard service"
  value       = "http://${azurerm_public_ip.main.ip_address}/dashboard"
}

output "mlflow_url" {
  description = "URL of the MLflow service"
  value       = "http://${azurerm_public_ip.main.ip_address}/mlflow"
}

# =============================================================================
# Identity Outputs
# =============================================================================

output "container_identity_client_id" {
  description = "User assigned identity client ID"
  value       = azurerm_user_assigned_identity.container_identity.client_id
  sensitive   = true
}

output "container_identity_principal_id" {
  description = "User assigned identity principal ID"
  value       = azurerm_user_assigned_identity.container_identity.principal_id
  sensitive   = true
}

# =============================================================================
# Deployment Summary
# =============================================================================

output "deployment_summary" {
  description = "Summary of deployed resources"
  value = {
    location                = var.azure_location
    environment            = var.environment
    resource_group         = azurerm_resource_group.main.name
    vnet_name              = azurerm_virtual_network.main.name
    postgres_server        = azurerm_postgresql_flexible_server.main.name
    redis_cache            = azurerm_redis_cache.main.name
    eventhub_namespace     = azurerm_eventhub_namespace.main.name
    container_registry     = azurerm_container_registry.main.name
    storage_account        = azurerm_storage_account.mlflow.name
    key_vault              = azurerm_key_vault.main.name
    application_gateway    = azurerm_application_gateway.main.name
    public_ip              = azurerm_public_ip.main.ip_address
    api_url               = "http://${azurerm_public_ip.main.ip_address}"
    dashboard_url         = "http://${azurerm_public_ip.main.ip_address}/dashboard"
    mlflow_url            = "http://${azurerm_public_ip.main.ip_address}/mlflow"
  }
  sensitive = true
}