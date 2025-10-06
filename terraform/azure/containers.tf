# =============================================================================
# Azure Container Instances
# =============================================================================

# User Assigned Identity for Container Groups
resource "azurerm_user_assigned_identity" "container_identity" {
  location            = azurerm_resource_group.main.location
  name                = "${var.project_name}-container-identity"
  resource_group_name = azurerm_resource_group.main.name

  tags = local.common_tags
}

# Role Assignment for Key Vault access
resource "azurerm_role_assignment" "key_vault_secrets_user" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.container_identity.principal_id
}

# Role Assignment for Storage Blob access
resource "azurerm_role_assignment" "storage_blob_contributor" {
  scope                = azurerm_storage_account.mlflow.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.container_identity.principal_id
}

# Role Assignment for ACR pull
resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.container_identity.principal_id
}

# =============================================================================
# API Server Container Group
# =============================================================================

resource "azurerm_container_group" "api" {
  name                = "${var.project_name}-api-cg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  ip_address_type     = "Private"
  subnet_ids          = [azurerm_subnet.private.id]
  os_type             = "Linux"
  restart_policy      = "Always"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_identity.id]
  }

  image_registry_credential {
    server   = azurerm_container_registry.main.login_server
    username = azurerm_container_registry.main.admin_username
    password = azurerm_container_registry.main.admin_password
  }

  container {
    name   = "api"
    image  = "${azurerm_container_registry.main.login_server}/${var.project_name}/api:latest"
    cpu    = var.api_container_cpu
    memory = var.api_container_memory

    ports {
      port     = 8000
      protocol = "TCP"
    }

    environment_variables = {
      POSTGRES_HOST         = azurerm_postgresql_flexible_server.main.fqdn
      POSTGRES_DB           = var.db_name
      POSTGRES_USER         = var.db_username
      REDIS_HOST            = azurerm_redis_cache.main.hostname
      REDIS_PORT            = azurerm_redis_cache.main.ssl_port
      REDIS_SSL             = "true"
      EVENTHUB_NAMESPACE    = azurerm_eventhub_namespace.main.name
      EVENTHUB_NAME         = azurerm_eventhub.events.name
      STORAGE_ACCOUNT_NAME  = azurerm_storage_account.mlflow.name
      STORAGE_CONTAINER     = azurerm_storage_container.mlflow_artifacts.name
      KEY_VAULT_URL         = azurerm_key_vault.main.vault_uri
      AZURE_CLIENT_ID       = azurerm_user_assigned_identity.container_identity.client_id
    }

    liveness_probe {
      http_get {
        path   = "/health"
        port   = 8000
        scheme = "Http"
      }
      initial_delay_seconds = 30
      period_seconds        = 30
      timeout_seconds       = 5
      failure_threshold     = 3
    }

    readiness_probe {
      http_get {
        path   = "/health"
        port   = 8000
        scheme = "Http"
      }
      initial_delay_seconds = 10
      period_seconds        = 10
      timeout_seconds       = 5
      failure_threshold     = 3
    }

    volume {
      name       = "logs"
      mount_path = "/app/logs"
      
      empty_dir {}
    }
  }

  tags = local.common_tags

  depends_on = [
    azurerm_role_assignment.key_vault_secrets_user,
    azurerm_role_assignment.storage_blob_contributor,
    azurerm_role_assignment.acr_pull
  ]
}

# =============================================================================
# Dashboard Container Group
# =============================================================================

resource "azurerm_container_group" "dashboard" {
  name                = "${var.project_name}-dashboard-cg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  ip_address_type     = "Private"
  subnet_ids          = [azurerm_subnet.private.id]
  os_type             = "Linux"
  restart_policy      = "Always"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_identity.id]
  }

  image_registry_credential {
    server   = azurerm_container_registry.main.login_server
    username = azurerm_container_registry.main.admin_username
    password = azurerm_container_registry.main.admin_password
  }

  container {
    name   = "dashboard"
    image  = "${azurerm_container_registry.main.login_server}/${var.project_name}/dashboard:latest"
    cpu    = var.dashboard_container_cpu
    memory = var.dashboard_container_memory

    ports {
      port     = 8501
      protocol = "TCP"
    }

    environment_variables = {
      API_BASE_URL = "http://${azurerm_container_group.api.ip_address}:8000"
    }

    volume {
      name       = "logs"
      mount_path = "/app/logs"
      
      empty_dir {}
    }
  }

  tags = local.common_tags

  depends_on = [
    azurerm_container_group.api,
    azurerm_role_assignment.acr_pull
  ]
}

# =============================================================================
# MLflow Container Group
# =============================================================================

resource "azurerm_container_group" "mlflow" {
  name                = "${var.project_name}-mlflow-cg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  ip_address_type     = "Private"
  subnet_ids          = [azurerm_subnet.private.id]
  os_type             = "Linux"
  restart_policy      = "Always"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_identity.id]
  }

  image_registry_credential {
    server   = azurerm_container_registry.main.login_server
    username = azurerm_container_registry.main.admin_username
    password = azurerm_container_registry.main.admin_password
  }

  container {
    name   = "mlflow"
    image  = "${azurerm_container_registry.main.login_server}/${var.project_name}/mlflow:latest"
    cpu    = var.mlflow_container_cpu
    memory = var.mlflow_container_memory

    ports {
      port     = 5000
      protocol = "TCP"
    }

    environment_variables = {
      POSTGRES_HOST         = azurerm_postgresql_flexible_server.main.fqdn
      POSTGRES_DB           = var.db_name
      POSTGRES_USER         = var.db_username
      STORAGE_ACCOUNT_NAME  = azurerm_storage_account.mlflow.name
      STORAGE_CONTAINER     = azurerm_storage_container.mlflow_artifacts.name
      KEY_VAULT_URL         = azurerm_key_vault.main.vault_uri
      AZURE_CLIENT_ID       = azurerm_user_assigned_identity.container_identity.client_id
    }

    liveness_probe {
      http_get {
        path   = "/health"
        port   = 5000
        scheme = "Http"
      }
      initial_delay_seconds = 60
      period_seconds        = 30
      timeout_seconds       = 5
      failure_threshold     = 3
    }

    volume {
      name       = "logs"
      mount_path = "/app/logs"
      
      empty_dir {}
    }
  }

  tags = local.common_tags

  depends_on = [
    azurerm_role_assignment.key_vault_secrets_user,
    azurerm_role_assignment.storage_blob_contributor,
    azurerm_role_assignment.acr_pull
  ]
}

# =============================================================================
# Application Gateway Backend Address Pool Associations
# =============================================================================

resource "azurerm_application_gateway_backend_address_pool_address" "api" {
  name                    = "api-backend-address"
  backend_address_pool_id = "${azurerm_application_gateway.main.id}/backendAddressPools/api-backend-pool"
  ip_address              = azurerm_container_group.api.ip_address
}

resource "azurerm_application_gateway_backend_address_pool_address" "dashboard" {
  name                    = "dashboard-backend-address"
  backend_address_pool_id = "${azurerm_application_gateway.main.id}/backendAddressPools/dashboard-backend-pool"
  ip_address              = azurerm_container_group.dashboard.ip_address
}

resource "azurerm_application_gateway_backend_address_pool_address" "mlflow" {
  name                    = "mlflow-backend-address"
  backend_address_pool_id = "${azurerm_application_gateway.main.id}/backendAddressPools/mlflow-backend-pool"
  ip_address              = azurerm_container_group.mlflow.ip_address
}