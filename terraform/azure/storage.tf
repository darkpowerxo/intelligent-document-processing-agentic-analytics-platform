# =============================================================================
# Azure Database for PostgreSQL
# =============================================================================

resource "azurerm_private_dns_zone" "postgres" {
  name                = "${var.project_name}-postgres.private.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name

  tags = local.common_tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "${var.project_name}-postgres-link"
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.main.id
  resource_group_name   = azurerm_resource_group.main.name

  tags = local.common_tags
}

resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "${var.project_name}-postgres"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "15"
  delegated_subnet_id    = azurerm_subnet.database.id
  private_dns_zone_id    = azurerm_private_dns_zone.postgres.id
  administrator_login    = var.db_username
  administrator_password = var.db_password

  zone = "1"

  storage_mb   = var.postgres_storage_mb
  storage_tier = var.postgres_storage_tier
  sku_name     = var.postgres_sku_name

  backup_retention_days        = 7
  geo_redundant_backup_enabled = false

  high_availability {
    mode                      = var.postgres_ha_mode
    standby_availability_zone = var.postgres_ha_mode != "Disabled" ? "2" : null
  }

  maintenance_window {
    day_of_week  = 0
    start_hour   = 8
    start_minute = 0
  }

  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]

  tags = local.common_tags
}

resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = var.db_name
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# =============================================================================
# Azure Cache for Redis
# =============================================================================

resource "azurerm_redis_cache" "main" {
  name                = "${var.project_name}-redis"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku_name
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  
  subnet_id = azurerm_subnet.private.id

  redis_configuration {
    enable_authentication = true
  }

  tags = local.common_tags
}

# =============================================================================
# Azure Event Hubs (Kafka alternative)
# =============================================================================

resource "azurerm_eventhub_namespace" "main" {
  name                = "${var.project_name}-eventhub-ns"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = var.eventhub_sku
  capacity            = var.eventhub_capacity

  kafka_enabled = true

  network_rulesets {
    default_action                 = "Deny"
    public_network_access_enabled = false
    trusted_service_access_enabled = true

    virtual_network_rule {
      subnet_id                                       = azurerm_subnet.private.id
      ignore_missing_virtual_network_service_endpoint = false
    }
  }

  tags = local.common_tags
}

resource "azurerm_eventhub" "events" {
  name                = "${var.project_name}-events"
  namespace_name      = azurerm_eventhub_namespace.main.name
  resource_group_name = azurerm_resource_group.main.name
  partition_count     = 4
  message_retention   = 7
}

resource "azurerm_eventhub" "metrics" {
  name                = "${var.project_name}-metrics"
  namespace_name      = azurerm_eventhub_namespace.main.name
  resource_group_name = azurerm_resource_group.main.name
  partition_count     = 2
  message_retention   = 3
}

# =============================================================================
# Azure Container Registry
# =============================================================================

resource "azurerm_container_registry" "main" {
  name                = replace("${var.project_name}acr${random_string.acr_suffix.result}", "-", "")
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.acr_sku
  admin_enabled       = true

  public_network_access_enabled = true
  quarantine_policy_enabled     = true
  retention_policy {
    days    = 7
    enabled = true
  }

  trust_policy {
    enabled = true
  }

  tags = local.common_tags
}

resource "random_string" "acr_suffix" {
  length  = 6
  special = false
  upper   = false
}

# =============================================================================
# Azure Storage Account for MLflow Artifacts
# =============================================================================

resource "azurerm_storage_account" "mlflow" {
  name                     = replace("${var.project_name}mlflow${random_string.storage_suffix.result}", "-", "")
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = var.storage_replication_type

  blob_properties {
    versioning_enabled = true
    
    delete_retention_policy {
      days = 7
    }
  }

  network_rules {
    default_action             = "Deny"
    virtual_network_subnet_ids = [azurerm_subnet.private.id]
    bypass                     = ["AzureServices"]
  }

  tags = local.common_tags
}

resource "azurerm_storage_container" "mlflow_artifacts" {
  name                  = "artifacts"
  storage_account_name  = azurerm_storage_account.mlflow.name
  container_access_type = "private"
}

resource "random_string" "storage_suffix" {
  length  = 6
  special = false
  upper   = false
}

# =============================================================================
# Azure Key Vault
# =============================================================================

resource "azurerm_key_vault" "main" {
  name                        = "${var.project_name}-kv-${random_string.kv_suffix.result}"
  location                    = azurerm_resource_group.main.location
  resource_group_name         = azurerm_resource_group.main.name
  enabled_for_disk_encryption = true
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  soft_delete_retention_days  = 7
  purge_protection_enabled    = false
  sku_name                    = "standard"

  network_acls {
    bypass         = "AzureServices"
    default_action = "Deny"
    virtual_network_subnet_ids = [azurerm_subnet.private.id]
  }

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    key_permissions = [
      "Get", "List", "Update", "Create", "Import", "Delete", "Recover",
      "Backup", "Restore", "Decrypt", "Encrypt", "UnwrapKey", "WrapKey",
      "Verify", "Sign", "Purge"
    ]

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Recover", "Backup", "Restore", "Purge"
    ]

    certificate_permissions = [
      "Get", "List", "Update", "Create", "Import", "Delete", "Recover",
      "Backup", "Restore", "ManageContacts", "ManageIssuers", "GetIssuers",
      "ListIssuers", "SetIssuers", "DeleteIssuers", "Purge"
    ]
  }

  tags = local.common_tags
}

resource "azurerm_key_vault_secret" "db_password" {
  name         = "db-password"
  value        = var.db_password
  key_vault_id = azurerm_key_vault.main.id

  tags = local.common_tags
}

resource "azurerm_key_vault_secret" "redis_key" {
  name         = "redis-key"
  value        = azurerm_redis_cache.main.primary_access_key
  key_vault_id = azurerm_key_vault.main.id

  tags = local.common_tags
}

resource "azurerm_key_vault_secret" "eventhub_connection" {
  name         = "eventhub-connection"
  value        = azurerm_eventhub_namespace.main.default_primary_connection_string
  key_vault_id = azurerm_key_vault.main.id

  tags = local.common_tags
}

resource "random_string" "kv_suffix" {
  length  = 6
  special = false
  upper   = false
}