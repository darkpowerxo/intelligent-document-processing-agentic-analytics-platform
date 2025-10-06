# =============================================================================
# Azure Terraform Configuration for AI Architecture Demo
# =============================================================================

terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }

  backend "azurerm" {
    # Configure your Azure backend here
    # resource_group_name  = "terraform-state-rg"
    # storage_account_name = "terraformstatestorage"
    # container_name       = "terraform-state"
    # key                  = "ai-architect-demo.tfstate"
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
  }
}

# =============================================================================
# Data Sources
# =============================================================================

data "azurerm_client_config" "current" {}

# =============================================================================
# Local Values
# =============================================================================

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = var.owner
  }
  
  location_short = {
    "East US"         = "eus"
    "East US 2"       = "eus2"
    "West US"         = "wus"
    "West US 2"       = "wus2"
    "Central US"      = "cus"
    "North Central US" = "ncus"
    "South Central US" = "scus"
    "West Central US"  = "wcus"
  }
}

# =============================================================================
# Resource Group
# =============================================================================

resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-rg"
  location = var.azure_location

  tags = local.common_tags
}

# =============================================================================
# Virtual Network
# =============================================================================

resource "azurerm_virtual_network" "main" {
  name                = "${var.project_name}-vnet"
  address_space       = [var.vnet_cidr]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = local.common_tags
}

resource "azurerm_subnet" "public" {
  name                 = "${var.project_name}-public-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 8, 1)]
}

resource "azurerm_subnet" "private" {
  name                 = "${var.project_name}-private-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 8, 2)]

  delegation {
    name = "container-delegation"
    service_delegation {
      name    = "Microsoft.ContainerInstance/containerGroups"
      actions = ["Microsoft.Network/virtualNetworks/subnets/action"]
    }
  }
}

resource "azurerm_subnet" "database" {
  name                 = "${var.project_name}-database-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 8, 3)]

  delegation {
    name = "postgresql-delegation"
    service_delegation {
      name = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action"
      ]
    }
  }
}

# =============================================================================
# Network Security Groups
# =============================================================================

resource "azurerm_network_security_group" "main" {
  name                = "${var.project_name}-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "AllowHTTP"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowAPI"
    priority                   = 1003
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8000"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowDashboard"
    priority                   = 1004
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8501"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowMLflow"
    priority                   = 1005
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "5000"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = local.common_tags
}

resource "azurerm_subnet_network_security_group_association" "public" {
  subnet_id                 = azurerm_subnet.public.id
  network_security_group_id = azurerm_network_security_group.main.id
}

resource "azurerm_subnet_network_security_group_association" "private" {
  subnet_id                 = azurerm_subnet.private.id
  network_security_group_id = azurerm_network_security_group.main.id
}

# =============================================================================
# Public IP and Load Balancer
# =============================================================================

resource "azurerm_public_ip" "main" {
  name                = "${var.project_name}-public-ip"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = local.common_tags
}

resource "azurerm_application_gateway" "main" {
  name                = "${var.project_name}-appgw"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  sku {
    name     = var.app_gateway_sku_name
    tier     = var.app_gateway_sku_tier
    capacity = var.app_gateway_capacity
  }

  gateway_ip_configuration {
    name      = "appGatewayIpConfig"
    subnet_id = azurerm_subnet.public.id
  }

  frontend_port {
    name = "port_80"
    port = 80
  }

  frontend_port {
    name = "port_443"
    port = 443
  }

  frontend_ip_configuration {
    name                 = "appGatewayFrontendIP"
    public_ip_address_id = azurerm_public_ip.main.id
  }

  # API Backend Pool
  backend_address_pool {
    name = "api-backend-pool"
  }

  # Dashboard Backend Pool
  backend_address_pool {
    name = "dashboard-backend-pool"
  }

  # MLflow Backend Pool
  backend_address_pool {
    name = "mlflow-backend-pool"
  }

  # Backend HTTP Settings
  backend_http_settings {
    name                  = "api-http-settings"
    cookie_based_affinity = "Disabled"
    port                  = 8000
    protocol              = "Http"
    request_timeout       = 60
    probe_name            = "api-health-probe"
  }

  backend_http_settings {
    name                  = "dashboard-http-settings"
    cookie_based_affinity = "Disabled"
    port                  = 8501
    protocol              = "Http"
    request_timeout       = 60
    probe_name            = "dashboard-health-probe"
  }

  backend_http_settings {
    name                  = "mlflow-http-settings"
    cookie_based_affinity = "Disabled"
    port                  = 5000
    protocol              = "Http"
    request_timeout       = 60
    probe_name            = "mlflow-health-probe"
  }

  # Health Probes
  probe {
    name                                      = "api-health-probe"
    protocol                                  = "Http"
    path                                      = "/health"
    host                                      = "127.0.0.1"
    interval                                  = 30
    timeout                                   = 30
    unhealthy_threshold                       = 3
    pick_host_name_from_backend_http_settings = false

    match {
      status_code = ["200"]
    }
  }

  probe {
    name                                      = "dashboard-health-probe"
    protocol                                  = "Http"
    path                                      = "/_stcore/health"
    host                                      = "127.0.0.1"
    interval                                  = 30
    timeout                                   = 30
    unhealthy_threshold                       = 3
    pick_host_name_from_backend_http_settings = false

    match {
      status_code = ["200"]
    }
  }

  probe {
    name                                      = "mlflow-health-probe"
    protocol                                  = "Http"
    path                                      = "/health"
    host                                      = "127.0.0.1"
    interval                                  = 30
    timeout                                   = 30
    unhealthy_threshold                       = 3
    pick_host_name_from_backend_http_settings = false

    match {
      status_code = ["200"]
    }
  }

  # HTTP Listeners
  http_listener {
    name                           = "api-listener"
    frontend_ip_configuration_name = "appGatewayFrontendIP"
    frontend_port_name             = "port_80"
    protocol                       = "Http"
  }

  # Request Routing Rules
  request_routing_rule {
    name                       = "api-routing-rule"
    rule_type                  = "PathBasedRouting"
    http_listener_name         = "api-listener"
    backend_address_pool_name  = "api-backend-pool"
    backend_http_settings_name = "api-http-settings"
    url_path_map_name          = "path-map"
    priority                   = 100
  }

  # URL Path Map
  url_path_map {
    name                               = "path-map"
    default_backend_address_pool_name  = "api-backend-pool"
    default_backend_http_settings_name = "api-http-settings"

    path_rule {
      name                       = "dashboard-path"
      paths                      = ["/dashboard*"]
      backend_address_pool_name  = "dashboard-backend-pool"
      backend_http_settings_name = "dashboard-http-settings"
    }

    path_rule {
      name                       = "mlflow-path"
      paths                      = ["/mlflow*"]
      backend_address_pool_name  = "mlflow-backend-pool"
      backend_http_settings_name = "mlflow-http-settings"
    }
  }

  tags = local.common_tags
}