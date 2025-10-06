# =============================================================================
# ECS and Container Services
# =============================================================================

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.common_tags
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = "FARGATE"
  }
}

# =============================================================================
# ECR Repositories
# =============================================================================

resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}/api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.common_tags
}

resource "aws_ecr_repository" "dashboard" {
  name                 = "${var.project_name}/dashboard"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.common_tags
}

resource "aws_ecr_repository" "mlflow" {
  name                 = "${var.project_name}/mlflow"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.common_tags
}

# =============================================================================
# Application Load Balancer
# =============================================================================

resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = var.environment == "production"

  tags = local.common_tags
}

resource "aws_lb_target_group" "api" {
  name        = "${var.project_name}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
    port                = "traffic-port"
    protocol            = "HTTP"
  }

  tags = local.common_tags
}

resource "aws_lb_target_group" "dashboard" {
  name        = "${var.project_name}-dashboard-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/_stcore/health"
    matcher             = "200"
    port                = "traffic-port"
    protocol            = "HTTP"
  }

  tags = local.common_tags
}

resource "aws_lb_target_group" "mlflow" {
  name        = "${var.project_name}-mlflow-tg"
  port        = 5000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
    port                = "traffic-port"
    protocol            = "HTTP"
  }

  tags = local.common_tags
}

resource "aws_lb_listener" "main" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  tags = local.common_tags
}

resource "aws_lb_listener_rule" "dashboard" {
  listener_arn = aws_lb_listener.main.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.dashboard.arn
  }

  condition {
    path_pattern {
      values = ["/dashboard*"]
    }
  }

  tags = local.common_tags
}

resource "aws_lb_listener_rule" "mlflow" {
  listener_arn = aws_lb_listener.main.arn
  priority     = 200

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow.arn
  }

  condition {
    path_pattern {
      values = ["/mlflow*"]
    }
  }

  tags = local.common_tags
}

# =============================================================================
# ECS Task Definitions and Services
# =============================================================================

# API Server Task Definition
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project_name}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.api_task_cpu
  memory                   = var.api_task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "api"
      image = "${aws_ecr_repository.api.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "POSTGRES_HOST"
          value = aws_db_instance.postgres.endpoint
        },
        {
          name  = "REDIS_HOST"
          value = aws_elasticache_replication_group.redis.primary_endpoint_address
        },
        {
          name  = "KAFKA_BOOTSTRAP_SERVERS"
          value = aws_msk_cluster.kafka.bootstrap_brokers_tls
        },
        {
          name  = "MLFLOW_TRACKING_URI"
          value = "http://${aws_lb.main.dns_name}/mlflow"
        }
      ]

      secrets = [
        {
          name      = "POSTGRES_PASSWORD"
          valueFrom = aws_ssm_parameter.db_password.arn
        },
        {
          name      = "REDIS_PASSWORD"
          valueFrom = aws_ssm_parameter.redis_auth_token.arn
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = local.common_tags
}

# API Service
resource "aws_ecs_service" "api" {
  name            = "${var.project_name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_desired_count

  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 100
  }

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.main]

  tags = local.common_tags
}

# Dashboard Task Definition
resource "aws_ecs_task_definition" "dashboard" {
  family                   = "${var.project_name}-dashboard"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.dashboard_task_cpu
  memory                   = var.dashboard_task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "dashboard"
      image = "${aws_ecr_repository.dashboard.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8501
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "API_BASE_URL"
          value = "http://${aws_lb.main.dns_name}"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.dashboard.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])

  tags = local.common_tags
}

# Dashboard Service
resource "aws_ecs_service" "dashboard" {
  name            = "${var.project_name}-dashboard"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.dashboard.arn
  desired_count   = var.dashboard_desired_count

  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 100
  }

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.dashboard.arn
    container_name   = "dashboard"
    container_port   = 8501
  }

  depends_on = [aws_lb_listener_rule.dashboard]

  tags = local.common_tags
}

# =============================================================================
# CloudWatch Log Groups
# =============================================================================

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project_name}/api"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "dashboard" {
  name              = "/ecs/${var.project_name}/dashboard"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/${var.project_name}/mlflow"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}