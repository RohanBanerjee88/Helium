resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Name = "${var.project_name}-${var.environment}-cluster" }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
  }
}

resource "aws_ecs_task_definition" "backend" {
  family                   = "${var.project_name}-${var.environment}-backend"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "backend"
      image     = "${aws_ecr_repository.backend.repository_url}:latest"
      essential = true

      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]

      environment = [
        { name = "HELIUM_HOST",              value = "0.0.0.0" },
        { name = "HELIUM_PORT",              value = tostring(var.container_port) },
        { name = "HELIUM_DATA_DIR",          value = var.helium_data_dir },
        { name = "HELIUM_MIN_IMAGES",        value = tostring(var.helium_min_images) },
        { name = "HELIUM_MAX_IMAGES",        value = tostring(var.helium_max_images) },
        { name = "HELIUM_MAX_IMAGE_SIZE_MB", value = tostring(var.helium_max_image_size_mb) },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.backend.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      # Do not run as root
      user = "1000"

      # Prevent privilege escalation inside the container
      privileged             = false
      readonlyRootFilesystem = false
    }
  ])

  tags = { Name = "${var.project_name}-${var.environment}-backend-td" }
}

resource "aws_ecs_service" "backend" {
  name            = "${var.project_name}-${var.environment}-backend"
  cluster         = aws_ecs_cluster.main.arn
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = var.desired_count

  # FARGATE_SPOT saves ~70% vs on-demand. ECS replaces interrupted tasks
  # automatically. Switch weight to FARGATE=1/FARGATE_SPOT=0 for production.
  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }

  # Allow the ALB health checks to warm up before ECS counts failures
  health_check_grace_period_seconds = 60

  # Trigger a new deployment whenever the task definition changes in Terraform
  force_new_deployment = true

  # Tasks run in public subnets and pull ECR images directly over the internet.
  # Security groups still restrict all inbound to the ALB only.
  # Move to private subnets and set assign_public_ip = false when adding a NAT Gateway.
  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.backend.arn
    container_name   = "backend"
    container_port   = var.container_port
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  deployment_controller {
    type = "ECS"
  }

  depends_on = [
    aws_lb_listener.http_forward,
    aws_lb_listener.http_redirect,
    aws_lb_listener.https,
    aws_iam_role_policy_attachment.task_execution_managed,
  ]

  tags = { Name = "${var.project_name}-${var.environment}-backend-svc" }

  lifecycle {
    # CI/CD manages the running image via task definition updates — ignore drift here
    ignore_changes = [task_definition]
  }
}
