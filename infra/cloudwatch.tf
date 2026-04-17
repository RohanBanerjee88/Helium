resource "aws_cloudwatch_log_group" "backend" {
  name              = "/ecs/${var.project_name}-${var.environment}/backend"
  retention_in_days = var.log_retention_days

  tags = { Name = "${var.project_name}-${var.environment}-backend-logs" }
}
