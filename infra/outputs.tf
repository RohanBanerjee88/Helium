output "alb_dns_name" {
  description = "ALB DNS name — point your domain CNAME here, or use directly for testing"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "ALB hosted zone ID — use for Route 53 alias A records"
  value       = aws_lb.main.zone_id
}

output "ecr_repository_url" {
  description = "ECR repository URL — set this as the ECR_REPOSITORY GitHub secret"
  value       = aws_ecr_repository.backend.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name — set as the ECS_CLUSTER GitHub secret"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name — set as the ECS_SERVICE GitHub secret"
  value       = aws_ecs_service.backend.name
}

output "task_definition_family" {
  description = "Task definition family name — set as the ECS_TASK_DEFINITION GitHub secret"
  value       = aws_ecs_task_definition.backend.family
}

output "ci_deploy_role_arn" {
  description = "IAM role ARN for GitHub Actions OIDC — add as AWS_ROLE_ARN GitHub secret"
  value       = aws_iam_role.ci_deploy.arn
}

output "github_oidc_provider_arn" {
  description = "GitHub OIDC provider ARN"
  value       = aws_iam_openid_connect_provider.github.arn
}
