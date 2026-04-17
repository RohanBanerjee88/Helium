variable "project_name" {
  description = "Project name — used as a prefix for all resource names"
  type        = string
  default     = "helium"
}

variable "environment" {
  description = "Deployment environment (prod, staging, dev)"
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of AZs to use (minimum 2 required by ALB)"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "container_port" {
  description = "Port the backend container listens on"
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "Fargate task CPU units (256 | 512 | 1024 | 2048 | 4096)"
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Number of running task replicas"
  type        = number
  default     = 1
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS. Leave empty to use HTTP only."
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
}

variable "helium_data_dir" {
  description = "Directory inside the container where job data is stored"
  type        = string
  default     = "/tmp/data"
}

variable "helium_min_images" {
  description = "Minimum number of images required per upload job"
  type        = number
  default     = 8
}

variable "helium_max_images" {
  description = "Maximum number of images allowed per upload job"
  type        = number
  default     = 20
}

variable "helium_max_image_size_mb" {
  description = "Maximum size in MB for a single uploaded image"
  type        = number
  default     = 20
}
