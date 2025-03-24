terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}

resource "kubernetes_namespace" "mlops" {
  metadata {
    name = "mlops"
  }
}

resource "kubernetes_deployment" "mlflow" {
  metadata {
    name      = "mlflow"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "mlflow"
      }
    }

    template {
      metadata {
        labels = {
          app = "mlflow"
        }
      }

      spec {
        container {
          image = "mlflow/mlflow"
          name  = "mlflow"
          port {
            container_port = 5000
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "mlflow" {
  metadata {
    name      = "mlflow-service"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }

  spec {
    selector = {
      app = "mlflow"
    }

    port {
      protocol    = "TCP"
      port        = 5000
      target_port = 5000
    }

    type = "NodePort"
  }
}
