import os

PROJECT_NAME = "RetailShield_AI_Platform"

STRUCTURE = {
    "config": {
        "deployment": ["docker-compose.yml"],
        "environment": [".env.example"],
        "application": [
            "camera_config.yaml",
            "alert_rules.yaml",
            "store_zones.yaml",
            "privacy_settings.yaml",
        ],
        "hardware": ["gpu_config.yaml"],
    },

    "src": {
        "core": {
            "engine": ["pipeline.py", "orchestrator.py"],
            "detection": ["person_detector.py", "object_detector.py"],
            "tracking": ["tracker.py"],
            "pose": ["pose_estimator.py", "gesture_recognizer.py"],
            "behavior": ["temporal_analyzer.py", "pattern_detector.py"],
            "models": ["base_model.py", "model_factory.py"],
        },

        "services": {
            "video": ["stream_reader.py"],
            "alert": ["alert_manager.py"],
            "storage": ["incident_store.py"],
            "integration": ["api_client.py"],
            "monitoring": ["health_check.py"],
        },

        "api": {
            "endpoints": ["incidents.py", "alerts.py", "cameras.py"],
            "schemas": ["incident_schema.py"],
            "middleware": ["auth.py"],
            "websocket": ["manager.py"],
        },

        "dashboard": {
            "frontend": ["README.md"],
            "backend": ["app.py"],
        },

        "database": {
            "models": ["incident.py", "camera.py"],
            "repositories": ["base_repository.py"],
            "migrations": [],
        },

        "utils": {
            "logging": ["logger.py"],
            "video": ["frame_utils.py"],
            "geometry": ["zone_utils.py"],
        },

        "main.py": None,
    },

    "models": {
        "pretrained": ["README.md"],
        "custom": ["train_model.py"],
    },

    "data": {
        "incidents": [],
        "logs": [],
        "exports": [],
    },

    "scripts": {
        "installation": ["install.sh"],
        "deployment": ["deploy.sh"],
        "maintenance": ["cleanup.py"],
    },

    "tests": {
        "unit": [],
        "integration": [],
    },

    "docs": {
        "user": ["user_manual.md"],
        "admin": ["deployment_guide.md"],
        "developer": ["architecture.md"],
    },
}

def create_structure(base_path, tree):
    for name, content in tree.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        elif isinstance(content, list):
            os.makedirs(path, exist_ok=True)
            for file in content:
                file_path = os.path.join(path, file)
                open(file_path, "a").close()
        elif content is None:
            open(path, "a").close()

def main():
    os.makedirs(PROJECT_NAME, exist_ok=True)
    create_structure(PROJECT_NAME, STRUCTURE)
    print(f"✅ Project structure '{PROJECT_NAME}' created successfully.")

if __name__ == "__main__":
    main()
