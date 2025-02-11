import yaml


def load_config(config_path='configs/config.yaml'):
    """Load the training configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
if __name__ == '__main__':
    pass