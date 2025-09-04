import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class ConfigLoader:
    """Utility class for loading and managing configuration files."""
    
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dictionary containing configuration
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], save_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            save_path: Path to save YAML file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Configuration dictionary with override values
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    @staticmethod
    def config_to_args(config: Dict[str, Any]) -> argparse.Namespace:
        """
        Convert configuration dictionary to argparse Namespace.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            argparse.Namespace object
        """
        args = argparse.Namespace()
        
        def flatten_dict(d: Dict[str, Any], parent_key: str = ''):
            """Flatten nested dictionary."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_config = flatten_dict(config)
        for key, value in flat_config.items():
            setattr(args, key, value)
            
        return args
    
    @staticmethod
    def load_config(
        config_path: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load configuration with optional overrides.
        
        Args:
            config_path: Path to configuration file
            override_config: Dictionary with override values
            
        Returns:
            Final configuration dictionary
        """
        # Load default config
        default_config_path = Path(__file__).parent.parent / 'configs' / 'default_config.yaml'
        config = ConfigLoader.load_yaml(str(default_config_path))
        
        # Load custom config if provided
        if config_path:
            custom_config = ConfigLoader.load_yaml(config_path)
            config = ConfigLoader.merge_configs(config, custom_config)
        
        # Apply overrides if provided
        if override_config:
            config = ConfigLoader.merge_configs(config, override_config)
        
        return config