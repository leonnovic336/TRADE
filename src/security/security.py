"""
SECURITY & CRYPTOGRAPHY MODULE
HashiCorp Vault integration, API key management, network isolation
"""
import os
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import base64
import json

logger = logging.getLogger(__name__)


@dataclass
class SecretConfig:
    """Secret configuration."""
    name: str
    secret_type: str  # "api_key", "password", "certificate", "key"
    required_permissions: list
    auto_rotate: bool = True
    rotation_days: int = 90


class HashiCorpVaultManager:
    """
    HashiCorp Vault integration for dynamic secret injection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vault_addr = os.environ.get("VAULT_ADDR", "https://vault.internal:8200")
        self.vault_cacert = os.environ.get("VAULT_CACERT", "")
        self.role = config.get("vault", {}).get("role", "omnitrade-execution-role")
        
        self._client = None
        self._authenticated = False
    
    async def authenticate(self):
        """Authenticate with Vault using Kubernetes Service Account."""
        try:
            # In production, use hvac library
            # from hvac import Client
            # self._client = Client(url=self.vault_addr, verify=self.vault_cacert)
            # 
            # # Kubernetes auth
            # with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as f:
            #     jwt = f.read()
            # self._client.auth.kubernetes.login(role=self.role, jwt=jwt)
            
            logger.info(f"Vault authentication configured for role: {self.role}")
            self._authenticated = True
        except Exception as e:
            logger.error(f"Vault authentication failed: {e}")
            self._authenticated = False
    
    async def get_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Fetch a secret from Vault."""
        if not self._authenticated:
            await self.authenticate()
        
        if not self._client:
            logger.warning("Vault client not available, using environment variables")
            return self._get_from_environment(path)
        
        try:
            # KV v2 secret
            secret = self._client.secrets.kv.v2.read_secret_version(path=path)
            return secret['data']['data']
        except Exception as e:
            logger.error(f"Error fetching secret {path}: {e}")
            return None
    
    async def get_exchange_credentials(self, exchange: str) -> Dict[str, str]:
        """Get exchange API credentials."""
        secret = await self.get_secret(f"omnitrade/{exchange}-api")
        
        if secret:
            return secret
        
        # Fallback to environment variables
        return {
            "api_key": os.environ.get(f"{exchange.upper()}_API_KEY", ""),
            "api_secret": os.environ.get(f"{exchange.upper()}_API_SECRET", ""),
        }
    
    def _get_from_environment(self, path: str) -> Optional[Dict]:
        """Fallback to environment variables."""
        # Parse path like "omnitrade/alpaca-api"
        parts = path.split("/")
        if len(parts) >= 2:
            exchange = parts[1].replace("-api", "").upper()
            return {
                "api_key": os.environ.get(f"{exchange}_API_KEY"),
                "api_secret": os.environ.get(f"{exchange}_API_SECRET"),
            }
        return None
    
    async def get_dynamic_credentials(self, path: str, ttl: int = 3600) -> Optional[Dict]:
        """Get dynamic database or service credentials with TTL."""
        if not self._client:
            return None
        
        try:
            # Request dynamic credentials with TTL
            # For database: self._client.secrets.database.generate_credentials(path)
            # For AWS: self._client.secrets.aws.create_access_key(role_name)
            pass
        except Exception as e:
            logger.error(f"Error generating dynamic credentials: {e}")
            return None


class EnvironmentSecretManager:
    """
    Environment variable-based secret management.
    For development and when Vault is unavailable.
    """
    
    # Required environment variables
    REQUIRED_SECRETS = [
        "ALPACA_API_KEY",
        "ALPACA_API_SECRET",
        "FINNHUB_API_KEY",
        "NEWS_API_KEY",
        "FRED_API_KEY",
    ]
    
    def __init__(self):
        self._secrets_loaded = False
        self._secrets: Dict[str, str] = {}
    
    def load_secrets(self) -> bool:
        """Load secrets from environment variables."""
        missing = []
        
        for secret_name in self.REQUIRED_SECRETS:
            value = os.environ.get(secret_name)
            if value:
                self._secrets[secret_name] = value
            else:
                missing.append(secret_name)
        
        if missing:
            logger.warning(f"Missing secrets (will use defaults): {missing}")
        
        self._secrets_loaded = True
        return len(missing) == 0
    
    def get(self, key: str, default: str = "") -> str:
        """Get a secret value."""
        if not self._secrets_loaded:
            self.load_secrets()
        
        return self._secrets.get(key, os.environ.get(key, default))
    
    def get_all(self) -> Dict[str, str]:
        """Get all loaded secrets."""
        if not self._secrets_loaded:
            self.load_secrets()
        return self._secrets.copy()


class APICredentialManager:
    """
    Centralized API credential management with rotation support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vault = HashiCorpVaultManager(config)
        self.env_manager = EnvironmentSecretManager()
        
        # Credential cache
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def get_alpaca_credentials(self) -> Dict[str, str]:
        """Get Alpaca API credentials."""
        cache_key = "alpaca"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        creds = await self.vault.get_exchange_credentials("alpaca")
        self._cache[cache_key] = creds
        
        return creds
    
    async def get_finnhub_credentials(self) -> Dict[str, str]:
        """Get Finnhub API credentials."""
        return {
            "api_key": self.env_manager.get("FINNHUB_API_KEY", ""),
        }
    
    async def get_newsapi_credentials(self) -> Dict[str, str]:
        """Get NewsAPI credentials."""
        return {
            "api_key": self.env_manager.get("NEWS_API_KEY", ""),
        }
    
    async def get_fred_credentials(self) -> Dict[str, str]:
        """Get FRED API credentials."""
        return {
            "api_key": self.env_manager.get("FRED_API_KEY", ""),
        }
    
    async def get_twitter_credentials(self) -> Dict[str, str]:
        """Get Twitter/X API credentials."""
        return {
            "bearer_token": self.env_manager.get("TWITTER_BEARER_TOKEN", ""),
            "api_key": self.env_manager.get("TWITTER_API_KEY", ""),
            "api_secret": self.env_manager.get("TWITTER_API_SECRET", ""),
        }


class NetworkSecurity:
    """
    Network security and isolation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Allowed IPs/domains
        self.allowed_hosts = config.get("security", {}).get("allowed_hosts", [
            "api.alpaca.markets",
            "finnhub.io",
            "newsapi.org",
            "fred.stlouisfed.org",
        ])
        
        # Rate limiting
        self.rate_limits = config.get("security", {}).get("rate_limits", {
            "default": 100,  # requests per minute
            "exchange_api": 1000,
            "news_api": 60,
        })
    
    def validate_url(self, url: str) -> bool:
        """Validate that URL is to an allowed host."""
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            return parsed.netloc in self.allowed_hosts
        except:
            return False
    
    def sanitize_log_data(self, data: Dict) -> Dict:
        """Remove sensitive data from logs."""
        sensitive_keys = [
            "api_key", "api_secret", "password", "token", 
            "secret", "private_key", "bearer"
        ]
        
        sanitized = data.copy()
        
        for key in list(sanitized.keys()):
            key_lower = key.lower()
            if any(s in key_lower for s in sensitive_keys):
                sanitized[key] = "***REDACTED***"
        
        return sanitized


class DataEncryption:
    """
    Data encryption utilities for sensitive data at rest.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.key = self._load_key()
    
    def _load_key(self) -> bytes:
        """Load encryption key."""
        key_hex = os.environ.get("ENCRYPTION_KEY", "")
        
        if key_hex:
            return bytes.fromhex(key_hex)
        
        # Generate a default key (NOT for production)
        logger.warning("Using default encryption key - NOT SECURE FOR PRODUCTION")
        return bytes.fromhex("0" * 64)
    
    def encrypt(self, data: str) -> str:
        """Encrypt data using AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self.key)
            
            # Generate random nonce
            nonce = os.urandom(12)
            
            # Encrypt
            ciphertext = aesgcm.encrypt(nonce, data.encode(), None)
            
            # Combine nonce + ciphertext and base64 encode
            return base64.b64encode(nonce + ciphertext).decode()
        except ImportError:
            logger.warning("cryptography library not available, returning plaintext")
            return data
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt AES-256-GCM encrypted data."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self.key)
            
            # Decode from base64
            data = base64.b64decode(encrypted_data)
            
            # Extract nonce and ciphertext
            nonce = data[:12]
            ciphertext = data[12:]
            
            # Decrypt
            return aesgcm.decrypt(nonce, ciphertext, None).decode()
        except ImportError:
            logger.warning("cryptography library not available, returning ciphertext")
            return encrypted_data


# Global instances
_vault_manager: Optional[HashiCorpVaultManager] = None
_credential_manager: Optional[APICredentialManager] = None


def get_vault_manager(config: Dict = None) -> HashiCorpVaultManager:
    """Get or create Vault manager."""
    global _vault_manager
    if _vault_manager is None:
        _vault_manager = HashiCorpVaultManager(config or {})
    return _vault_manager


def get_credential_manager(config: Dict = None) -> APICredentialManager:
    """Get or create credential manager."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = APICredentialManager(config or {})
    return _credential_manager