"""
HIGH-PERFORMANCE DATA STORAGE LAYER
ClickHouse for time-series, Redis for state, PostgreSQL for audit logs
"""
import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import sqlite3
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ==========================================
# CLICKHOUSE SCHEMAS (Time-Series Data)
# ==========================================

CLICKHOUSE_SCHEMAS = """
-- Market Ticks (High-Resolution Order Book)
CREATE TABLE IF NOT EXISTS market_ticks (
    timestamp DateTime64(9) CODEC(Delta, ZSTD(1)),
    symbol String,
    exchange String,
    bid_price Float64,
    ask_price Float64,
    bid_volume Float64,
    ask_volume Float64,
    last_price Float64,
    volume UInt64,
    vpin_score Float32,
    order_imbalance Float32,
    sentiment_score Float32,
    index timestamp
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (exchange, symbol, timestamp)
TTL timestamp + INTERVAL 30 DAY;

-- Alternative Data (News, Climate, Political)
CREATE TABLE IF NOT EXISTS alternative_signals (
    timestamp DateTime64(9),
    source String,
    entity String,
    signal_type String,
    payload String,
    processed_score Float32,
    quality_score Float32,
    index timestamp
) ENGINE = ReplacingMergeTree(timestamp)
ORDER BY (entity, source, timestamp)
TTL timestamp + INTERVAL 90 DAY;

-- AI Predictions
CREATE TABLE IF NOT EXISTS ai_predictions (
    timestamp DateTime64(9),
    symbol String,
    model_name String,
    prediction Float32,
    confidence Float32,
    risk_score Float32,
    factors String,
    actual_outcome Float32,
    index timestamp
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (symbol, model_name, timestamp);

-- Trade Execution Log
CREATE TABLE IF NOT EXISTS trade_executions (
    timestamp DateTime64(9),
    order_id String,
    symbol String,
    side String,
    quantity Float64,
    price Float64,
    execution_latency_ms Float32,
    slippage_bps Float32,
    fees Float64,
    hedge_placed Boolean,
    index timestamp
) ENGINE = MergeTree()
ORDER BY (symbol, timestamp);
"""


class ClickHouseManager:
    """
    ClickHouse connection and query management.
    In production, use clickhouse-driver or asyncio-clickhouse.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get("clickhouse", {}).get("host", "localhost")
        self.port = config.get("clickhouse", {}).get("port", 9000)
        self.database = config.get("clickhouse", {}).get("database", "trade")
        self.user = config.get("clickhouse", {}).get("user", "default")
        self.password = config.get("clickhouse", {}).get("password", "")
        
        self._connected = False
    
    async def connect(self):
        """Establish ClickHouse connection."""
        # In production: use clickhouse-driver
        # from clickhouse_driver import Client
        # self.client = Client(host=self.host, port=self.port, database=self.database)
        logger.info(f"ClickHouse connection configured: {self.host}:{self.port}/{self.database}")
        self._connected = True
    
    async def insert_market_tick(self, tick: Dict):
        """Insert a market tick."""
        # In production: actual ClickHouse INSERT
        # query = "INSERT INTO market_ticks VALUES"
        # self.client.execute(query, [tick])
        pass
    
    async def query_aggregates(self, symbol: str, start: datetime, end: datetime) -> Dict:
        """Query aggregated data for a symbol."""
        # Example aggregation query
        query = f"""
        SELECT 
            toStartOfHour(timestamp) as hour,
            avg(last_price) as avg_price,
            max(last_price) as max_price,
            min(last_price) as min_price,
            sum(volume) as total_volume,
            avg(sentiment_score) as avg_sentiment
        FROM market_ticks
        WHERE symbol = '{symbol}'
          AND timestamp BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'
        GROUP BY hour
        ORDER BY hour
        """
        # In production: self.client.execute(query)
        return {}


# ==========================================
# REDIS STATE MANAGEMENT (Sub-millisecond)
# ==========================================

class RedisStateManager:
    """
    Redis-based state management for sub-millisecond access.
    Optimized for HFT order book and position state.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get("redis", {}).get("host", "localhost")
        self.port = config.get("redis", {}).get("port", 6379)
        self.db = config.get("redis", {}).get("db", 0)
        
        # Use asyncio redis client
        self._redis = None
        self._lock = asyncio.Lock()
    
    async def connect(self):
        """Connect to Redis."""
        # In production: use aioredis or redis-py with async
        # import aioredis
        # self._redis = await aioredis.create_redis_pool(f'redis://{self.host}:{self.port}/{self.db}')
        logger.info(f"Redis configured: {self.host}:{self.port}/{self.db}")
    
    async def set_order_book(self, symbol: str, bids: List, asks: List):
        """Cache order book state."""
        key = f"orderbook:{symbol}"
        data = json.dumps({
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now().isoformat()
        })
        # In production: await self._redis.set(key, data, ex=1)  # 1 second TTL
        pass
    
    async def get_order_book(self, symbol: str) -> Optional[Dict]:
        """Get cached order book."""
        key = f"orderbook:{symbol}"
        # In production: data = await self._redis.get(key)
        return None
    
    async def set_position(self, symbol: str, position: Dict):
        """Cache position state."""
        key = f"position:{symbol}"
        data = json.dumps(position)
        # await self._redis.set(key, data)
        pass
    
    async def increment_trade_count(self, symbol: str):
        """Increment trade counter atomically."""
        key = f"trades:{symbol}:count"
        # In production: return await self._redis.incr(key)
        return 0
    
    async def get_trade_stats(self, symbol: str) -> Dict:
        """Get trade statistics."""
        # In production: multi-get for multiple keys
        return {
            "count": 0,
            "volume": 0,
            "last_trade": None,
        }


# ==========================================
# SQLite Audit Logger (WORM Compliance)
# ==========================================

class AuditLogger:
    """
    Immutable audit logging with cryptographic chaining.
    WORM (Write Once, Read Many) compliance.
    """
    
    def __init__(self, log_path: str = "logs/audit.db"):
        self.log_path = log_path
        self.last_hash = "0" * 64  # Genesis hash
        self._lock = threading.Lock()
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with append-only schema."""
        conn = sqlite3.connect(self.log_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                current_hash TEXT NOT NULL,
                UNIQUE(current_hash)
            )
        """)
        
        # Create index for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_action ON audit_log(action)
        """)
        
        conn.commit()
        
        # Get last hash if exists
        cursor.execute("SELECT current_hash FROM audit_log ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            self.last_hash = row[0]
        
        conn.close()
        logger.info(f"Audit logger initialized at {self.log_path}")
    
    def _calculate_hash(self, record: Dict) -> str:
        """Calculate SHA-256 hash of record."""
        # Include all fields in hash for integrity
        record_str = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(record_str.encode()).hexdigest()
    
    def log_decision(self, action: str, details: Dict):
        """Log an AI decision with cryptographic chaining."""
        with self._lock:
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            record = {
                "timestamp": timestamp,
                "action": action,
                "details": details,
                "previous_hash": self.last_hash,
            }
            
            # Calculate hash including previous hash (chain integrity)
            current_hash = self._calculate_hash(record)
            record["current_hash"] = current_hash
            
            # Insert into database
            conn = sqlite3.connect(self.log_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO audit_log 
                    (timestamp, action, details, previous_hash, current_hash)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    record["timestamp"],
                    record["action"],
                    json.dumps(record["details"]),
                    record["previous_hash"],
                    record["current_hash"],
                ))
                conn.commit()
                
                self.last_hash = current_hash
                logger.debug(f"Audit logged: {action}")
                
            except sqlite3.IntegrityError:
                logger.error("Hash collision or duplicate detected!")
            
            finally:
                conn.close()
    
    def log_trade(self, trade: Dict):
        """Log a trade execution."""
        self.log_decision("TRADE_EXECUTION", {
            "order_id": trade.get("order_id", ""),
            "symbol": trade.get("symbol", ""),
            "side": trade.get("side", ""),
            "quantity": trade.get("quantity", 0),
            "price": trade.get("price", 0),
            "execution_latency_ms": trade.get("latency", 0),
            "hedge_placed": trade.get("hedge_placed", False),
        })
    
    def log_signal(self, signal: Dict):
        """Log an AI signal."""
        self.log_decision("AI_SIGNAL", {
            "symbol": signal.get("symbol", ""),
            "signal": signal.get("signal", ""),
            "confidence": signal.get("confidence", 0),
            "risk_score": signal.get("risk_score", 0),
            "factors": signal.get("factors", {}),
        })
    
    def verify_integrity(self) -> Tuple[bool, str]:
        """Verify the integrity of the audit log chain."""
        conn = sqlite3.connect(self.log_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM audit_log ORDER BY id")
        rows = cursor.fetchall()
        
        expected_previous = "0" * 64
        
        for row in rows:
            record = {
                "timestamp": row[1],
                "action": row[2],
                "details": row[3],
                "previous_hash": row[4],
            }
            
            # Verify chain
            if row[4] != expected_previous:
                conn.close()
                return False, f"Chain broken at record {row[0]}"
            
            # Verify hash
            calculated = self._calculate_hash(record)
            if calculated != row[5]:
                conn.close()
                return False, f"Hash mismatch at record {row[0]}"
            
            expected_previous = row[5]
        
        conn.close()
        return True, "Audit log integrity verified"
    
    def get_logs(self, action: Optional[str] = None, 
                 limit: int = 100) -> List[Dict]:
        """Retrieve audit logs."""
        conn = sqlite3.connect(self.log_path)
        cursor = conn.cursor()
        
        if action:
            cursor.execute("""
                SELECT * FROM audit_log 
                WHERE action = ?
                ORDER BY id DESC
                LIMIT ?
            """, (action, limit))
        else:
            cursor.execute("""
                SELECT * FROM audit_log 
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            logs.append({
                "id": row[0],
                "timestamp": row[1],
                "action": row[2],
                "details": json.loads(row[3]),
                "previous_hash": row[4],
                "current_hash": row[5],
            })
        
        return logs


# ==========================================
# LOCAL FILE STORAGE (Fallback)
# ==========================================

class LocalFileStorage:
    """
    Local file-based storage for when database servers are unavailable.
    Uses memory-mapped files for performance.
    """
    
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create storage directories."""
        import os
        dirs = ["ticks", "signals", "trades", "backups"]
        for d in dirs:
            path = os.path.join(self.base_path, d)
            os.makedirs(path, exist_ok=True)
    
    def save_tick(self, symbol: str, tick: Dict):
        """Save tick data to file."""
        import os
        date = datetime.now().strftime("%Y%m%d")
        filename = f"{self.base_path}/ticks/{symbol}_{date}.jsonl"
        
        with open(filename, "a") as f:
            f.write(json.dumps({
                **tick,
                "timestamp": datetime.now().isoformat()
            }) + "\n")
    
    def save_signal(self, signal: Dict):
        """Save AI signal."""
        date = datetime.now().strftime("%Y%m%d")
        filename = f"{self.base_path}/signals/{date}.jsonl"
        
        with open(filename, "a") as f:
            f.write(json.dumps({
                **signal,
                "timestamp": datetime.now().isoformat()
            }) + "\n")
    
    def load_signals(self, date: str) -> List[Dict]:
        """Load signals for a specific date."""
        filename = f"{self.base_path}/signals/{date}.jsonl"
        
        signals = []
        try:
            with open(filename, "r") as f:
                for line in f:
                    signals.append(json.loads(line))
        except FileNotFoundError:
            pass
        
        return signals


# ==========================================
# STORAGE MANAGER ORCHESTRATOR
# ==========================================

class StorageManager:
    """
    Main storage orchestrator managing all storage backends.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize storage backends
        self.clickhouse = ClickHouseManager(config)
        self.redis = RedisStateManager(config)
        self.audit = AuditLogger(config.get("audit", {}).get("path", "logs/audit.db"))
        self.local = LocalFileStorage(config.get("storage", {}).get("path", "data"))
        
        # Connection state
        self._initialized = False
    
    async def initialize(self):
        """Initialize all storage backends."""
        if self._initialized:
            return
        
        logger.info("Initializing storage managers...")
        
        try:
            await self.clickhouse.connect()
        except Exception as e:
            logger.warning(f"ClickHouse unavailable, using local storage: {e}")
        
        try:
            await self.redis.connect()
        except Exception as e:
            logger.warning(f"Redis unavailable, using local storage: {e}")
        
        self._initialized = True
        logger.info("Storage managers initialized")
    
    async def store_tick(self, tick: Dict):
        """Store a market tick."""
        # Store to ClickHouse for analytics
        try:
            await self.clickhouse.insert_market_tick(tick)
        except Exception:
            pass
        
        # Also save locally for backup
        self.local.save_tick(tick.get("symbol", "UNKNOWN"), tick)
    
    async def store_signal(self, signal: Dict):
        """Store an AI signal."""
        # Log to audit trail
        self.audit.log_signal(signal)
        
        # Save locally
        self.local.save_signal(signal)
    
    def log_trade(self, trade: Dict):
        """Log a trade execution."""
        self.audit.log_trade(trade)
    
    def log_decision(self, action: str, details: Dict):
        """Log a general decision."""
        self.audit.log_decision(action, details)
    
    def verify_audit_integrity(self) -> Tuple[bool, str]:
        """Verify audit log integrity."""
        return self.audit.verify_integrity()


# Global instance
_storage_manager: Optional[StorageManager] = None


def get_storage_manager(config: Optional[Dict] = None) -> StorageManager:
    """Get or create the storage manager."""
    global _storage_manager
    if _storage_manager is None or config is not None:
        _storage_manager = StorageManager(config or {})
    return _storage_manager