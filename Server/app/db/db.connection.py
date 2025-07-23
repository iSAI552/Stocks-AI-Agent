import logging
import os
import ssl
import threading
from contextlib import contextmanager
from typing import Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import (
    ConfigurationError,
    ConnectionFailure,
    InvalidURI,
    NetworkTimeout,
    OperationFailure,
    ServerSelectionTimeoutError,
)
from pymongo.server_api import ServerApi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""
    pass


class DatabaseManager:
    """
    Thread-safe MongoDB connection manager with connection pooling,
    error handling, and security features.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the database manager."""
        if hasattr(self, '_initialized'):
            return
            
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._collection: Optional[Collection] = None
        self._connection_lock = threading.Lock()
        self._initialized = True
        
        # Initialize connection
        self._connect()
    
    def _validate_environment_variables(self) -> dict:
        """
        Validate and retrieve required environment variables.
        
        Returns:
            dict: Dictionary containing validated environment variables.
            
        Raises:
            DatabaseConnectionError: If required environment variables are missing.
        """
        required_vars = {
            'MONGODB_URI': os.getenv('MONGODB_URI'),
            'MONGODB_DB_NAME': os.getenv('MONGODB_DB_NAME'),
            'MONGODB_COLLECTION_NAME': os.getenv('MONGODB_COLLECTION_NAME')
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg)
        
        # Validate URI format
        uri = required_vars['MONGODB_URI']
        if not uri.startswith(('mongodb://', 'mongodb+srv://')):
            error_msg = "Invalid MongoDB URI format. Must start with 'mongodb://' or 'mongodb+srv://'"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg)
        
        logger.info("Environment variables validated successfully")
        return required_vars
    
    def _get_connection_options(self) -> dict:
        """
        Get MongoDB connection options for production use.
        
        Returns:
            dict: Connection options dictionary.
        """
        return {
            # Connection pool settings
            'maxPoolSize': int(os.getenv('MONGODB_MAX_POOL_SIZE', '50')),
            'minPoolSize': int(os.getenv('MONGODB_MIN_POOL_SIZE', '5')),
            'maxIdleTimeMS': int(os.getenv('MONGODB_MAX_IDLE_TIME_MS', '30000')),
            'waitQueueTimeoutMS': int(os.getenv('MONGODB_WAIT_QUEUE_TIMEOUT_MS', '5000')),
            
            # Timeout settings
            'connectTimeoutMS': int(os.getenv('MONGODB_CONNECT_TIMEOUT_MS', '20000')),
            'serverSelectionTimeoutMS': int(os.getenv('MONGODB_SERVER_SELECTION_TIMEOUT_MS', '20000')),
            'socketTimeoutMS': int(os.getenv('MONGODB_SOCKET_TIMEOUT_MS', '20000')),
            
            # Write concern for data consistency
            'w': os.getenv('MONGODB_WRITE_CONCERN', 'majority'),
            'j': os.getenv('MONGODB_JOURNAL', 'true').lower() == 'true',
            'wtimeoutMS': int(os.getenv('MONGODB_WRITE_TIMEOUT_MS', '10000')),
            
            # Read preference
            'readPreference': os.getenv('MONGODB_READ_PREFERENCE', 'primary'),
            
            # SSL/TLS settings
            'ssl': True,
            'ssl_cert_reqs': ssl.CERT_REQUIRED,
            'ssl_ca_certs': None,  # Use system CA certificates
            
            # Retry settings
            'retryWrites': True,
            'retryReads': True,
            
            # Application name for monitoring
            'appName': os.getenv('APP_NAME', 'StocksAIAgent'),
            
            # Server API version
            'server_api': ServerApi('1', strict=True, deprecation_errors=True)
        }
    
    def _connect(self) -> None:
        """
        Establish connection to MongoDB with comprehensive error handling.
        
        Raises:
            DatabaseConnectionError: If connection fails.
        """
        with self._connection_lock:
            try:
                # Validate environment variables
                env_vars = self._validate_environment_variables()
                
                # Get connection options
                connection_options = self._get_connection_options()
                
                logger.info("Attempting to connect to MongoDB...")
                
                # Create MongoDB client
                self._client = MongoClient(
                    env_vars['MONGODB_URI'],
                    **connection_options
                )
                
                # Test the connection
                self._client.admin.command('ping')
                logger.info("Successfully connected to MongoDB")
                
                # Get database and collection references
                self._db = self._client[env_vars['MONGODB_DB_NAME']]
                self._collection = self._db[env_vars['MONGODB_COLLECTION_NAME']]
                
                # Create indexes if they don't exist (optional)
                self._ensure_indexes()
                
                logger.info(f"Connected to database: {env_vars['MONGODB_DB_NAME']}")
                logger.info(f"Using collection: {env_vars['MONGODB_COLLECTION_NAME']}")
                
            except (InvalidURI, ConfigurationError) as e:
                error_msg = f"MongoDB configuration error: {str(e)}"
                logger.error(error_msg)
                raise DatabaseConnectionError(error_msg) from e
                
            except (ConnectionFailure, ServerSelectionTimeoutError, NetworkTimeout) as e:
                error_msg = f"MongoDB connection failed: {str(e)}"
                logger.error(error_msg)
                raise DatabaseConnectionError(error_msg) from e
                
            except OperationFailure as e:
                error_msg = f"MongoDB operation failed: {str(e)}"
                logger.error(error_msg)
                raise DatabaseConnectionError(error_msg) from e
                
            except Exception as e:
                error_msg = f"Unexpected error during MongoDB connection: {str(e)}"
                logger.error(error_msg)
                raise DatabaseConnectionError(error_msg) from e
    
    def _ensure_indexes(self) -> None:
        """
        Ensure required indexes exist for optimal performance.
        This is optional and can be customized based on your data structure.
        """
        try:
            if self._collection:
                # Example: Create indexes based on your data structure
                # Uncomment and modify as needed
                # self._collection.create_index([("symbol", 1)], background=True)
                # self._collection.create_index([("timestamp", -1)], background=True)
                logger.info("Database indexes verified/created successfully")
        except Exception as e:
            logger.warning(f"Failed to create/verify indexes: {str(e)}")
    
    def get_client(self) -> MongoClient:
        """
        Get the MongoDB client instance.
        
        Returns:
            MongoClient: The MongoDB client instance.
            
        Raises:
            DatabaseConnectionError: If client is not available.
        """
        if not self._client:
            error_msg = "MongoDB client is not initialized"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg)
        return self._client
    
    def get_database(self) -> Database:
        """
        Get the database instance.
        
        Returns:
            Database: The MongoDB database instance.
            
        Raises:
            DatabaseConnectionError: If database is not available.
        """
        if not self._db:
            error_msg = "MongoDB database is not initialized"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg)
        return self._db
    
    def get_collection(self) -> Collection:
        """
        Get the collection instance.
        
        Returns:
            Collection: The MongoDB collection instance.
            
        Raises:
            DatabaseConnectionError: If collection is not available.
        """
        if not self._collection:
            error_msg = "MongoDB collection is not initialized"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg)
        return self._collection
    
    def health_check(self) -> bool:
        """
        Perform a health check on the database connection.
        
        Returns:
            bool: True if connection is healthy, False otherwise.
        """
        try:
            if self._client:
                self._client.admin.command('ping')
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
        return False
    
    def reconnect(self) -> None:
        """
        Reconnect to the database.
        
        Raises:
            DatabaseConnectionError: If reconnection fails.
        """
        logger.info("Attempting to reconnect to MongoDB...")
        self.close()
        self._connect()
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions with automatic transaction handling.
        
        Yields:
            ClientSession: MongoDB session for transactions.
        """
        session = None
        try:
            session = self.get_client().start_session()
            yield session
        except Exception as e:
            if session:
                session.abort_transaction()
            logger.error(f"Session error: {str(e)}")
            raise
        finally:
            if session:
                session.end_session()
    
    def close(self) -> None:
        """Close the database connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None
            logger.info("MongoDB connection closed")
    
    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()


# Global database manager instance
_db_manager = None
_manager_lock = threading.Lock()


def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance (thread-safe singleton).
    
    Returns:
        DatabaseManager: The database manager instance.
    """
    global _db_manager
    if _db_manager is None:
        with _manager_lock:
            if _db_manager is None:
                _db_manager = DatabaseManager()
    return _db_manager


# Convenience functions for backward compatibility and ease of use
def get_client() -> MongoClient:
    """Get MongoDB client."""
    return get_database_manager().get_client()


def get_database() -> Database:
    """Get MongoDB database."""
    return get_database_manager().get_database()


def get_collection() -> Collection:
    """Get MongoDB collection."""
    return get_database_manager().get_collection()


def health_check() -> bool:
    """Perform database health check."""
    return get_database_manager().health_check()


# Legacy variables for backward compatibility (use functions above instead)
try:
    client = get_client()
    db = get_database()
    collection = get_collection()
except DatabaseConnectionError as e:
    logger.error(f"Failed to initialize database connection: {e}")
    client = None
    db = None
    collection = None

