"""
Database MCP Tools for Fractal
Supports MySQL, PostgreSQL, and MongoDB connections
"""
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from langsmith import traceable
import asyncpg
import aiomysql
from motor.motor_asyncio import AsyncIOMotorClient

class DatabaseMCP:
    """MCP server for database operations"""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.tools = self._create_tools()
    
    def _create_tools(self):
        """Create all database tools as LangChain tools"""
        tools_list = []
        
        # PostgreSQL Tools
        @tool
        @traceable('tool', name='connect_postgres')
        async def connect_postgres(
            connection_string: str,
            alias: str = "default"
        ) -> str:
            """
            Connect to a PostgreSQL database.
            
            Args:
                connection_string: PostgreSQL connection string (e.g., 'postgresql://user:pass@localhost:5432/dbname')
                alias: Alias name for this connection (default: 'default')
            
            Returns:
                Connection status message
            """
            try:
                conn = await asyncpg.connect(connection_string)
                self.connections[f"postgres_{alias}"] = conn
                return f"✓ Connected to PostgreSQL as '{alias}'"
            except Exception as e:
                return f"Error connecting to PostgreSQL: {str(e)}"
        
        @tool
        @traceable('tool', name='query_postgres')
        async def query_postgres(
            query: str,
            alias: str = "default",
            params: Optional[List[str]] = None
        ) -> str:
            """
            Execute a SELECT query on PostgreSQL.
            
            Args:
                query: SQL SELECT query
                alias: Connection alias to use
                params: Optional query parameters
            
            Returns:
                Query results as formatted string
            """
            conn_key = f"postgres_{alias}"
            if conn_key not in self.connections:
                return f"Error: No PostgreSQL connection found with alias '{alias}'"
            
            try:
                conn = self.connections[conn_key]
                results = await conn.fetch(query, *(params or []))
                
                if not results:
                    return "Query returned no results"
                
                output = []
                for row in results[:10]:
                    output.append(dict(row))
                
                return f"Query returned {len(results)} rows:\n{output}"
            except Exception as e:
                return f"Error executing query: {str(e)}"
        
        @tool
        @traceable('tool', name='execute_postgres')
        async def execute_postgres(
            query: str,
            alias: str = "default",
            params: Optional[List[str]] = None
        ) -> str:
            """
            Execute an INSERT/UPDATE/DELETE query on PostgreSQL.
            
            Args:
                query: SQL query (INSERT/UPDATE/DELETE)
                alias: Connection alias to use
                params: Optional query parameters
            
            Returns:
                Execution status message
            """
            conn_key = f"postgres_{alias}"
            if conn_key not in self.connections:
                return f"Error: No PostgreSQL connection found with alias '{alias}'"
            
            try:
                conn = self.connections[conn_key]
                result = await conn.execute(query, *(params or []))
                return f"✓ Query executed: {result}"
            except Exception as e:
                return f"Error executing query: {str(e)}"
        
        # MySQL Tools
        @tool
        @traceable('tool', name='connect_mysql')
        async def connect_mysql(
            host: str,
            user: str,
            password: str,
            database: str,
            port: int = 3306,
            alias: str = "default"
        ) -> str:
            """
            Connect to a MySQL database.
            
            Args:
                host: MySQL host address
                user: Database user
                password: Database password
                database: Database name
                port: MySQL port (default: 3306)
                alias: Alias name for this connection
            
            Returns:
                Connection status message
            """
            try:
                pool = await aiomysql.create_pool(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    db=database
                )
                self.connections[f"mysql_{alias}"] = pool
                return f"✓ Connected to MySQL as '{alias}'"
            except Exception as e:
                return f"Error connecting to MySQL: {str(e)}"
        
        @tool
        @traceable('tool', name='query_mysql')
        async def query_mysql(
            query: str,
            alias: str = "default"
        ) -> str:
            """
            Execute a SELECT query on MySQL.
            
            Args:
                query: SQL SELECT query
                alias: Connection alias to use
            
            Returns:
                Query results as formatted string
            """
            conn_key = f"mysql_{alias}"
            if conn_key not in self.connections:
                return f"Error: No MySQL connection found with alias '{alias}'"
            
            try:
                pool = self.connections[conn_key]
                async with pool.acquire() as conn:
                    async with conn.cursor(aiomysql.DictCursor) as cursor:
                        await cursor.execute(query)
                        results = await cursor.fetchall()
                        
                        if not results:
                            return "Query returned no results"
                        
                        return f"Query returned {len(results)} rows:\n{results[:50]}"
            except Exception as e:
                return f"Error executing query: {str(e)}"
        
        @tool
        @traceable('tool', name='execute_mysql')
        async def execute_mysql(
            query: str,
            alias: str = "default"
        ) -> str:
            """
            Execute an INSERT/UPDATE/DELETE query on MySQL.
            
            Args:
                query: SQL query (INSERT/UPDATE/DELETE)
                alias: Connection alias to use
            
            Returns:
                Execution status message
            """
            conn_key = f"mysql_{alias}"
            if conn_key not in self.connections:
                return f"Error: No MySQL connection found with alias '{alias}'"
            
            try:
                pool = self.connections[conn_key]
                async with pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute(query)
                        await conn.commit()
                        return f"✓ Query executed: {cursor.rowcount} rows affected"
            except Exception as e:
                return f"Error executing query: {str(e)}"
        
        # MongoDB Tools
        @tool
        @traceable('tool', name='connect_mongodb')
        async def connect_mongodb(
            connection_string: str,
            alias: str = "default"
        ) -> str:
            """
            Connect to a MongoDB database.
            
            Args:
                connection_string: MongoDB connection string (e.g., 'mongodb://user:pass@localhost:27017/')
                alias: Alias name for this connection
            
            Returns:
                Connection status message
            """
            try:
                client = AsyncIOMotorClient(connection_string)
                # Test connection
                await client.admin.command('ping')
                self.connections[f"mongodb_{alias}"] = client
                return f"✓ Connected to MongoDB as '{alias}'"
            except Exception as e:
                return f"Error connecting to MongoDB: {str(e)}"
        
        @tool
        @traceable('tool', name='query_mongodb')
        async def query_mongodb(
            database: str,
            collection: str,
            filter_query: Dict[str, Any],
            alias: str = "default",
            limit: int = 50
        ) -> str:
            """
            Query documents from MongoDB collection.
            
            Args:
                database: Database name
                collection: Collection name
                filter_query: MongoDB filter query (e.g., {"status": "active"})
                alias: Connection alias to use
                limit: Maximum number of documents to return (default: 50)
            
            Returns:
                Query results as formatted string
            """
            conn_key = f"mongodb_{alias}"
            if conn_key not in self.connections:
                return f"Error: No MongoDB connection found with alias '{alias}'"
            
            try:
                client = self.connections[conn_key]
                db = client[database]
                coll = db[collection]
                
                cursor = coll.find(filter_query).limit(limit)
                results = await cursor.to_list(length=limit)
                
                if not results:
                    return "Query returned no results"
                
                return f"Query returned {len(results)} documents:\n{results}"
            except Exception as e:
                return f"Error executing query: {str(e)}"
        
        @tool
        @traceable('tool', name='insert_mongodb')
        async def insert_mongodb(
            database: str,
            collection: str,
            document: Dict[str, Any],
            alias: str = "default"
        ) -> str:
            """
            Insert a document into MongoDB collection.
            
            Args:
                database: Database name
                collection: Collection name
                document: Document to insert
                alias: Connection alias to use
            
            Returns:
                Insert status message with inserted ID
            """
            conn_key = f"mongodb_{alias}"
            if conn_key not in self.connections:
                return f"Error: No MongoDB connection found with alias '{alias}'"
            
            try:
                client = self.connections[conn_key]
                db = client[database]
                coll = db[collection]
                
                result = await coll.insert_one(document)
                return f"✓ Document inserted with ID: {result.inserted_id}"
            except Exception as e:
                return f"Error inserting document: {str(e)}"
        
        @tool
        @traceable('tool', name='update_mongodb')
        async def update_mongodb(
            database: str,
            collection: str,
            filter_query: Dict[str, Any],
            update_data: Dict[str, Any],
            alias: str = "default"
        ) -> str:
            """
            Update documents in MongoDB collection.
            
            Args:
                database: Database name
                collection: Collection name
                filter_query: Filter to match documents
                update_data: Update operations (e.g., {"$set": {"status": "updated"}})
                alias: Connection alias to use
            
            Returns:
                Update status message
            """
            conn_key = f"mongodb_{alias}"
            if conn_key not in self.connections:
                return f"Error: No MongoDB connection found with alias '{alias}'"
            
            try:
                client = self.connections[conn_key]
                db = client[database]
                coll = db[collection]
                
                result = await coll.update_many(filter_query, update_data)
                return f"✓ Updated {result.modified_count} documents"
            except Exception as e:
                return f"Error updating documents: {str(e)}"
        
        @tool
        @traceable('tool', name='delete_mongodb')
        async def delete_mongodb(
            database: str,
            collection: str,
            filter_query: Dict[str, Any],
            alias: str = "default"
        ) -> str:
            """
            Delete documents from MongoDB collection.
            
            Args:
                database: Database name
                collection: Collection name
                filter_query: Filter to match documents to delete
                alias: Connection alias to use
            
            Returns:
                Delete status message
            """
            conn_key = f"mongodb_{alias}"
            if conn_key not in self.connections:
                return f"Error: No MongoDB connection found with alias '{alias}'"
            
            try:
                client = self.connections[conn_key]
                db = client[database]
                coll = db[collection]
                
                result = await coll.delete_many(filter_query)
                return f"✓ Deleted {result.deleted_count} documents"
            except Exception as e:
                return f"Error deleting documents: {str(e)}"
        
        # Connection management
        @tool
        @traceable('tool', name='list_connections')
        async def list_connections() -> str:
            """
            List all active database connections.
            
            Returns:
                List of active connections with their types and aliases
            """
            if not self.connections:
                return "No active database connections"
            
            output = ["Active Database Connections:"]
            for key in self.connections.keys():
                db_type, alias = key.split('_', 1)
                output.append(f"  • {db_type.upper()} - {alias}")
            
            return "\n".join(output)
        
        @tool
        @traceable('tool', name='disconnect_database')
        async def disconnect_database(
            db_type: str,
            alias: str = "default"
        ) -> str:
            """
            Disconnect from a database.
            
            Args:
                db_type: Database type ('postgres', 'mysql', or 'mongodb')
                alias: Connection alias
            
            Returns:
                Disconnection status message
            """
            conn_key = f"{db_type}_{alias}"
            if conn_key not in self.connections:
                return f"No connection found for {db_type} with alias '{alias}'"
            
            try:
                conn = self.connections[conn_key]
                
                if db_type == "postgres":
                    await conn.close()
                elif db_type == "mysql":
                    conn.close()
                    await conn.wait_closed()
                elif db_type == "mongodb":
                    conn.close()
                
                del self.connections[conn_key]
                return f"✓ Disconnected from {db_type} ({alias})"
            except Exception as e:
                return f"Error disconnecting: {str(e)}"
        
        # Add all tools to the list
        tools_list.extend([
            connect_postgres,
            query_postgres,
            execute_postgres,
            connect_mysql,
            query_mysql,
            execute_mysql,
            connect_mongodb,
            query_mongodb,
            insert_mongodb,
            update_mongodb,
            delete_mongodb,
            list_connections,
            disconnect_database
        ])
        
        return tools_list
    
    def get_tools_sync(self):
        """Get all registered tools synchronously"""
        return self.tools
    
    async def cleanup(self):
        """Close all database connections"""
        for key, conn in self.connections.items():
            try:
                db_type = key.split('_')[0]
                if db_type == "postgres":
                    await conn.close()
                elif db_type == "mysql":
                    conn.close()
                    await conn.wait_closed()
                elif db_type == "mongodb":
                    conn.close()
            except Exception as e:
                print(f"Error closing connection {key}: {e}")
        
        self.connections.clear()

# Global instance
_db_mcp_instance: Optional[DatabaseMCP] = None

def get_db_mcp() -> DatabaseMCP:
    """Get or create the global DatabaseMCP instance"""
    global _db_mcp_instance
    if _db_mcp_instance is None:
        _db_mcp_instance = DatabaseMCP()
    return _db_mcp_instance