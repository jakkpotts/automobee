import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: str) -> bool:
    """Run a shell command and return True if successful"""
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.cmd}")
        logger.error(f"Error output: {e.stderr}")
        return False

def setup_database():
    """Set up the AutomoBee database"""
    DB_NAME = "automobee"
    
    try:
        # Check if database exists
        check_db = f"psql -lqt | cut -d \\| -f 1 | grep -w {DB_NAME}"
        result = subprocess.run(check_db, shell=True, capture_output=True, text=True)
        
        if DB_NAME not in result.stdout:
            logger.info(f"Creating database {DB_NAME}...")
            if not run_command(f"createdb {DB_NAME}"):
                logger.error("Failed to create database")
                return False
        else:
            logger.info(f"Database {DB_NAME} already exists")
        
        # Apply initial schema
        logger.info("Applying initial schema...")
        migrations_dir = Path("migrations")
        
        # Run migrations in order
        for migration in sorted(migrations_dir.glob("*.sql")):
            logger.info(f"Applying migration: {migration.name}")
            if not run_command(f"psql -d {DB_NAME} -f {migration}"):
                logger.error(f"Failed to apply migration: {migration.name}")
                return False
        
        # Update .env file with database URL
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_content = f.read()
            
            if "DATABASE_URL" not in env_content:
                with open(env_path, 'a') as f:
                    f.write(f"\nDATABASE_URL=postgresql://localhost/{DB_NAME}\n")
        else:
            with open(env_path, 'w') as f:
                f.write(f"DATABASE_URL=postgresql://localhost/{DB_NAME}\n")
        
        logger.info("Database setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False

if __name__ == "__main__":
    if setup_database():
        logger.info("✅ Database setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Database setup failed!")
        sys.exit(1) 