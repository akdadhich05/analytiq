import subprocess
import sys

# Function to install a package if it's not already installed
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install SQLAlchemy if not installed
try:
    from sqlalchemy import create_engine, Column, Integer, String, Float
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
except ImportError:
    install_package("SQLAlchemy")
    from sqlalchemy import create_engine, Column, Integer, String, Float
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker

# Define the SQLite database
DATABASE_URL = "sqlite:///./mydatabase.db"

# Create an engine and a session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for our models
Base = declarative_base()

# Define your database models
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    filepath = Column(String)

# Create the database tables
Base.metadata.create_all(bind=engine)

# Dependency to get the session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
