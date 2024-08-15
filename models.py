import subprocess
import sys

# Function to install a package if it's not already installed
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install SQLAlchemy if not installed
try:
    from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, sessionmaker
except ImportError:
    install_package("SQLAlchemy")
    from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, sessionmaker

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

    # Relationship to DQRule
    rules = relationship("DQRule", back_populates="dataset")


class DQRule(Base):
    __tablename__ = "dq_rules"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    rule_name = Column(String, nullable=False)
    rule_type = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    condition = Column(Text, nullable=False)
    severity = Column(String, nullable=False)
    message = Column(Text, nullable=True)

    dataset = relationship("Dataset", back_populates="rules")

# Create the database tables
Base.metadata.create_all(bind=engine)

# Dependency to get the session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
