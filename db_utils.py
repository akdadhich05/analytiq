import subprocess
import sys

from models import Dataset, get_db


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install SQLAlchemy if not installed
try:
    from sqlalchemy.orm import Session
except ImportError:
    install_package("SQLAlchemy")
    from sqlalchemy.orm import Session


# Function to add a new dataset to the database
def add_dataset(name, description, filepath):
    db: Session = next(get_db())
    new_dataset = Dataset(name=name, description=description, filepath=filepath)
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)
    return new_dataset
