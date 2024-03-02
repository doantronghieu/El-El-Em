# First Time Setup

```
# Create a virtual environment
python -m venv .venv

# On MacOS, WSL, Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize the database
flask --app app.web init-db
```

# Running the app

There are three separate processes that need to be running for the app to work: the server, the worker, and Redis.

If you stop any of these processes, you will need to start them back up!

Commands to start each are listed below.  If you need to stop them, select the terminal window the process is running in and press Control-C

### To run the Python server

```
inv dev
```

### To run the worker
```
inv devworker
```

### To run Redis
```
redis-server
```

### To reset the database
```
flask --app app.web init-db
```
upload -> python app.py