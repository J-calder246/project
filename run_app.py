import os
from pathlib import Path
from routes import app


ROOT = Path('/workspaces/project').resolve()
TEMPLATES = ROOT / 'templates'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) #runs app on port 5000  
