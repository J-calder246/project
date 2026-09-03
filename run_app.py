#module for initialising Flask app

import os
from pathlib import Path
from routes import app


ROOT = Path('/workspaces/project').resolve()
TEMPLATES = ROOT / 'templates'

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=4000) #runs app on port 4000  
