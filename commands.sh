# Create .gitignore file
cat > .gitignore << 'EOL'
# Virtual Environment
venv/
env/
ENV/
.env
.venv
.ENV
.VENV

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/raw/*
data/processed/*
data/embeddings/*
logs/*
!logs/.gitkeep
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/embeddings/.gitkeep

# Chrome extension build
src/chrome_extension/dist/

# Coverage
htmlcov/
.coverage

# Environment variables
.env
.env.*
!.env.example

# System
.DS_Store
Thumbs.db
EOL

# Now add and commit the files
git add .gitignore
git commit -m "Add .gitignore"
git add .
git commit -m "Initial commit: YouTube RAG Pipeline"
git remote add origin https://github.com/Lakshrajjain/YouTube_RAG.git
git push -u origin main
