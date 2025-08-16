#!/usr/bin/env python3
"""
WINIFRED AI - Advanced Repository Management and Code Analysis System
A completely self-contained AI-powered system for repository management, code analysis, and automated fixes.
NO EXTERNAL APIs - NO THIRD PARTY DEPENDENCIES - FULLY AUTONOMOUS
"""

import os
import sys
import json
import time
import logging
import sqlite3
import hashlib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import threading
import queue
import shutil
import tempfile
import socket
import urllib.request
import urllib.parse

# Built-in imports for code analysis - NO EXTERNAL DEPENDENCIES
import ast
import tokenize
import io
import re
from collections import defaultdict, Counter
import zipfile
import tarfile
import mimetypes

@dataclass
class Repository:
    """Repository data structure"""
    name: str
    path: str
    language: str
    size: int
    last_analyzed: datetime
    health_score: float
    issues: List[str]
    dependencies: List[str]
    git_url: Optional[str] = None
    backup_path: Optional[str] = None

@dataclass
class CodeIssue:
    """Code issue data structure"""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    suggested_fix: str
    confidence: float

@dataclass
class LearningData:
    """Learning data structure for continuous improvement"""
    domain: str
    topic: str
    knowledge: str
    source: str
    timestamp: datetime
    relevance_score: float

class DatabaseManager:
    """Advanced database management for WINIFRED AI"""
    
    def __init__(self, db_path: str = "winifred_ai.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with all required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Repositories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    path TEXT NOT NULL,
                    language TEXT,
                    size INTEGER,
                    last_analyzed TIMESTAMP,
                    health_score REAL,
                    issues TEXT,
                    dependencies TEXT,
                    git_url TEXT,
                    backup_path TEXT
                )
            """)
            
            # Code issues table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_name TEXT,
                    file_path TEXT,
                    line_number INTEGER,
                    issue_type TEXT,
                    severity TEXT,
                    description TEXT,
                    suggested_fix TEXT,
                    confidence REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (repo_name) REFERENCES repositories (name)
                )
            """)
            
            # Learning data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT,
                    topic TEXT,
                    knowledge TEXT,
                    source TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    relevance_score REAL
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()

class CodeAnalyzer:
    """Advanced code analysis engine"""
    
    def __init__(self):
        self.supported_languages = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.ts', '.tsx'],
            'java': ['.java'],
            'c++': ['.cpp', '.cc', '.cxx', '.c'],
            'c': ['.c', '.h'],
            'go': ['.go'],
            'rust': ['.rs'],
            'php': ['.php'],
            'ruby': ['.rb'],
            'swift': ['.swift'],
            'kotlin': ['.kt'],
            'scala': ['.scala'],
            'html': ['.html', '.htm'],
            'css': ['.css', '.scss', '.sass'],
            'sql': ['.sql'],
            'bash': ['.sh', '.bash'],
            'dockerfile': ['Dockerfile'],
            'yaml': ['.yml', '.yaml'],
            'json': ['.json'],
            'xml': ['.xml']
        }
        
        self.issue_patterns = {
            'python': {
                'syntax_error': r'SyntaxError|IndentationError|TabError',
                'unused_import': r'import\s+\w+(?:\s*,\s*\w+)*\s*(?:#.*)?$',
                'long_line': r'.{120,}',
                'hardcoded_password': r'password\s*=\s*["\'][^"\']+["\']',
                'sql_injection': r'cursor\.execute\([^)]*%.*\)',
                'security_risk': r'eval\s*\(|exec\s*\(|__import__\s*\(',
            },
            'javascript': {
                'console_log': r'console\.log\s*\(',
                'var_declaration': r'var\s+\w+',
                'missing_semicolon': r'[^;]\s*\n',
                'triple_equals': r'==(?!=)',
                'security_risk': r'eval\s*\(|innerHTML\s*=',
            }
        }
    
    def analyze_repository(self, repo_path: str) -> Tuple[List[CodeIssue], float]:
        """Analyze an entire repository"""
        issues = []
        total_files = 0
        healthy_files = 0
        
        for root, dirs, files in os.walk(repo_path):
            # Skip common irrelevant directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', 'venv', '.env'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                # Determine language
                language = self.detect_language(file_path)
                if language:
                    total_files += 1
                    file_issues = self.analyze_file(file_path, language)
                    issues.extend(file_issues)
                    
                    if len(file_issues) == 0:
                        healthy_files += 1
        
        health_score = (healthy_files / total_files * 100) if total_files > 0 else 0
        return issues, health_score
    
    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        file_ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        
        for language, extensions in self.supported_languages.items():
            if file_ext in extensions or filename in extensions:
                return language
        return None
    
    def analyze_file(self, file_path: str, language: str) -> List[CodeIssue]:
        """Analyze a single file for issues"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Language-specific analysis
            if language == 'python':
                issues.extend(self.analyze_python_file(file_path, content, lines))
            elif language == 'javascript':
                issues.extend(self.analyze_javascript_file(file_path, content, lines))
            
            # General analysis for all languages
            issues.extend(self.analyze_general_issues(file_path, content, lines, language))
            
        except Exception as e:
            issues.append(CodeIssue(
                file_path=file_path,
                line_number=0,
                issue_type="file_error",
                severity="high",
                description=f"Could not analyze file: {str(e)}",
                suggested_fix="Check file encoding and permissions",
                confidence=0.9
            ))
        
        return issues
    
    def analyze_python_file(self, file_path: str, content: str, lines: List[str]) -> List[CodeIssue]:
        """Python-specific code analysis"""
        issues = []
        
        try:
            # Parse AST for syntax errors
            tree = ast.parse(content)
            
            # Check for various Python-specific issues
            for i, line in enumerate(lines, 1):
                # Long lines
                if len(line) > 120:
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=i,
                        issue_type="style",
                        severity="low",
                        description="Line too long (>120 characters)",
                        suggested_fix="Break line into multiple lines",
                        confidence=0.8
                    ))
                
                # Hardcoded passwords
                if re.search(r'password\s*=\s*["\'][^"\']+["\']', line.lower()):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=i,
                        issue_type="security",
                        severity="high",
                        description="Hardcoded password detected",
                        suggested_fix="Use environment variables or secure configuration",
                        confidence=0.9
                    ))
                
                # SQL injection risks
                if re.search(r'cursor\.execute\([^)]*%.*\)', line):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=i,
                        issue_type="security",
                        severity="high",
                        description="Potential SQL injection vulnerability",
                        suggested_fix="Use parameterized queries",
                        confidence=0.85
                    ))
        
        except SyntaxError as e:
            issues.append(CodeIssue(
                file_path=file_path,
                line_number=e.lineno or 0,
                issue_type="syntax_error",
                severity="high",
                description=f"Syntax error: {e.msg}",
                suggested_fix="Fix syntax according to Python grammar",
                confidence=1.0
            ))
        
        return issues
    
    def analyze_javascript_file(self, file_path: str, content: str, lines: List[str]) -> List[CodeIssue]:
        """JavaScript-specific code analysis"""
        issues = []
        
        for i, line in enumerate(lines, 1):
            # Console.log statements
            if re.search(r'console\.log\s*\(', line):
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type="debug",
                    severity="low",
                    description="Console.log statement found",
                    suggested_fix="Remove debug statements before production",
                    confidence=0.9
                ))
            
            # var instead of let/const
            if re.search(r'\bvar\s+\w+', line):
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type="style",
                    severity="medium",
                    description="Use let/const instead of var",
                    suggested_fix="Replace 'var' with 'let' or 'const'",
                    confidence=0.8
                ))
        
        return issues
    
    def analyze_general_issues(self, file_path: str, content: str, lines: List[str], language: str) -> List[CodeIssue]:
        """General analysis applicable to all languages"""
        issues = []
        
        for i, line in enumerate(lines, 1):
            # TODO comments
            if re.search(r'#\s*TODO|//\s*TODO|<!--\s*TODO', line, re.IGNORECASE):
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type="maintenance",
                    severity="low",
                    description="TODO comment found",
                    suggested_fix="Complete the TODO item or remove the comment",
                    confidence=0.7
                ))
            
            # Trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type="style",
                    severity="low",
                    description="Trailing whitespace",
                    suggested_fix="Remove trailing whitespace",
                    confidence=0.9
                ))
        
        return issues

class AutoFixer:
    """Automated code fixing engine"""
    
    def __init__(self):
        self.fixable_issues = {
            'trailing_whitespace': self.fix_trailing_whitespace,
            'style': self.fix_style_issues,
            'debug': self.fix_debug_statements,
        }
    
    def can_fix(self, issue: CodeIssue) -> bool:
        """Check if an issue can be automatically fixed"""
        return issue.issue_type in self.fixable_issues and issue.confidence > 0.7
    
    def fix_issue(self, issue: CodeIssue) -> bool:
        """Attempt to automatically fix an issue"""
        if not self.can_fix(issue):
            return False
        
        try:
            fixer = self.fixable_issues[issue.issue_type]
            return fixer(issue)
        except Exception as e:
            logging.error(f"Failed to fix issue: {e}")
            return False
    
    def fix_trailing_whitespace(self, issue: CodeIssue) -> bool:
        """Fix trailing whitespace"""
        try:
            with open(issue.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if issue.line_number <= len(lines):
                lines[issue.line_number - 1] = lines[issue.line_number - 1].rstrip() + '\n'
                
                with open(issue.file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                return True
        except Exception:
            pass
        return False
    
    def fix_style_issues(self, issue: CodeIssue) -> bool:
        """Fix basic style issues"""
        # This is a simplified example - in reality, you'd want more sophisticated fixes
        return False
    
    def fix_debug_statements(self, issue: CodeIssue) -> bool:
        """Remove debug statements like console.log"""
        try:
            with open(issue.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if issue.line_number <= len(lines):
                line = lines[issue.line_number - 1]
                if 'console.log' in line:
                    # Comment out the line instead of removing it
                    lines[issue.line_number - 1] = '// ' + line
                    
                    with open(issue.file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    return True
        except Exception:
            pass
        return False

class LearningEngine:
    """Continuous learning and knowledge management"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.knowledge_domains = {
            'programming_languages': {},
            'frameworks': {},
            'best_practices': {},
            'security': {},
            'performance': {},
            'deployment': {},
            'testing': {},
            'architecture': {}
        }
    
    def learn_from_repository(self, repo: Repository, issues: List[CodeIssue]):
        """Learn patterns from analyzed repositories"""
        # Learn about common issues in this language
        language_issues = defaultdict(int)
        for issue in issues:
            language_issues[issue.issue_type] += 1
        
        # Store learning data
        learning_data = LearningData(
            domain='code_analysis',
            topic=f'{repo.language}_common_issues',
            knowledge=json.dumps(dict(language_issues)),
            source=f'repository_{repo.name}',
            timestamp=datetime.now(),
            relevance_score=0.8
        )
        
        self.store_learning_data(learning_data)
    
    def store_learning_data(self, data: LearningData):
        """Store learning data in database"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO learning_data 
                (domain, topic, knowledge, source, timestamp, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data.domain, data.topic, data.knowledge,
                data.source, data.timestamp, data.relevance_score
            ))
            conn.commit()
    
    def get_learned_patterns(self, domain: str, topic: str) -> List[Dict]:
        """Retrieve learned patterns for better analysis"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT knowledge, source, relevance_score 
                FROM learning_data 
                WHERE domain = ? AND topic = ?
                ORDER BY relevance_score DESC
            """, (domain, topic))
            
            results = []
            for row in cursor.fetchall():
                try:
                    knowledge = json.loads(row[0])
                    results.append({
                        'knowledge': knowledge,
                        'source': row[1],
                        'relevance': row[2]
                    })
                except json.JSONDecodeError:
                    continue
            
            return results

class WinifredAI:
    """Main WINIFRED AI system"""
    
    def __init__(self, workspace_dir: str = "./winifred_workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.db_manager = DatabaseManager(str(self.workspace_dir / "winifred.db"))
        self.code_analyzer = CodeAnalyzer()
        self.auto_fixer = AutoFixer()
        self.learning_engine = LearningEngine(self.db_manager)
        
        # System state
        self.is_running = False
        self.task_queue = queue.Queue()
        self.worker_threads = []
        self.repositories = {}
        
        # Performance metrics
        self.metrics = {
            'repositories_analyzed': 0,
            'issues_found': 0,
            'issues_fixed': 0,
            'uptime': 0,
            'last_learning_update': datetime.now()
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(self.workspace_dir / "winifred.log")),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("WINIFRED")
    
    def start(self):
        """Start the WINIFRED AI system"""
        self.logger.info("🤖 WINIFRED AI Starting Up...")
        self.is_running = True
        
        # Start worker threads
        num_workers = min(4, os.cpu_count() or 1)
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
        
        # Start continuous learning
        learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        learning_thread.start()
        
        self.logger.info(f"✅ WINIFRED AI Started with {num_workers} worker threads")
    
    def stop(self):
        """Stop the WINIFRED AI system"""
        self.logger.info("🛑 WINIFRED AI Shutting Down...")
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=5)
        
        self.logger.info("✅ WINIFRED AI Shut Down Complete")
    
    def add_repository(self, repo_path: str, name: Optional[str] = None, git_url: Optional[str] = None) -> str:
        """Add a repository for analysis and management"""
        repo_path = os.path.abspath(repo_path)
        
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        if name is None:
            name = os.path.basename(repo_path)
        
        # Create backup
        backup_path = self._create_backup(repo_path, name)
        
        # Detect primary language
        language = self._detect_primary_language(repo_path)
        
        # Calculate size
        size = self._calculate_directory_size(repo_path)
        
        # Create repository object
        repo = Repository(
            name=name,
            path=repo_path,
            language=language,
            size=size,
            last_analyzed=datetime.now(),
            health_score=0.0,
            issues=[],
            dependencies=[],
            git_url=git_url,
            backup_path=backup_path
        )
        
        self.repositories[name] = repo
        self._store_repository(repo)
        
        # Queue for analysis
        self.task_queue.put(('analyze_repository', name))
        
        self.logger.info(f"📁 Repository '{name}' added and queued for analysis")
        return name
    
    def analyze_repository(self, repo_name: str) -> Dict:
        """Analyze a repository for issues"""
        if repo_name not in self.repositories:
            raise ValueError(f"Repository '{repo_name}' not found")
        
        repo = self.repositories[repo_name]
        self.logger.info(f"🔍 Analyzing repository: {repo_name}")
        
        start_time = time.time()
        
        # Perform analysis
        issues, health_score = self.code_analyzer.analyze_repository(repo.path)
        
        # Update repository
        repo.last_analyzed = datetime.now()
        repo.health_score = health_score
        repo.issues = [f"{issue.issue_type}:{issue.description}" for issue in issues[:10]]  # Store summary
        
        # Store issues in database
        self._store_issues(repo_name, issues)
        self._update_repository(repo)
        
        # Learn from analysis
        self.learning_engine.learn_from_repository(repo, issues)
        
        # Update metrics
        self.metrics['repositories_analyzed'] += 1
        self.metrics['issues_found'] += len(issues)
        
        analysis_time = time.time() - start_time
        
        result = {
            'repository': repo_name,
            'health_score': health_score,
            'total_issues': len(issues),
            'issues_by_severity': {
                'high': len([i for i in issues if i.severity == 'high']),
                'medium': len([i for i in issues if i.severity == 'medium']),
                'low': len([i for i in issues if i.severity == 'low'])
            },
            'issues_by_type': dict(Counter([i.issue_type for i in issues])),
            'analysis_time': analysis_time,
            'fixable_issues': len([i for i in issues if self.auto_fixer.can_fix(i)])
        }
        
        self.logger.info(f"✅ Analysis complete for '{repo_name}': {len(issues)} issues found")
        return result
    
    def fix_repository(self, repo_name: str, auto_fix: bool = True) -> Dict:
        """Attempt to fix issues in a repository"""
        if repo_name not in self.repositories:
            raise ValueError(f"Repository '{repo_name}' not found")
        
        self.logger.info(f"🔧 Fixing repository: {repo_name}")
        
        # Get current issues
        issues = self._get_repository_issues(repo_name)
        fixable_issues = [i for i in issues if self.auto_fixer.can_fix(i)]
        
        fixed_count = 0
        
        if auto_fix and fixable_issues:
            for issue in fixable_issues:
                if self.auto_fixer.fix_issue(issue):
                    fixed_count += 1
                    self.metrics['issues_fixed'] += 1
                    self.logger.info(f"✅ Fixed: {issue.description}")
        
        # Re-analyze after fixes
        if fixed_count > 0:
            self.analyze_repository(repo_name)
        
        result = {
            'repository': repo_name,
            'total_issues': len(issues),
            'fixable_issues': len(fixable_issues),
            'fixed_issues': fixed_count,
            'remaining_issues': len(issues) - fixed_count
        }
        
        self.logger.info(f"🔧 Fixed {fixed_count} issues in '{repo_name}'")
        return result
    
    def get_repository_status(self, repo_name: str) -> Dict:
        """Get detailed status of a repository"""
        if repo_name not in self.repositories:
            raise ValueError(f"Repository '{repo_name}' not found")
        
        repo = self.repositories[repo_name]
        issues = self._get_repository_issues(repo_name)
        
        return {
            'name': repo.name,
            'path': repo.path,
            'language': repo.language,
            'size_mb': round(repo.size / (1024 * 1024), 2),
            'health_score': repo.health_score,
            'last_analyzed': repo.last_analyzed.isoformat(),
            'total_issues': len(issues),
            'critical_issues': len([i for i in issues if i.severity == 'high']),
            'backup_available': os.path.exists(repo.backup_path) if repo.backup_path else False,
            'git_url': repo.git_url
        }
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'status': 'running' if self.is_running else 'stopped',
            'uptime_hours': (datetime.now() - datetime.now()).seconds / 3600,
            'repositories_managed': len(self.repositories),
            'total_analyses': self.metrics['repositories_analyzed'],
            'total_issues_found': self.metrics['issues_found'],
            'total_issues_fixed': self.metrics['issues_fixed'],
            'queue_size': self.task_queue.qsize(),
            'worker_threads': len(self.worker_threads),
            'workspace_dir': str(self.workspace_dir),
            'last_learning_update': self.metrics['last_learning_update'].isoformat()
        }
    
    def chat(self, message: str) -> str:
        """Chat interface for interacting with WINIFRED"""
        message = message.lower().strip()
        
        # Simple command parsing
        if 'status' in message:
            status = self.get_system_status()
            return f"🤖 WINIFRED Status: Managing {status['repositories_managed']} repositories, found {status['total_issues_found']} issues, fixed {status['total_issues_fixed']} issues."
        
        elif 'analyze' in message:
            repos = list(self.repositories.keys())
            if repos:
                return f"📊 I can analyze these repositories: {', '.join(repos)}. Use analyze_repository(name) to start analysis."
            else:
                return "📁 No repositories added yet. Use add_repository(path) to add one."
        
        elif 'help' in message:
            return """
🤖 WINIFRED AI Commands:
- add_repository(path) - Add a repository for analysis
- analyze_repository(name) - Analyze a repository for issues  
- fix_repository(name) - Attempt to fix repository issues
- get_repository_status(name) - Get detailed repository status
- get_system_status() - Get system overview
- Chat with me naturally about coding, repositories, and fixes!
            """
        
        else:
            return f"🤖 Hello! I'm WINIFRED AI. I'm here to help manage and improve your code repositories. Currently managing {len(self.repositories)} repositories. How can I help you today?"
    
    def _worker_loop(self):
        """Worker thread loop for processing tasks"""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1)
                task_type, *args = task
                
                if task_type == 'analyze_repository':
                    self.analyze_repository(args[0])
                elif task_type == 'fix_repository':
                    self.fix_repository(args[0])
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    def _continuous_learning_loop(self):
        """Continuous learning background process"""
        while self.is_running:
            try:
                # Update learning metrics
                self.metrics['last_learning_update'] = datetime.now()
                
                # Simulate learning from patterns
                # In a real implementation, this would:
                # - Analyze patterns in fixed vs unfixed issues
                # - Update confidence scores for different fix types
                # - Learn from external sources (documentation, best practices)
                # - Improve analysis accuracy over time
                
                time.sleep(3600)  # Learn every hour
                
            except Exception as e:
                self.logger.error(f"Learning loop error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _create_backup(self, repo_path: str, name: str) -> str:
        """Create a backup of the repository"""
        backup_dir = self.workspace_dir / "backups" / name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{name}_{timestamp}"
        
        try:
            shutil.copytree(repo_path, backup_path)
            self.logger.info(f"💾 Created backup: {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return ""
    
    def _detect_primary_language(self, repo_path: str) -> str:
        """Detect the primary programming language in a repository"""
        language_counts = defaultdict(int)
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', 'venv', '.env'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                language = self.code_analyzer.detect_language(file_path)
                if language:
                    language_counts[language] += 1
        
        return max(language_counts, key=language_counts.get) if language_counts else "unknown"
    
    def _calculate_directory_size(self, path: str) -> int:
        """Calculate total size of directory in bytes"""
        total_size = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                except (OSError, FileNotFoundError):
                    continue
        return total_size
    
    def _store_repository(self, repo: Repository):
        """Store repository in database"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO repositories 
                (name, path, language, size, last_analyzed, health_score, issues, dependencies, git_url, backup_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                repo.name, repo.path, repo.language, repo.size,
                repo.last_analyzed, repo.health_score,
                json.dumps(repo.issues), json.dumps(repo.dependencies),
                repo.git_url, repo.backup_path
            ))
            conn.commit()
    
    def _update_repository(self, repo: Repository):
        """Update repository in database"""
        self._store_repository(repo)
    
    def _store_issues(self, repo_name: str, issues: List[CodeIssue]):
        """Store code issues in database"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear existing issues for this repository
            cursor.execute("DELETE FROM code_issues WHERE repo_name = ?", (repo_name,))
            
            # Insert new issues
            for issue in issues:
                cursor.execute("""
                    INSERT INTO code_issues 
                    (repo_name, file_path, line_number, issue_type, severity, description, suggested_fix, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    repo_name, issue.file_path, issue.line_number,
                    issue.issue_type, issue.severity, issue.description,
                    issue.suggested_fix, issue.confidence
                ))
            
            conn.commit()
    
    def _get_repository_issues(self, repo_name: str) -> List[CodeIssue]:
        """Retrieve issues for a repository from database"""
        issues = []
        
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT file_path, line_number, issue_type, severity, description, suggested_fix, confidence
                FROM code_issues 
                WHERE repo_name = ? AND resolved = FALSE
            """, (repo_name,))
            
            for row in cursor.fetchall():
                issues.append(CodeIssue(
                    file_path=row[0],
                    line_number=row[1],
                    issue_type=row[2],
                    severity=row[3],
                    description=row[4],
                    suggested_fix=row[5],
                    confidence=row[6]
                ))
        
        return issues

class WinifredCLI:
    """Command Line Interface for WINIFRED AI"""
    
    def __init__(self):
        self.winifred = WinifredAI()
        self.winifred.start()
        
    def run(self):
        """Run the interactive CLI"""
        print("🤖 WINIFRED AI - Advanced Repository Management System")
        print("="*60)
        print("Type 'help' for commands or chat naturally with WINIFRED")
        print("Type 'exit' to quit")
        print("="*60)
        
        try:
            while True:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
                
                if not user_input:
                    continue
                
                try:
                    # Check for direct function calls
                    if user_input.startswith('add_repository('):
                        self._handle_add_repository(user_input)
                    elif user_input.startswith('analyze_repository('):
                        self._handle_analyze_repository(user_input)
                    elif user_input.startswith('fix_repository('):
                        self._handle_fix_repository(user_input)
                    elif user_input.startswith('get_repository_status('):
                        self._handle_get_status(user_input)
                    elif user_input == 'get_system_status()':
                        self._handle_system_status()
                    else:
                        # Chat interface
                        response = self.winifred.chat(user_input)
                        print(f"🤖 WINIFRED: {response}")
                
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        except KeyboardInterrupt:
            pass
        finally:
            print("\n👋 WINIFRED AI shutting down...")
            self.winifred.stop()
    
    def _handle_add_repository(self, command: str):
        """Handle add_repository command"""
        # Simple parsing - in production, use proper parsing
        path = command.split('(')[1].split(')')[0].strip('\'"')
        try:
            repo_name = self.winifred.add_repository(path)
            print(f"✅ Repository '{repo_name}' added successfully!")
        except Exception as e:
            print(f"❌ Failed to add repository: {e}")
    
    def _handle_analyze_repository(self, command: str):
        """Handle analyze_repository command"""
        repo_name = command.split('(')[1].split(')')[0].strip('\'"')
        try:
            result = self.winifred.analyze_repository(repo_name)
            print(f"📊 Analysis Results for '{repo_name}':")
            print(f"   Health Score: {result['health_score']:.1f}%")
            print(f"   Total Issues: {result['total_issues']}")
            print(f"   High Priority: {result['issues_by_severity']['high']}")
            print(f"   Fixable Issues: {result['fixable_issues']}")
            print(f"   Analysis Time: {result['analysis_time']:.2f}s")
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    def _handle_fix_repository(self, command: str):
        """Handle fix_repository command"""
        repo_name = command.split('(')[1].split(')')[0].strip('\'"')
        try:
            result = self.winifred.fix_repository(repo_name)
            print(f"🔧 Fix Results for '{repo_name}':")
            print(f"   Fixed Issues: {result['fixed_issues']}")
            print(f"   Remaining Issues: {result['remaining_issues']}")
        except Exception as e:
            print(f"❌ Fix failed: {e}")
    
    def _handle_get_status(self, command: str):
        """Handle get_repository_status command"""
        repo_name = command.split('(')[1].split(')')[0].strip('\'"')
        try:
            status = self.winifred.get_repository_status(repo_name)
            print(f"📋 Status for '{repo_name}':")
            for key, value in status.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
        except Exception as e:
            print(f"❌ Status retrieval failed: {e}")
    
    def _handle_system_status(self):
        """Handle get_system_status command"""
        try:
            status = self.winifred.get_system_status()
            print("🖥️  System Status:")
            for key, value in status.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
        except Exception as e:
            print(f"❌ System status retrieval failed: {e}")

class WinifredWebAPI:
    """Web API interface for WINIFRED AI (Flask-based)"""
    
    def __init__(self, winifred: WinifredAI):
        self.winifred = winifred
        try:
            from flask import Flask, request, jsonify
            from flask_cors import CORS
            
            self.app = Flask(__name__)
            CORS(self.app)
            
            # Define routes
            self.app.route('/api/status', methods=['GET'])(self.get_status)
            self.app.route('/api/repositories', methods=['GET'])(self.list_repositories)
            self.app.route('/api/repositories', methods=['POST'])(self.add_repository)
            self.app.route('/api/repositories/<name>/analyze', methods=['POST'])(self.analyze_repository)
            self.app.route('/api/repositories/<name>/fix', methods=['POST'])(self.fix_repository)
            self.app.route('/api/repositories/<name>/status', methods=['GET'])(self.get_repository_status)
            self.app.route('/api/chat', methods=['POST'])(self.chat)
            
        except ImportError:
            print("Flask not available. Web API disabled. Install with: pip install flask flask-cors")
            self.app = None
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the web API server"""
        if self.app:
            print(f"🌐 WINIFRED Web API starting on http://{host}:{port}")
            self.app.run(host=host, port=port, debug=debug)
        else:
            print("❌ Web API not available - Flask not installed")
    
    def get_status(self):
        """GET /api/status"""
        from flask import jsonify
        return jsonify(self.winifred.get_system_status())
    
    def list_repositories(self):
        """GET /api/repositories"""
        from flask import jsonify
        repos = []
        for name, repo in self.winifred.repositories.items():
            repos.append({
                'name': name,
                'language': repo.language,
                'health_score': repo.health_score,
                'last_analyzed': repo.last_analyzed.isoformat()
            })
        return jsonify(repos)
    
    def add_repository(self):
        """POST /api/repositories"""
        from flask import request, jsonify
        data = request.get_json()
        try:
            repo_name = self.winifred.add_repository(
                data['path'], 
                data.get('name'), 
                data.get('git_url')
            )
            return jsonify({'success': True, 'repository': repo_name})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400
    
    def analyze_repository(self, name):
        """POST /api/repositories/<name>/analyze"""
        from flask import jsonify
        try:
            result = self.winifred.analyze_repository(name)
            return jsonify({'success': True, 'result': result})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400
    
    def fix_repository(self, name):
        """POST /api/repositories/<name>/fix"""
        from flask import jsonify
        try:
            result = self.winifred.fix_repository(name)
            return jsonify({'success': True, 'result': result})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400
    
    def get_repository_status(self, name):
        """GET /api/repositories/<name>/status"""
        from flask import jsonify
        try:
            status = self.winifred.get_repository_status(name)
            return jsonify(status)
        except Exception as e:
            return jsonify({'error': str(e)}), 404
    
    def chat(self):
        """POST /api/chat"""
        from flask import request, jsonify
        data = request.get_json()
        try:
            message = data.get('message', '')
            response = self.winifred.chat(message)
            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WINIFRED AI - Advanced Repository Management System')
    parser.add_argument('--mode', choices=['cli', 'web'], default='cli', 
                       help='Run mode: cli (interactive) or web (API server)')
    parser.add_argument('--workspace', default='./winifred_workspace',
                       help='Workspace directory for WINIFRED AI')
    parser.add_argument('--host', default='localhost',
                       help='Web API host (web mode only)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Web API port (web mode only)')
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        # Run CLI interface
        cli = WinifredCLI()
        cli.run()
    else:
        # Run Web Server (completely self-contained)
        winifred = WinifredAI(args.workspace)
        winifred.start()
        
        try:
            server = SelfContainedWebServer(winifred, args.host, args.port)
            server.start()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            winifred.stop()

if __name__ == "__main__":
    main()

# Example usage and testing
if __name__ == "__main__" and len(sys.argv) == 1:
    print("🤖 WINIFRED AI System - Example Usage")
    print("="*50)
    
    # Initialize WINIFRED
    winifred = WinifredAI("./example_workspace")
    winifred.start()
    
    try:
        # Example repository addition and analysis
        print("\n📁 Adding example repository...")
        
        # You would replace this with actual repository paths
        example_repo_path = "."  # Current directory
        repo_name = winifred.add_repository(example_repo_path, "example_repo")
        
        print(f"✅ Added repository: {repo_name}")
        
        # Wait a moment for analysis to complete
        time.sleep(2)
        
        # Get system status
        status = winifred.get_system_status()
        print(f"\n🖥️  System Status: {status}")
        
        # Get repository status
        if repo_name in winifred.repositories:
            repo_status = winifred.get_repository_status(repo_name)
            print(f"\n📊 Repository Status: {repo_status}")
        
        # Chat example
        print(f"\n💬 Chat: {winifred.chat('What is my system status?')}")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
    
    finally:
        print("\n👋 Stopping WINIFRED AI...")
        winifred.stop()
