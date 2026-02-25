import sys
import os

print('Python', sys.version)
# Ensure the project root is on sys.path so 'import server.*' works reliably.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
print('Added to sys.path:', repo_root)
try:
    from server.services.orchestrator_adapter import MockOrchestrator, get_orchestrator_adapter
    from server.services.llm_adapter import get_llm_adapter
    from server.services.faiss_store import store
    print('Imports OK')
except Exception as e:
    print('Import error', e)
    raise
