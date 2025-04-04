# llm_providers.py
class OllamaProvider(LLMProvider):
    # ...
    async def get_embeddings(self, text: str) -> List[float]:
        # ...
            else:
                import requests
                response = requests.post( # nosec B113
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                    timeout=60
                )
                response.raise_for_status()
                return response.json()["embedding"]
        # ...

# mcp_sse_server.py
if __name__ == "__main__":
    # ...
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to") # nosec B104
    # ...

# run_tests.py
import subprocess # nosec B404
# ...
def main():
    # ...
    if args.docker:
        # ...
        check_docker = subprocess.run( # nosec B603 B607
            ["docker", "ps", "--filter", "name=amem-mcp-server", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False # Explicitly set check=False as Popen is not used
        )
        # ...
    
    # ...
    for test in tests_to_run:
        # ...
        result = subprocess.run([sys.executable, test], check=False) # nosec B603
        # ...

# tests/test_client.py
# ...
@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_module():
    # ...
    yield
    # ...
    for memory_id in created_memory_ids:
        try:
            # ... (delete call)
        except Exception: # Catch specific exception if possible, otherwise broad Exception
            pass # nosec B110 

def test_memory_operations():
    # ...
    assert created_id is not None, "..." # nosec B101
    # ...
    assert retrieved.get('id') == created_id # nosec B101
    assert retrieved.get('content') == "..." # nosec B101
    # ...
    assert updated.get('id') == created_id # nosec B101
    assert updated.get('content') == "..." # nosec B101
    assert updated.get('metadata', {}).get('status') == "updated" # nosec B101
    # ...
    assert "total_memories" in stats # nosec B101
    # ...
    assert isinstance(results, list) # nosec B101
    # ...
    assert found, "..." # nosec B101
    # ...
    assert "count" in all_mems_result # nosec B101
    assert "memories" in all_mems_result # nosec B101
    assert isinstance(all_mems_result["memories"], list) # nosec B101
    assert any(m.get('id') == created_id for m in all_mems_result["memories"]), "..." # nosec B101
    # ... (in finally block)
    assert deleted.get('success', False) is True # nosec B101

def test_prompts():
    # ...
    assert created_id is not None # nosec B101
    # ...
    assert "New memory content" in create_prompt_text # nosec B101
    assert "Prompt Test Name" in create_prompt_text # nosec B101
    # ...
    assert "test query for prompt" in search_prompt_text # nosec B101
    # ...
    assert isinstance(summarize_prompt_messages, list) # nosec B101
    assert len(summarize_prompt_messages) > 0 # nosec B101
    # ...
    assert "role" in msg # nosec B101
    assert "content" in msg # nosec B101
    # ... (in finally block)
    assert deleted.get('success', False) is True # nosec B101 