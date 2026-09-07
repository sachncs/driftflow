"""Tests for CONTRIBUTING.md fix."""
import re
from pathlib import Path


def test_contributing_no_placeholders():
    """Test that CONTRIBUTING.md has no placeholder text."""
    contributing_path = Path(__file__).parent.parent / "CONTRIBUTING.md"
    
    with open(contributing_path, "r") as f:
        content = f.read()
    
    # Check for placeholder patterns
    placeholders = [
        r"YOUR_USERNAME",
        r"example/driftflow",
        r"example\.com",
    ]
    
    for pattern in placeholders:
        matches = re.findall(pattern, content)
        assert len(matches) == 0, f"Found placeholder '{pattern}' in CONTRIBUTING.md"


def test_contributing_has_correct_repo():
    """Test that CONTRIBUTING.md has correct repository URL."""
    contributing_path = Path(__file__).parent.parent / "CONTRIBUTING.md"
    
    with open(contributing_path, "r") as f:
        content = f.read()
    
    # Check for correct repository URL
    assert "sachncs/driftflow" in content, "CONTRIBUTING.md should reference sachncs/driftflow"


def test_contributing_has_clone_command():
    """Test that CONTRIBUTING.md has valid clone command."""
    contributing_path = Path(__file__).parent.parent / "CONTRIBUTING.md"
    
    with open(contributing_path, "r") as f:
        content = f.read()
    
    # Check for valid clone command
    assert "git clone https://github.com/sachncs/driftflow.git" in content, \
        "CONTRIBUTING.md should have valid clone command"


if __name__ == "__main__":
    test_contributing_no_placeholders()
    test_contributing_has_correct_repo()
    test_contributing_has_clone_command()
    print("✅ All tests passed!")
