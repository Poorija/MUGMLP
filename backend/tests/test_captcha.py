
import pytest
from backend import captcha_handler

def test_generate_captcha():
    """Test that a session ID and image data are returned."""
    session_id, image_bytes = captcha_handler.generate_captcha()
    assert isinstance(session_id, str)
    assert len(session_id) == 16
    assert image_bytes.getbuffer().nbytes > 0

def test_verify_captcha_correct():
    """Test successful verification."""
    session_id, _ = captcha_handler.generate_captcha()
    # Retrieve the correct text from the store for testing
    correct_text = captcha_handler.captcha_store[session_id]
    assert captcha_handler.verify_captcha(session_id, correct_text) is True

def test_verify_captcha_incorrect():
    """Test failed verification with wrong text."""
    session_id, _ = captcha_handler.generate_captcha()
    assert captcha_handler.verify_captcha(session_id, "WRONGTEXT") is False

def test_verify_captcha_case_insensitive():
    """Test that verification is case-insensitive."""
    session_id, _ = captcha_handler.generate_captcha()
    correct_text = captcha_handler.captcha_store[session_id]
    assert captcha_handler.verify_captcha(session_id, correct_text.lower()) is True
    # It should be deleted after the first verification
    assert captcha_handler.verify_captcha(session_id, correct_text) is False

def test_verify_captcha_single_use():
    """Test that a captcha is deleted after use."""
    session_id, _ = captcha_handler.generate_captcha()
    correct_text = captcha_handler.captcha_store[session_id]
    # First verification should succeed
    assert captcha_handler.verify_captcha(session_id, correct_text) is True
    # Second verification should fail as it's been deleted
    assert captcha_handler.verify_captcha(session_id, correct_text) is False

def test_verify_non_existent_session():
    """Test verification with a session ID that does not exist."""
    assert captcha_handler.verify_captcha("nonexistentid", "sometext") is False
