from captcha.image import ImageCaptcha
import random
import string
import io

# In-memory storage for captcha text. In production, use Redis or another cache.
captcha_store = {}

image = ImageCaptcha()

def generate_captcha():
    """Generates a new captcha image and text."""
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    # Generate a unique session ID for this captcha
    session_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))

    # Store the text with the session ID
    captcha_store[session_id] = captcha_text

    # Generate image data
    data = image.generate(captcha_text)
    image_bytes = io.BytesIO()
    data.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    return session_id, image_bytes

def verify_captcha(session_id: str, user_input: str) -> bool:
    """Verifies the user's input against the stored captcha text."""
    correct_text = captcha_store.get(session_id)
    if not correct_text:
        return False

    # Captcha is single-use, remove it after verification
    del captcha_store[session_id]

    return user_input.upper() == correct_text.upper()
