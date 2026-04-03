from typing import Any, Dict


class OCREngine():
    """Заглушка для будущего OCR движка."""

    def __init__(self):
        pass

    def recognize_text(self, image) -> Dict[str, Any]:
        """Заглушка метода распознавания текста."""
        return {
            "recognized_text": None,
            "confidence": 0.0,
            "message": "OCR Engine not implemented yet"
        }