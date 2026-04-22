import os
from app.services.ocr_service import OCRService
from app.services.advanced_handwriting_ocr import get_advanced_ocr

ocr_service = OCRService()
advanced_ocr = get_advanced_ocr()

def quick_test():
    # Test the best sample first
    test_file = "Samples/IMG-20260422-WA0002.jpg"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    print(f"=== Quick Test: {test_file} ===")

    try:
        # Enhanced OCR with fallback
        tokens, text, confidence = ocr_service.process_image_with_fallback(test_file)
        print(f"✅ Final confidence: {confidence}%")
        print(f"📝 Text: {text[:200].replace(chr(10), ' ')}...")

        # Advanced OCR analysis
        adv_result = advanced_ocr.process_prescription_image(test_file)
        print(f"🔍 Quality metrics: {adv_result.get('quality_metrics')}")

        # Handwriting style
        style = advanced_ocr.analyze_handwriting_style(test_file)
        if style.get('success'):
            print(f"✍️ Handwriting quality: {style.get('handwriting_quality_index')}%")

        print(f"\n🎯 SUCCESS: Achieved {confidence}% confidence!")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()