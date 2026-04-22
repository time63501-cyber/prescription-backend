import os
from app.services.ocr_service import OCRService
from app.services.advanced_handwriting_ocr import get_advanced_ocr

SAMPLES_DIR = "Samples"

ocr_service = OCRService()
advanced_ocr = get_advanced_ocr()

def batch_test_samples():
    results = []
    for fname in os.listdir(SAMPLES_DIR):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            continue
        path = os.path.join(SAMPLES_DIR, fname)
        print(f"\n=== Testing: {fname} ===")
        try:
            # Enhanced OCR with fallback
            tokens, text, confidence = ocr_service.process_image_with_fallback(path)
            print(f"[Enhanced OCR] Final confidence: {confidence}%")
            
            # Advanced OCR analysis
            adv_result = advanced_ocr.process_prescription_image(path)
            print(f"[Advanced OCR] Quality: {adv_result.get('overall_confidence')}%")
            print(f"[Advanced OCR] Recommendation: {adv_result.get('recommendation')}")
            
            # Handwriting enhancement
            if tokens:
                enhanced_tokens = ocr_service.enhance_tokens_with_handwriting(tokens, path)
                print(f"[Handwriting Enhancement] Enhanced {len(enhanced_tokens)} tokens")
            
            # Print a snippet of recognized text
            print(f"[OCR] Text: {text[:150].replace(chr(10), ' ')}...")
            
            # Handwriting style analysis
            style_analysis = advanced_ocr.analyze_handwriting_style(path)
            if style_analysis.get('success'):
                print(f"[Handwriting Style] Quality: {style_analysis.get('handwriting_quality_index')}%")
                print(f"[Handwriting Style] Characters: {style_analysis.get('character_count')}")
            
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            import traceback
            traceback.print_exc()
        
        results.append({
            'file': fname,
            'confidence': confidence if 'confidence' in locals() else 0,
            'adv_conf': adv_result.get('overall_confidence', 0) if 'adv_result' in locals() else 0,
            'recommendation': adv_result.get('recommendation', 'error') if 'adv_result' in locals() else 'error',
            'tokens': len(tokens) if 'tokens' in locals() else 0
        })
    
    print("\n=== Batch Test Summary ===")
    for r in results:
        status = "✅" if r['confidence'] >= 70 else "⚠️" if r['confidence'] >= 50 else "❌"
        print(f"{status} {r['file']}: Conf={r['confidence']}%  Tokens={r['tokens']}  Rec={r['recommendation']}")
    
    # Overall statistics
    high_conf = sum(1 for r in results if r['confidence'] >= 70)
    med_conf = sum(1 for r in results if 50 <= r['confidence'] < 70)
    low_conf = sum(1 for r in results if r['confidence'] < 50)
    
    print(f"\n=== Statistics ===")
    print(f"High confidence (≥70%): {high_conf}/{len(results)}")
    print(f"Medium confidence (50-69%): {med_conf}/{len(results)}")
    print(f"Low confidence (<50%): {low_conf}/{len(results)}")
    print(".1f")

if __name__ == "__main__":
    batch_test_samples()
