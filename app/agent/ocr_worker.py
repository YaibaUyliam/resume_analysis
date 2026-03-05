import tempfile
import numpy as np

from paddleocr import PaddleOCR


def paddleocrv3_output_to_text(rec_polys, rec_texts, rec_scores):
    text_by_line = ""
    ocrOnly = {}

    for idx in range(len(rec_polys)):
        points = np.array(rec_polys[idx]).astype(np.int32).tolist()
        x1 = min(points[0][0], points[1][0], points[2][0], points[3][0])
        x2 = max(points[0][0], points[1][0], points[2][0], points[3][0])
        y1 = min(points[0][1], points[1][1], points[2][1], points[3][1])
        y2 = max(points[0][1], points[1][1], points[2][1], points[3][1])
        y_c = int((y1 + y2) / 2)

        if idx == 0:
            ocrOnly[0] = [[x1, y1, x2, y2]]
            if rec_scores[idx] > 0.3:
                text_by_line += rec_texts[idx]
                text_by_line += " "

        if idx > 0:
            sameLine = False
            for key in ocrOnly:
                for idxBb, bbox in enumerate(ocrOnly[key]):
                    x1_l, y1_l, x2_l, y2_l = ocrOnly[key][idxBb]
                    if y1_l < y_c < y2_l:
                        sameLine = True
                        ocrOnly[key].append([x1, y1, x2, y2])
                        if rec_scores[idx] > 0.3:
                            text_by_line += rec_texts[idx]
                            text_by_line += " "

                    if sameLine:
                        break
                if sameLine:
                    break
            if sameLine == False:
                key = [key for key in ocrOnly][-1] + 1
                ocrOnly[key] = [[x1, y1, x2, y2]]
                if rec_scores[idx] > 0.3:
                    text_by_line = text_by_line.strip()
                    text_by_line += "\n"
                    text_by_line += rec_texts[idx]
                    text_by_line += " "

    return text_by_line.strip()


def run_ocr_process(data, result_queue):
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="ch",
        text_detection_model_dir="./ckpts/PP-OCRv5_server_det",
        text_recognition_model_dir="./ckpts/PP-OCRv5_server_rec",
    )

    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        temp_pdf.write(data)
        temp_pdf.flush()

        result = ocr.predict(temp_pdf.name)
        text_by_line = ""
        for res in result:
            convert = paddleocrv3_output_to_text(
                res["rec_polys"], res["rec_texts"], res["rec_scores"]
            )
            text_by_line += convert + "\n\n"

        result_queue.put(text_by_line)