from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
import re
import logging

app = FastAPI()

# Cấu hình API key cho Google Generative AI
genai.configure(api_key="AIzaSyDmzM-2FgbpkBH-A3yH-jZdH3eZp-mtUGo")

# Cấu hình mô hình sinh nội dung
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

class PatientHistory(BaseModel):
    address: str
    exercises_regularly: str
    exposure_to_toxic_substances: str
    family_autoimmune_disease: str
    family_cardiovascular_disease: str
    family_genetic_disease: str
    family_malignant_disease: str
    gender: str
    had_surgery: str
    job: str
    past_disease_treatment: str
    past_diseases: str
    surgery_type: str
    user: str
    uses_alcohol: str
    uses_stimulants: str
    uses_tobacco: str
    visited_epidemic_areas: str

class DiagnosisRequest(BaseModel):
    patientHistory: PatientHistory
    question1: str
    question2: str
    question3: str
    question4: str
    question5: str
    question6: str
    question7: str

@app.post("/diagnosis")
async def get_diagnosis(request: DiagnosisRequest):
    # Tạo prompt từ dữ liệu request
    prompt = f"""
    Bạn là một bác sĩ tuyệt vời, bạn cần chẩn đoán bệnh cho bệnh nhân này.
    Bệnh nhân sinh sống tại {request.patientHistory.address}, giới tính {request.patientHistory.gender}, {request.patientHistory.past_diseases} mắc bệnh trước đây và {request.patientHistory.past_disease_treatment}.
    Bệnh nhân {request.patientHistory.had_surgery} phẫu thuật trước đây và đó là phẫu thuật {request.patientHistory.surgery_type}.
    Bệnh nhân {request.patientHistory.exposure_to_toxic_substances} tiền sử tiếp xúc với chất độc hại, làm việc trong môi trường nguy hiểm.
    Bệnh nhân {request.patientHistory.visited_epidemic_areas} đi đến vùng dịch tễ trước đây.
    Trong gia đình bệnh nhân {request.patientHistory.family_genetic_disease} người từng mắc bệnh di truyền, {request.patientHistory.family_cardiovascular_disease} người nhà mắc bệnh tim mạch, {request.patientHistory.family_malignant_disease} người mắc bệnh ác tính, {request.patientHistory.family_autoimmune_disease} người mắc bệnh tự miễn.
    Bệnh nhân {request.patientHistory.uses_tobacco} thói quen hút thuốc, {request.patientHistory.uses_alcohol} thói quen uống rượu, {request.patientHistory.uses_stimulants} thói quen sử dụng kích thích.
    Bệnh nhân {request.patientHistory.exercises_regularly} thường xuyên vận động và luyện tập thể dục thể thao.
    Hôm nay, bệnh nhân cảm thấy {request.question1}. Triệu chứng đầu tiên mà bệnh nhân gặp phải là {request.question2}. Triệu chứng này bắt đầu từ {request.question3}.
    Từ lúc bắt đầu cho đến nay, triệu chứng {request.question4} gây cho bệnh nhân những điều khó chịu.
    Trước đó, bệnh nhân {request.question5} làm để giảm nhẹ triệu chứng này. Trước đó, bệnh nhân {request.question6} sử dụng thuốc. Thuốc đó {request.question7} hiệu quả.
    Hãy chẩn đoán kết quả bệnh trả về là gồm tên bệnh, mô tả ngắn gọn của bệnh, lời khuyên sức khỏe, top 3 bệnh có khả năng gặp.
    Trả lời theo dàn ý cố định, không thay đổi bất cứ thứ gì như sau:

    ## Chẩn đoán bệnh cho bệnh nhân:

    **1. Top 3 Bệnh bệnh nhân có khả năng gặp là:**

    - **Top 1: Tên bệnh:**
        - **Mô tả ngắn gọn Top 1 bệnh:**
    - **Top 2: Tên bệnh:**
        - **Mô tả ngắn gọn Top 2 bệnh:**
    - **Top 3: Tên bệnh:**
        - **Mô tả ngắn gọn Top 3 bệnh:**
    
    **2. Top 3 lời khuyên sức khỏe:**

    - **Top 1:**
    - **Top 2:** 
    - **Top 3:** 

    Thêm dòng này sau mỗi lần chẩn đoán: Lưu ý: Những thông tin này chỉ mang tính chất tham khảo, không thay thế cho lời khuyên của bác sĩ. Hãy đến gặp bác sĩ để được chẩn đoán và điều trị phù hợp.
    
    Ví dụ cụ thể như sau:
    ## Chẩn đoán bệnh cho bệnh nhân:

    **1. Top 3 Bệnh bệnh nhân có khả năng gặp là:**

    - **Top 1: Cảm cúm:**
        - **Mô tả ngắn gọn Top 1 bệnh:** Cảm cúm là bệnh nhiễm trùng đường hô hấp do virus cúm gây ra. Triệu chứng thường gặp là chảy nước mũi, hắt hơi, sốt, ho, đau đầu, mệt mỏi. Bệnh thường tự khỏi trong vòng 1-2 tuần.
    - **Top 2: Viêm xoang:**
        - **Mô tả ngắn gọn Top 2 bệnh:** Viêm xoang là tình trạng viêm nhiễm niêm mạc xoang, gây ra các triệu chứng như nghẹt mũi, chảy nước mũi, đau 
    đầu, đau mặt.
    - **Top 3:  Viêm mũi dị ứng:**
        - **Mô tả ngắn gọn Top 3 bệnh:** Viêm mũi dị ứng là phản ứng của cơ thể với các tác nhân dị ứng như phấn hoa, bụi, nấm mốc, lông thú,... gây ra các triệu chứng như nghẹt mũi, chảy nước mũi, hắt hơi, ngứa mũi, ngứa mắt.

    **2. Top 3 lời khuyên sức khỏe:**

    - **Top 1:** Nghỉ ngơi, uống nhiều nước, ăn uống đầy đủ chất dinh dưỡng để tăng cường sức đề kháng.
    - **Top 2:** Sử dụng thuốc theo chỉ định của bác sĩ để giảm triệu chứng và phòng ngừa biến chứng.
    - **Top 3:** Tránh tiếp xúc với khói bụi, hóa chất, người bệnh để hạn chế nguy cơ lây nhiễm.

    **Lưu ý:** Những thông tin này chỉ mang tính chất tham khảo, không thay thế cho lời khuyên của bác sĩ. Hãy đến gặp bác sĩ để được chẩn đoán và điều trị phù hợp.
    """

    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)

        # Trích xuất thông tin từ response
        diagnosis_text = response.text
        logging.info(f"Diagnosis response text: {diagnosis_text}")
        print(f"Diagnosis response text: {diagnosis_text}")

        # Sử dụng regex để lấy thông tin từ response text
        disease_match = re.search(r"- \*\*Top 1:\s*(.*?):\*\*", diagnosis_text)
        description_match = re.search(r"- \*\*Mô tả ngắn gọn Top 1 bệnh:\s*(.*?)\n", diagnosis_text)
        other_diseases_matches = re.findall(r"- \*\*Top (\d+):\s*(.*?):\*\*\n\s*- \*\*Mô tả ngắn gọn Top \d+ bệnh:\s*(.*?)\n", diagnosis_text)
        treatment_matches = re.findall(r"- \*\*Top \d+:\*\* (.*?)\n", diagnosis_text.split("**2. Top 3 lời khuyên sức khỏe:**")[1])

        if not disease_match or not description_match or not treatment_matches:
            raise ValueError("Failed to parse diagnosis response")

        # Xử lý mô tả bệnh và mô tả bệnh khác để loại bỏ ** và khoảng trắng không mong muốn
        description = description_match.group(1).strip().replace("** ", "").lstrip()
        other_diseases = []
        for match in other_diseases_matches:
            other_diseases.append({
                "name": match[1],
                "description": match[2].strip().replace("** ", "").lstrip()
            })

        # Xử lý treatment để loại bỏ khoảng trắng không mong muốn ở đầu
        cleaned_treatments = [treatment.strip() for treatment in treatment_matches]

        # Cấu trúc lại response theo yêu cầu
        diagnosis_response = {
            "disease": disease_match.group(1),
            "description": description,
            "treatment": cleaned_treatments,
            "otherDiseases": other_diseases
        }

        diagnosis_response["otherDiseases"] = diagnosis_response["otherDiseases"][1:]
        return diagnosis_response

    except Exception as e:
        logging.error(f"Error during diagnosis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Chạy ứng dụng FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)