# =====================================
# LLM Module - Llama 3 Integration
# Vietnamese Student Feedback Chatbot
# Expert-Assistant Architecture
# =====================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

# =====================================
# CONFIG
# =====================================
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# PROMPT TEMPLATES
# =====================================

SYSTEM_PROMPT = """Bạn là trợ lý ảo thông minh của Phòng Đào tạo Đại học Công nghệ Thông tin (UIT).
Nhiệm vụ của bạn là đọc phản hồi từ sinh viên, phân tích kết quả từ hệ thống AI, và viết câu trả lời cảm thông, chuyên nghiệp.

QUY TẮC QUAN TRỌNG:
- Luôn thể hiện sự lắng nghe và trân trọng ý kiến sinh viên
- Với phản hồi TIÊU CỰC: Thừa nhận vấn đề, xin lỗi, cam kết chuyển phản ánh đến bộ phận liên quan
- Với phản hồi TÍCH CỰC: Ghi nhận và cảm ơn, khuyến khích tiếp tục đóng góp
- Với phản hồi TRUNG TÍNH: Ghi nhận thông tin, hỏi thêm nếu cần
- Câu trả lời ngắn gọn (2-4 câu), tự nhiên, giọng văn thân thiện
- Không đưa ra giải pháp cụ thể, chỉ cam kết chuyển phản ánh hoặc ghi nhận"""

def build_prompt(feedback_text, sentiment, topics):
    """
    Build context-injected prompt for Llama 3.
    
    Args:
        feedback_text: Nội dung feedback gốc từ sinh viên
        sentiment: Kết quả phân loại cảm xúc (negative/neutral/positive)
        topics: List các chủ đề được nhận diện (lecturer/facility/training_program/others)
    
    Returns:
        Prompt string formatted cho Llama 3 Instruct
    """
    # Map sentiment to Vietnamese
    sentiment_vn = {
        "negative": "Tiêu cực",
        "neutral": "Trung tính", 
        "positive": "Tích cực"
    }
    
    # Map topics to Vietnamese
    topic_vn = {
        "lecturer": "Giảng viên",
        "facility": "Cơ sở vật chất",
        "training_program": "Chương trình đào tạo",
        "others": "Vấn đề khác"
    }
    
    sent_label = sentiment_vn.get(sentiment.lower(), sentiment)
    topic_labels = [topic_vn.get(t.lower().replace(" ", "_"), t) for t in topics]
    topic_str = ", ".join(topic_labels) if topic_labels else "Chưa xác định rõ"
    
    # Context injection
    user_message = f"""Sinh viên vừa gửi phản hồi sau:
"{feedback_text}"

Hệ thống AI phân tích:
- Cảm xúc: {sent_label}
- Chủ đề: {topic_str}

Hãy viết một câu trả lời ngắn gọn, thích hợp với ngữ cảnh này."""

    # Format theo Llama 3 Instruct template
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    return prompt

# =====================================
# MODEL LOADING
# =====================================

def load_llama_model(model_id=DEFAULT_MODEL_ID, use_quantization=True, cache_dir=None):
    """
    Load Llama 3 model với optional 4-bit quantization.
    
    Args:
        model_id: HuggingFace model ID
        use_quantization: Có sử dụng 4-bit quantization không (khuyên dùng cho GPU < 16GB)
        cache_dir: Thư mục cache model
    
    Returns:
        (model, tokenizer)
    """
    print(f"🔄 Đang load Llama 3 model: {model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Thiết lập pad token nếu chưa có
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config (4-bit để tiết kiệm VRAM)
    if use_quantization and torch.cuda.is_available():
        print("⚙️  Sử dụng 4-bit quantization (tiết kiệm VRAM)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    else:
        print("⚙️  Load model bình thường (không quantization)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            model = model.to(DEVICE)
    
    model.eval()
    print(f"✅ Model loaded thành công! Device: {model.device}")
    
    return model, tokenizer

# =====================================
# TEXT GENERATION
# =====================================

def generate_response(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
    """
    Generate response từ Llama 3.
    
    Args:
        model: Llama model
        tokenizer: Tokenizer
        prompt: Input prompt (đã được format)
        max_new_tokens: Số token tối đa sinh ra
        temperature: Độ ngẫu nhiên (0.0 = deterministic, 1.0 = creative)
        top_p: Nucleus sampling threshold
    
    Returns:
        Generated text (chỉ phần assistant response, đã strip prompt)
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only assistant's response (remove prompt)
    # Tìm vị trí bắt đầu của assistant response
    if "<|start_header_id|>assistant<|end_header_id|>" in prompt:
        # Split by assistant header và lấy phần sau cùng
        parts = full_text.split("assistant")
        if len(parts) >= 2:
            response = parts[-1].strip()
            # Remove special tokens nếu còn sót
            response = response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
            return response
    
    # Fallback: return toàn bộ (nếu không parse được)
    return full_text

# =====================================
# HIGH-LEVEL API
# =====================================

class LlamaAssistant:
    """
    Wrapper class để dễ sử dụng Llama 3 chatbot.
    """
    def __init__(self, model_id=DEFAULT_MODEL_ID, use_quantization=True, cache_dir=None):
        self.model, self.tokenizer = load_llama_model(
            model_id=model_id,
            use_quantization=use_quantization,
            cache_dir=cache_dir
        )
    
    def respond(self, feedback_text, sentiment, topics, max_tokens=200, temperature=0.7):
        """
        Generate response dựa trên kết quả phân tích từ Expert model (PhoBERT/mBERT).
        
        Args:
            feedback_text: Nội dung feedback gốc
            sentiment: Cảm xúc (negative/neutral/positive)
            topics: List chủ đề
            max_tokens: Độ dài tối đa response
            temperature: Độ sáng tạo
        
        Returns:
            Response string
        """
        prompt = build_prompt(feedback_text, sentiment, topics)
        response = generate_response(
            self.model, 
            self.tokenizer, 
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        return response

# =====================================
# DEMO / TESTING
# =====================================

if __name__ == "__main__":
    # Test module
    print("="*60)
    print("Testing LLM Module - Llama 3 Assistant")
    print("="*60)
    
    # Example test case
    test_feedback = "Phòng học nóng quá, máy lạnh hỏng không học nổi"
    test_sentiment = "negative"
    test_topics = ["Facility"]
    
    print("\n📝 Input:")
    print(f"  Feedback: {test_feedback}")
    print(f"  Sentiment: {test_sentiment}")
    print(f"  Topics: {test_topics}")
    
    print("\n🔧 Building prompt...")
    prompt = build_prompt(test_feedback, test_sentiment, test_topics)
    print(prompt)
    
    print("\n" + "="*60)
    print("Để test generation, cần load model (yêu cầu GPU/RAM lớn)")
    print("Chạy: assistant = LlamaAssistant()")
    print("      response = assistant.respond(text, sentiment, topics)")
    print("="*60)
