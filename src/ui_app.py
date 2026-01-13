 # =====================================
# Streamlit Demo App
# Multitask Sentiment & Topic Analysis
# Vietnamese Student Feedback
# =====================================

import os
import sys
import numpy as np
import torch
import streamlit as st

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from model_utils import load_model

# =====================================
# CONFIG
# =====================================
# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHOBERT_DIR = os.path.join(PROJECT_ROOT, "outputs/models/phobert_multitask/final_model")
MBERT_DIR   = os.path.join(PROJECT_ROOT, "outputs/models/mbert_multitask/final_model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# LOAD MODEL (với caching)
# =====================================
@st.cache_resource
def load_cached_model_v3(model_dir):
    """Load model with Streamlit caching - v3 with embedding support"""
    return load_model(model_dir, device=DEVICE)

# =====================================
# INFERENCE
# =====================================
def predict(text, model, tokenizer, sent_map, topic_map, threshold):
    """
    Predict sentiment and topics for input text.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)

    with torch.no_grad():
        # Get logits and embedding
        sent_logits, topic_logits, embedding = model(**enc, return_embedding=True)

    # Sentiment prediction
    sent_probs_all = torch.softmax(sent_logits, dim=-1).cpu().numpy()[0] # All probs
    sent_id = int(sent_logits.argmax(-1).item())
    sent_label = sent_map[sent_id]

    # Topic prediction (multi-label with threshold)
    prob = torch.sigmoid(topic_logits).cpu().numpy()[0]
    topic_ids = np.where(prob >= threshold)[0].tolist()

    # If no topic above threshold, take the highest probability one
    if len(topic_ids) == 0:
        topic_ids = [int(prob.argmax())]

    topics = [topic_map[i] for i in topic_ids]
    
    # Return embedding as numpy array
    emb_np = embedding.cpu().numpy()[0]
    
    return sent_label, topics, prob, sent_probs_all, emb_np

# =====================================
# STREAMLIT UI
# =====================================
st.set_page_config(page_title="Vietnamese Student Feedback Analysis", layout="wide", page_icon="🎓")

st.title("🎓 Vietnamese Student Feedback Analysis")
st.caption("🚀 AI-Powered Sentiment & Topic Analysis")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Cấu hình Model")
    model_choice = st.selectbox(
        "Chọn mô hình Expert", 
        ["PhoBERT", "mBERT"],
        help="PhoBERT: Pretrained cho tiếng Việt, mBERT: Multilingual BERT"
    )
    
    threshold = st.slider(
        "Topic Threshold", 
        0.0, 1.0, 0.35, 0.05,
        help="Ngưỡng xác suất để quyết định một topic có xuất hiện hay không."
    )
    
    st.divider()
    st.markdown("### 💡 Ví dụ mẫu")
    examples = [
        "Giảng viên nhiệt tình, slide rõ ràng dễ hiểu",
        "Phòng học nóng, máy chiếu mờ không nhìn rõ",
        "Chương trình học quá nặng, deadline nhiều quá",
        "Thầy giảng hay nhưng đi học xa quá mệt"
    ]
    
    for ex in examples:
        if st.button(f"📝 {ex[:30]}..."):
            st.session_state.example_input = ex

    st.divider()
    st.markdown("### ℹ️ Chú thích Labels")
    st.info("""
    **Phân loại cảm xúc (Sentiment):**
    * Negative (Tiêu cực)
    * Neutral (Trung tính)
    * Positive (Tích cực)
    
    **Phân loại chủ đề (Topics):**
    * Lecturer (Giảng viên)
    * Training Program (Chương trình)
    * Facility (Cơ sở vật chất)
    * Others (Khác)
    """)

# Load Expert model (PhoBERT/mBERT)
model_dir = PHOBERT_DIR if model_choice == "PhoBERT" else MBERT_DIR
try:
    model, tokenizer, sent_map, topic_map, config = load_cached_model_v3(model_dir)
except Exception as e:
    st.error(f"❌ Lỗi khi load Expert model: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Hãy gửi cho tôi một câu đánh giá của sinh viên, tôi sẽ phân tích cảm xúc và chủ đề giúp bạn. 👇"}]

# Function to display analysis result
def display_analysis(text, sent, topics, probs, sent_map, topic_map, sent_probs_all=None, embedding=None):
    # Custom CSS for badges
    st.markdown("""
    <style>
    .custom-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.3rem 0.8rem;
        font-size: 0.9rem;
        font-weight: 600;
        border-radius: 8px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .badge-neg { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
    .badge-neu { background-color: #f5f5f5; color: #616161; border: 1px solid #e0e0e0; }
    .badge-pos { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
    .badge-topic { background-color: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }
    </style>
    """, unsafe_allow_html=True)

    # Sentiment Logic
    sent_class = "badge-neu"
    sent_label = "Neutral"
    if sent == "negative": 
        sent_class = "badge-neg"
        sent_label = "Negative"
    elif sent == "positive": 
        sent_class = "badge-pos"
        sent_label = "Positive"
    
    st.markdown("### Kết quả phân tích")
    
    # --- SENTIMENT SECTION ---
    st.markdown(f"""
    <div style="margin-bottom: 15px;">
        <span style="font-weight: bold; margin-right: 8px;">Cảm xúc chủ đạo:</span>
        <span class="custom-badge {sent_class}">{sent_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # --- TOPIC SECTION ---
    st.markdown("**Chủ đề nhận diện:**")
    if topics:
        topic_html = ""
        for t in topics:
            topic_html += f'<span class="custom-badge badge-topic" style="margin-right: 8px;">{t}</span>'
        st.markdown(topic_html, unsafe_allow_html=True)
    else:
        st.markdown("<span style='color: gray; font-style: italic;'>Không tìm thấy chủ đề cụ thể</span>", unsafe_allow_html=True)
    
    st.divider()
    
    # --- DETAILED ANALYSIS (COLLAPSIBLE) ---
    with st.expander(" Xem phân tích chi tiết", expanded=False):
        # Sentiment Probabilities Chart
        if sent_probs_all is not None:
            import pandas as pd
            import altair as alt
            
            st.markdown("#### Độ tin cậy của dự đoán cảm xúc")
            sent_df = pd.DataFrame({
                "Sentiment": ["Negative", "Neutral", "Positive"],
                "Confidence": sent_probs_all,
            })
            
            # Use Altair for multi-color horizontal bar chart
            chart = alt.Chart(sent_df).mark_bar().encode(
                x=alt.X('Confidence:Q', scale=alt.Scale(domain=[0, 1]), title='Độ tin cậy'),
                y=alt.Y('Sentiment:N', sort=['Negative', 'Neutral', 'Positive'], title=None),
                color=alt.Color('Sentiment:N', scale=alt.Scale(
                    domain=['Negative', 'Neutral', 'Positive'],
                    range=['#ef5350', '#bdbdbd', '#66bb6a']
                ), legend=None),
                tooltip=['Sentiment', alt.Tooltip('Confidence:Q', format='.2%')]
            ).properties(height=150)
            
            st.altair_chart(chart, width='stretch')
            
        st.markdown("---")
        
        # Topic probabilities
        st.markdown("#### Xác suất chi tiết các chủ đề")
        import pandas as pd
        topic_names = list(topic_map.values())
        
        chart_data = pd.DataFrame({
            "Topic": topic_names,
            "Probability": probs
        })
        
        st.bar_chart(
            chart_data.set_index("Topic"),
            color="#29B5E8", 
            height=200
        )
        
        # Show table
        st.dataframe(
            chart_data.style.format({"Probability": "{:.2%}"}), 
            hide_index=True
        )
    
    # --- EMBEDDING VISUALIZATION (COLLAPSIBLE) ---
    if embedding is not None:
        with st.expander(" Xem Vector Embedding (Nâng cao)", expanded=False):
            st.markdown("**CLS Token Representation - Vector 768 chiều**")
            st.caption("Đây là biểu diễn số học (vector) mà mô hình BERT 'hiểu' về câu văn này trước khi đưa ra quyết định.")
            
            # Heatmap visualization
            import altair as alt
            
            # Create a long-form dataframe for Altair
            rows, cols = np.indices((24, 32))
            source = pd.DataFrame({
                'x': cols.ravel(),
                'y': rows.ravel(),
                'value': embedding.ravel()
            })
            
            heatmap = alt.Chart(source).mark_rect().encode(
                x=alt.X('x:O', axis=None),
                y=alt.Y('y:O', axis=None),
                color=alt.Color('value:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                tooltip=['value']
            ).properties(
                width='container',
                height=150
            ).configure_axis(
                grid=False
            ).configure_view(
                strokeWidth=0
            )
            
            st.altair_chart(heatmap, width='stretch')
            
            st.caption("Biểu diễn dưới dạng mảng số (Array):")
            # Show snippet: [0.123, -0.456, ..., 0.789]
            snippet = f"[{', '.join([f'{x:.4f}' for x in embedding[:8]])}, ..., {', '.join([f'{x:.4f}' for x in embedding[-8:]])}]"
            st.code(snippet, language="python")
            
            with st.expander("Xem toàn bộ giá trị vector (768 chiều)"):
                st.code(str(embedding.tolist()), language="json")

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            if isinstance(msg["content"], str):
                st.markdown(msg["content"])
            else:
                # It's a result dictionary
                res = msg["content"]
                display_analysis(
                    res["text"], res["sent"], res["topics"], 
                    res["probs"], sent_map, topic_map,
                    res.get("sent_probs_all"), res.get("embedding")
                )

# Handle input (Chat input or Example button)
prompt = st.chat_input("Nhập feedback của sinh viên tại đây...")

# Check if example button was clicked
if "example_input" in st.session_state and st.session_state.example_input:
    prompt = st.session_state.example_input
    st.session_state.example_input = None  # Reset

if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        # Expert Analysis (PhoBERT/mBERT)
        with st.spinner("🧠 Đang phân tích..."):
            sent, topics, probs, sent_probs_all, embedding = predict(
                prompt, model, tokenizer, sent_map, topic_map, threshold
            )
        
        # Display analysis
        display_analysis(prompt, sent, topics, probs, sent_map, topic_map, sent_probs_all, embedding)
        
        # Save result to history
        result_data = {
            "text": prompt,
            "sent": sent,
            "topics": topics,
            "probs": probs,
            "sent_probs_all": sent_probs_all,
            "embedding": embedding
        }
        st.session_state.messages.append({"role": "assistant", "content": result_data})


