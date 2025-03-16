import streamlit as st
import os
import joblib
import pickle
from PIL import Image
import torch
from torchvision import models
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import warnings  # Import the warnings module
import plotly.graph_objects as go
import plotly.io as pio

# โหลดโมเดล Animal Classification ด้วย PyTorch
@st.cache_resource
def load_animal_model():
    # โหลด ResNet18 model จาก PyTorch
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 90)  # เปลี่ยนจำนวน output layer ตามจำนวน class ของคุณ (เช่น 90 คลาส)
    model.load_state_dict(torch.load("animal.pth", map_location=torch.device('cpu')))
    model.eval()  # โหมดสำหรับการทำนาย (inference)
    return model

animal_model = load_animal_model()

# กำหนดค่าเริ่มต้นสำหรับ session state
if 'page' not in st.session_state:
    st.session_state.page = 'NN'

# CSS จัดปุ่มเมนู

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Prompt&display=swap&subset=thai" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Prompt', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .stApp { margin-top: 0; }
    div.block-container { padding-top: 2rem; }
    .button-row {
        display: flex;
        justify-content: space-around;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: white;
        z-index: 9999;
        padding: 10px 0;
        border-bottom: 1px solid #ddd;
        margin-bottom: 0;
    }
    .button-row div { flex: 1; text-align: center; }
    .stButton>button {
        width: 100%; padding: 10px;
        border: 1px solid #ccc;
        border-radius: 10px;
        color: #333333; font-weight: bold;
    }
    .selected-button > button {
        color: red !important;
        border: 1px solid red !important;
    }
    .main-content { margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# ปุ่มนำทาง
st.markdown('<div class="button-row">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Machine Learning", key="nav_ml"):
        st.session_state.page = 'ML'
    if st.session_state.page == 'ML':
        st.markdown('<style>[data-testid="stButton"][data-streamlit-key="nav_ml"] > button {color:red !important; border:1px solid red !important;}</style>', unsafe_allow_html=True)

with col2:
    if st.button("Neural Network", key="nav_nn"):
        st.session_state.page = 'NN'
    if st.session_state.page == 'NN':
        st.markdown('<style>[data-testid="stButton"][data-streamlit-key="nav_nn"] > button {color:red !important; border:1px solid red !important;}</style>', unsafe_allow_html=True)

with col3:
    if st.button("HR Turnover ML", key="nav_hr"):
        st.session_state.page = 'HR'
    if st.session_state.page == 'HR':
        st.markdown('<style>[data-testid="stButton"][data-streamlit-key="nav_hr"] > button {color:red !important; border:1px solid red !important;}</style>', unsafe_allow_html=True)

with col4:
    if st.button("Animal Classification", key="nav_animal"):
        st.session_state.page = 'Animal'
    if st.session_state.page == 'Animal':
        st.markdown('<style>[data-testid="stButton"][data-streamlit-key="nav_animal"] > button {color:red !important; border:1px solid red !important;}</style>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# หน้า Machine Learning
if st.session_state.page == 'ML':
    st.title("Machine Learning")
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px;'>
        ในการบริหารงานทรัพยากรบุคคล (HR) หนึ่งในปัญหาสำคัญที่องค์กรเผชิญคือ <b>การลาออกของพนักงาน</b> ซึ่งส่งผลต่อประสิทธิภาพการทำงานและต้นทุนในการสรรหาพนักงานใหม่
    ดังนั้น จึงมีการนำ Machine Learning เข้ามาช่วยในการพยากรณ์ว่าพนักงานคนใดมีแนวโน้มที่จะลาออกจากองค์กร เพื่อให้ฝ่ายบริหารสามารถวางแผนป้องกันล่วงหน้าได้อย่างเหมาะสม
    โมเดลนี้ถูกเทรนจากข้อมูลจริงของพนักงาน เช่น อายุ รายได้ สถานภาพสมรส ประสบการณ์การทำงาน และรูปแบบการเดินทาง ซึ่งทั้งหมดนี้สามารถบ่งชี้พฤติกรรมการทำงานที่สัมพันธ์กับการลาออกได้อย่างชัดเจน
    </p>
""", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px;'>
    ผมได้เริ่มจากไปหา Dataset จาก 
    <a href="https://www.kaggle.com" style="color:#3498db; font-weight:bold;" target="_blank">Kaggle</a> 
    ที่เกี่ยวกับ 
    <a href="https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset" style="color:#3498db; font-weight:bold;" target="_blank">HR Analytics</a> 
    และทำการทำ Feature Engineering และสร้างโมเดล Machine Learning ด้วย XGBoost ซึ่งเป็นโมเดลที่มีประสิทธิภาพสูงในการจำแนกประเภท
    โดยใช้ข้อมูลของพนักงานที่ทำงานในบริษัทต่าง ๆ และทำการทำนายว่าพนักงานคนใดมีแนวโน้มที่จะลาออกจากองค์กร
    </p>
""", unsafe_allow_html=True)
    st.markdown(" ")

    st.markdown("""
<p style='font-family:Prompt; font-size:18px;'>
    โดยมี Feature ที่ใช้ในการทำนายดังนี้:<br>
    • อายุ (Age)<br>
    • เพศ (Gender)<br>
    • สถานะสมรส (Marital Status)<br>
    • อาชีพ (Job Role)<br>
    • การทำงานล่วงเวลา (OverTime)<br>
    • ระยะทางจากบ้าน (Distance from Home)<br>
    • รายได้ต่อเดือน (Monthly Income)<br>
    • การเดินทางไปทำงาน (Business Travel)<br>
    • แผนกงาน (Department)<br>
    • คุณภาพชีวิตที่ทำงาน (Work-Life Balance)<br>
    • จำนวนบริษัทที่ทำงานมา (Number of Companies Worked)
""", unsafe_allow_html=True)

    st.markdown(" ")
    img = Image.open("code/1.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    เริ่มจากผมนำเข้าเครื่องมือพื้นฐานที่จะเป็น เช่น pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost และ joblib
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    img = Image.open("code/2.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    ขั้นตอนต่อมาให้โปรแกรมอ่าน อ่านไฟล์ CSV ที่มีข้อมูลพนักงาน และทำการแปลงค่าของ Attrition ให้เป็น 0 หรือ 1
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    img = Image.open("code/3.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    แปลงค่า target จาก Yes/No เป็น 1/0 เพื่อใช้กับโมเดล classification
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    img = Image.open("code/4.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    เลือกเฉพาะ column ที่จำเป็นเพื่อลดความซับซ้อนของ dataset
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/5.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    แปลงข้อมูลที่เป็นตัวอักษร (เช่น ‘Male’, ‘Female’) ให้เป็นตัวเลข
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/6.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    แยกข้อมูลสำหรับ training
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/7.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    แบ่งข้อมูล 80% train / 20% test
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/8.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    สร้าง pipeline ที่รวมขั้นตอน preprocessing (scaling) และ training model
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/9.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    ดึงค่าความสำคัญของแต่ละ feature ออกมาจากโมเดล
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/9.5.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    เรียงลำดับ feature จากสำคัญมาก → น้อย
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/10.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    แสดงผลเป็นกราฟแท่ง
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/9.6.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    แผนภูมิแสดงความสำคัญของ feature แต่ละตัว
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/11.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    ทดสอบโมเดลบน test set และประเมินผลด้วย Accuracy + Report
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/13.png")
    st.image(img, use_container_width=True)


    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/12.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    บันทึกโมเดลเป็นไฟล์ .pkl เพื่อเรียกใช้ในหน้าเว็บ
    </p>
""", unsafe_allow_html=True)

# หน้า Neural Network
elif st.session_state.page == 'NN':
    st.title("Neural Network")
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px;'>
        ในปัจจุบัน เทคโนโลยีปัญญาประดิษฐ์ (Artificial Intelligence) ได้เข้ามามีบทบาทสำคัญในด้านการวิเคราะห์และประมวลผลข้อมูลภาพ โดยเฉพาะในงานด้านการจำแนกประเภทวัตถุต่าง ๆ เช่น การจำแนกภาพสัตว์ ซึ่งสามารถนำไปประยุกต์ใช้ในหลายด้าน เช่น ระบบการศึกษาด้านชีววิทยา, การติดตามสัตว์ป่า, การจัดหมวดหมู่ข้อมูลภาพ และระบบช่วยเหลือผู้พิการทางสายตา
    โครงการนี้จึงมีวัตถุประสงค์เพื่อพัฒนาโมเดลปัญญาประดิษฐ์สำหรับการจำแนกภาพสัตว์หลากหลายชนิด โดยใช้เทคนิค Convolutional Neural Network (CNN) ซึ่งเป็นโมเดลที่มีประสิทธิภาพสูงในการประมวลผลภาพ เพื่อให้สามารถทำนายประเภทของสัตว์จากภาพได้อย่างแม่นยำ
    ระบบนี้สามารถเรียนรู้จากภาพตัวอย่าง และประมวลผลภาพใหม่ได้แบบอัตโนมัติ ซึ่งเป็นการนำเทคโนโลยี Deep Learning มาช่วยแก้ปัญหาที่ซับซ้อนในลักษณะนี้ได้อย่างมีประสิทธิภาพ
    </p>
""", unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px;'>
    ผมได้เริ่มจากไปหา Dataset จาก 
    <a href="https://www.kaggle.com" style="color:#3498db; font-weight:bold;" target="_blank">Kaggle</a> 
    ที่เกี่ยวกับ 
    <a href="https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals" style="color:#3498db; font-weight:bold;" target="_blank">Animal Image Dataset (90 Different Animals)</a> 
    ภาพใน Dataset ประกอบด้วยสัตว์หลากหลายกลุ่ม เช่น สัตว์เลี้ยงลูกด้วยนม นก แมลง และสัตว์น้ำ โดยผมได้ทำการเตรียมข้อมูลเบื้องต้น เช่น การแปลงขนาดภาพ (Resizing), การปรับสีภาพ (Normalization) และการเพิ่มข้อมูลภาพเทียม (Data Augmentation) เพื่อเพิ่มความหลากหลายของข้อมูล
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/1.1.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:17px; color:black; text-align:center;'>
    import ไลบรารีที่จำเป็น สำหรับการทำ Deep Learning ด้วย PyTorch และใช้ Mixed Precision training (autocast, GradScaler) เพื่อเพิ่มประสิทธิภาพในการเทรนโมเดลด้วย GPU
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/1.2.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    ตรวจสอบการใช้ GPU ถ้าไม่มี GPU จะใช้ CPU แทน
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/1.3.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    กำหนดค่า Hyperparameters และใช้ patience ในการหยุดการเทรนเมื่อไม่มีการปรับปรุงค่า loss ในจำนวนรอบที่กำหนด
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/1.4.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    การใช้ Data Augmentation ในการปรับข้อมูลภาพเพื่อเพิ่มความหลากหลายของข้อมูล และ Normalize ค่า Pixel ของรูปภาพ โดยใช้ค่า Mean และ Std ของ ImageNet
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/1.5.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    สร้าง DataLoader สำหรับ Training Set และ Validation Set
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/1.6.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    โหลดโมเดล ResNet50 ที่เทรนด้วย ImageNet มาแล้ว
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/1.7.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    ปรับแต่ง Layer สุดท้ายของโมเดลให้เหมาะสมกับจำนวนคลาส และ ลดค่า learning rate ลงอัตโนมัติหาก loss ไม่ลดลงต่อเนื่อง 3 epochs
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/1.8.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    ปรับแต่ง Layer สุดท้ายของโมเดลให้เหมาะสมกับจำนวนคลาส และ ลดค่า learning rate ลงอัตโนมัติหาก loss ไม่ลดลงต่อเนื่อง 3 epochs
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/1.10.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    ในแต่ละ epoch ทำการฝึกโมเดลจากข้อมูล Train แล้วตรวจสอบผลลัพธ์จากข้อมูล Validation
    </p>
""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    img = Image.open("code/1.11.png")
    st.image(img, use_container_width=True)
    st.markdown("""
    <p style='font-family:Prompt; font-size:18px; color:black; text-align:center;'>
    ถ้าผล Validation loss ดีขึ้นจะบันทึกโมเดลใหม่ ถ้าไม่ดีขึ้นติดต่อกัน 3 epochs จะลด learning rate ลง และ จะบันทึกโมเดลที่ดีที่สุด
    </p>
""", unsafe_allow_html=True)




# หน้า HR Employee Attrition Prediction
elif st.session_state.page == 'HR':
    st.title("HR Employee Attrition Prediction")
    st.write("กรุณากรอกข้อมูลของพนักงานเพื่อทำนายว่าเขาจะลาออกหรือไม่")

    with st.form("hr_form"):
        st.subheader("ข้อมูลพนักงาน")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=60, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                                 "Manufacturing Director", "Healthcare Representative", 
                                                 "Manager", "Sales Representative", "Research Director", 
                                                 "Human Resources"])
            overtime = st.selectbox("OverTime", ["Yes", "No"])
        with col2:
            distance = st.number_input("Distance from Home (km)", min_value=1, max_value=50, value=10)
            monthly_income = st.number_input("Monthly Income (USD/month)", min_value=1000, max_value=20000, value=5000)
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            worklife = st.selectbox("Work-Life Balance (1=Bad, 4=Best)", [1, 2, 3, 4])
        num_companies = st.number_input("Number of Companies Worked (รวมจำนวนบริษัทที่เคยทำงานก่อนหน้านี้, 0 = งานแรก)", 
                                        min_value=0, max_value=20, value=2)
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            import numpy as np
            import joblib

            model = joblib.load("hr_attrition_model.pkl")

            gender_map = {"Male": 1, "Female": 0}
            marital_map = {"Single": 2, "Married": 1, "Divorced": 0}
            job_map = {"Sales Executive": 7, "Research Scientist": 6, "Laboratory Technician": 3,
                       "Manufacturing Director": 2, "Healthcare Representative": 1,
                       "Manager": 4, "Sales Representative": 8, "Research Director": 5, "Human Resources": 0}
            overtime_map = {"Yes": 1, "No": 0}
            travel_map = {"Travel_Rarely": 2, "Travel_Frequently": 1, "Non-Travel": 0}
            dept_map = {"Sales": 2, "Research & Development": 1, "Human Resources": 0}

            input_data = np.array([[age, gender_map[gender], marital_map[marital_status],
                                    job_map[job_role], overtime_map[overtime], distance, monthly_income,
                                    travel_map[business_travel], dept_map[department], worklife,
                                    num_companies]])

            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.error(f"💼 พนักงานนี้มีแนวโน้ม 'ลาออก' (Attrition)")
            else:
                st.success(f"✅ พนักงานนี้ 'อยู่ต่อในองค์กร'")

            st.info(f"ความน่าจะเป็นในการลาออก: {prob * 100:.2f}%")

            # สร้างกราฟ Plotly Gauge Chart เพื่อแสดงผลความน่าจะเป็น
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Attrition Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red" if prediction == 1 else "green"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 100], "color": "gray"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")

# หน้า Animal Classification
elif st.session_state.page == 'Animal':
    st.title("Animal Image Classification")
    st.write("อัปโหลดภาพสัตว์เพื่อทำนายผลการจำแนกประเภท")

    # ตัวอย่างของ 90 คลาส
    class_labels = [
        "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat", 
        "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", "dolphin", 
        "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish", 
        "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse", 
        "hummingbird", "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", 
        "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", 
        "parrot", "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", 
        "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel", 
        "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
    ]  # จำนวน 90 ชื่อสัตว์

    # สร้างฟอร์มอัปโหลดภาพ
    with st.form("animal_form"):
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # แปลงภาพเป็น RGB และแสดง
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

        # ปุ่มส่งข้อมูล (submit)
        submitted = st.form_submit_button("วิเคราะห์ภาพ")

        if submitted:
            # ตรวจสอบว่าได้อัปโหลดภาพหรือยัง
            if uploaded_file is not None:
                # ปรับขนาดภาพให้ตรงกับ input shape ที่โมเดลต้องการ
                image = image.resize((224, 224)) 
                image_array = np.array(image) / 255.0  # Normalize ค่าให้เป็น [0, 1]
                image_array = np.expand_dims(image_array, axis=0)  # แปลงให้เป็น batch

                # แปลงเป็น Tensor ของ PyTorch
                image_tensor = torch.tensor(image_array).float()
                image_tensor = image_tensor.permute(0, 3, 1, 2)  # เปลี่ยน shape เป็น [batch_size, channels, height, width]

                # การพยากรณ์จากโมเดล
                with torch.no_grad():
                    output = animal_model(image_tensor)  # ใช้ PyTorch model
                    _, predicted_index = torch.max(output, 1)

                    # ตรวจสอบว่า predicted_index อยู่ในขอบเขตของ class_labels
                    if predicted_index.item() < len(class_labels):
                        predicted_label = class_labels[predicted_index.item()]
                        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_index.item()] * 100
                        st.success(f"🔍 ผลการวิเคราะห์: {predicted_label}")
                        st.markdown(f"**ความมั่นใจในการทำนาย: {confidence:.2f}%**")
                    else:
                        st.error("⚠️ การทำนายเกินขอบเขตของคลาสที่กำหนด")
            else:
                st.error("⚠️ กรุณาอัปโหลดภาพก่อนวิเคราะห์")
else:
    pass  # Optional: Handle other cases

st.markdown('</div>', unsafe_allow_html=True)
