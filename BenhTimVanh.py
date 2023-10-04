import tkinter as tk
import numpy as np
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



#đọc dữ liệu từ file csv
data = pd.read_csv('Heart.csv')
#kết nối db
conn = sqlite3.connect("heart.db")
# Lưu dữ liệu vào database
data.to_sql('heart', conn, if_exists='replace', index=False)
# Load dữ liệu bệnh tim
dulieu = pd.read_sql_query("Select * from heart", conn)


X_train, X_test, y_train, y_test = train_test_split(dulieu.drop('target', axis=1), dulieu['target'], test_size=0.2, random_state=42)

# Huấn luyện mô hình bằng thuật toán Random Forest algorithm¶

rf = RandomForestClassifier(n_estimators=100, random_state=42)


def train_model():
    rf.fit(X_train, y_train)
    # Hiển thị kết quả huấn luyện
    score = rf.score(X_test, y_test)
    result_label.config(text=f"Độ chính xác:{score:.4f}")
    
# Tạo giao diện
root = tk.Tk()
root.title("Chẩn đoán bệnh tim")

# Tạo các widget
age_label = tk.Label(root, text="Tuổi:")
age_entry = tk.Entry(root)

sex_label = tk.Label(root, text="Giới tính:")
sex_entry = tk.Entry(root)

cp_label = tk.Label(root, text="Chest Pain Type:")
cp_entry = tk.Entry(root)

trestbps_label = tk.Label(root, text="trestbps:")
trestbps_entry = tk.Entry(root)

chol_label = tk.Label(root, text="Cholesterol:")
chol_entry = tk.Entry(root)

fbs_label = tk.Label(root, text="fbs:")
fbs_entry = tk.Entry(root)

restecg_label = tk.Label(root, text="restECG:")
restecg_entry = tk.Entry(root)

exang_label = tk.Label(root, text="exAng:")
exang_entry = tk.Entry(root)

oldpeak_label = tk.Label(root, text="oldpeak:")
oldpeak_entry = tk.Entry(root)

slope_label = tk.Label(root, text="Slope:")
slope_entry = tk.Entry(root)

ca_label = tk.Label(root, text="ca:")
ca_entry = tk.Entry(root)

thal_label = tk.Label(root, text="thal:")
thal_entry = tk.Entry(root)

# Tạo các widget
train_button = tk.Button(root, text="Huấn luyện", command=train_model)
result_label = tk.Label(root, text="Nhấn nút 'Huấn luyện' để bắt đầu huấn luyện mô hình")


# Định vị các widget trên giao diện
age_label.grid(row=0, column=0)
age_entry.grid(row=0, column=1)

sex_label.grid(row=1, column=0)
sex_entry.grid(row=1, column=1)

cp_label.grid(row=2, column=0)
cp_entry.grid(row=2, column=1)

trestbps_label.grid(row=3, column=0)
trestbps_entry.grid(row=3, column=1)

chol_label.grid(row=4, column=0)
chol_entry.grid(row=4, column=1)

fbs_label.grid(row=5, column=0)
fbs_entry.grid(row=5, column=1)

restecg_label.grid(row=6, column=0)
restecg_entry.grid(row=6, column=1)

exang_label.grid(row=8, column=0)
exang_entry.grid(row=8, column=1)

oldpeak_label.grid(row=9, column=0)
oldpeak_entry.grid(row=9, column=1)

slope_label.grid(row=10, column=0)
slope_entry.grid(row=10, column=1)

ca_label.grid(row=11, column=0)
ca_entry.grid(row=11, column=1)

thal_label.grid(row=12, column=0)
thal_entry.grid(row=12, column=1)

result_label.grid(row=13, column=0)
# Định vị các widget trên cửa sổ
train_button.grid(row=14, column=0)
result_label.grid(row=15, column=1)

# Hàm chẩn đoán bệnh tim
def diagnose():
    age = int(age_entry.get())
    sex = int(sex_entry.get())
    cp = int(cp_entry.get())
    trestbps = int(trestbps_entry.get())
    chol = int(chol_entry.get())
    fbs = int(fbs_entry.get())
    restecg = int(restecg_entry.get())
    exang = int(exang_entry.get())
    oldpeak = float(oldpeak_entry.get())
    slope = int(slope_entry.get())
    ca = int(ca_entry.get())
    thal = int(thal_entry.get())
   
    # Chẩn đoán bệnh tim
    result = np.array([age, sex, cp, trestbps, chol, fbs, restecg, exang, oldpeak, slope, ca, thal])
    test = result.reshape(1, -1)
    result_test = rf.predict(test)
    # Hiển thị kết quả
    if result_test == 0:
        result_label.config(text="Không Bị bệnh tim")
    else:
        result_label.config(text="Bị bệnh tim")
# Tạo nút chẩn đoán
diagnose_button = tk.Button(root,text="Chẩn đoán", command=diagnose)
diagnose_button.grid(row=13, column=1)

# Chạy giao diện
#root.mainloop()



def main():
    root.mainloop()

main()