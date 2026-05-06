import os
import shutil

# --- 1. Configuration (إعدادات المسارات) ---
images_dir = r'D:\Computer_Science\DroneProject\data_src\images\important_classes'
labels_dir = r'D:\Computer_Science\DroneProject\src\Autonomous_Drone_Project\runs\detect\Drone_Auto_Labels'
target_dir = r'D:\Computer_Science\Newfolder' # عدلت حرف الـ e الناقص في مسارك

# الـ Classes بتاعتك بنفس ترتيب الـ Training
classes = ['Animal', 'Obstacle', 'Person', 'Vehicle']

# --- 2. Create Target Directory (تجهيز الفولدر النهائي) ---
os.makedirs(target_dir, exist_ok=True)

# --- 3. Create classes.txt (إنشاء ملف الأسماء) ---
classes_file_path = os.path.join(target_dir, 'classes.txt')
with open(classes_file_path, 'w') as f:
    for cls in classes:
        f.write(f"{cls}\n")
print("✅ Created classes.txt successfully.")

# --- 4. Indexing Images (أرشفة مسارات الصور لتسريع البحث) ---
# دي حركة Optimization عشان منعملش Nested Loops تبطئ الكود
image_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
images_dict = {}

print("🔍 Scanning images...")
for root, dirs, files in os.walk(images_dir):
    for file in files:
        if file.lower().endswith(image_exts):
            base_name = os.path.splitext(file)[0]
            # بنحفظ اسم الصورة بدون امتداد كـ Key، ومسارها الكامل كـ Value
            images_dict[base_name] = os.path.join(root, file)

# --- 5. Matching and Copying (الربط والنسخ) ---
copied_count = 0
print("⚙️ Matching labels with images and copying to target folder...")

for root, dirs, files in os.walk(labels_dir):
    for file in files:
        if file.endswith('.txt') and file != 'classes.txt':
            base_name = os.path.splitext(file)[0]
            
            # لو الموديل عمل وسم، والصورة موجودة فعلاً في القاموس بتاعنا
            if base_name in images_dict:
                src_label_path = os.path.join(root, file)
                src_image_path = images_dict[base_name]
                
                # مسارات الحفظ الجديدة
                dst_label_path = os.path.join(target_dir, file)
                dst_image_path = os.path.join(target_dir, os.path.basename(src_image_path))
                
                # عملية النسخ
                try:
                    shutil.copy(src_label_path, dst_label_path)
                    shutil.copy(src_image_path, dst_image_path)
                    copied_count += 1
                except Exception as e:
                    print(f"❌ Error copying {base_name}: {e}")

# --- 6. Final Report (التقرير النهائي) ---
print("\n" + "="*50)
print(f"🚀 SUCCESS! Process completed from A to Z.")
print(f"📦 Successfully paired and moved {copied_count} images with their labels.")
print(f"📂 Open this folder in AnyLabeling: {target_dir}")
print("="*50)