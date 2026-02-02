# استخدم Python 3.11 (أفضل توافق مع DeepFace)
FROM python:3.11-slim

# تثبيت المتطلبات للنظام
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# إعداد مجلد العمل
WORKDIR /app

# نسخ ملفات المتطلبات أولاً (للاستفادة من cache)
COPY requirements.txt .

# تثبيت المتطلبات
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ باقي الملفات
COPY . .

# إنشاء مجلد الرفع
RUN mkdir -p uploads

# تعيين المتغيرات البيئية
ENV HOST=0.0.0.0
ENV PORT=8000

# فتح المنفذ
EXPOSE 8000

# تشغيل التطبيق
CMD ["python", "main.py"]
