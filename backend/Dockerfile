# استفاده از نسخه Slim برای کاهش حجم
FROM python:3.9-slim

# تنظیمات برای جلوگیری از ساخت فایل‌های اضافی و نمایش آنی لاگ‌ها
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# نصب وابستگی‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کدهای پروژه
COPY . .

# پورت کانتینر
EXPOSE 8000

# دستور اجرا (با حالت reload برای دولوپ)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]