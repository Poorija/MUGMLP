# استفاده از Node 22 Alpine طبق نیاز package.json شما
FROM node:22-alpine

WORKDIR /app

# کپی کردن پکیج‌ها و فایل قفل برای نصب دقیق نسخه‌ها
COPY package.json ./

# نصب وابستگی‌ها
RUN npm install

# کپی کردن سورس کد
COPY . .

# پورت کانتینر
EXPOSE 3000

# دستور اجرا
CMD ["npm", "start"]