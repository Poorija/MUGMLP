import i18n from "i18next";
import { initReactI18next } from "react-i18next";

// English
const en = {
    translation: {
        "login": "Login",
        "register": "Register",
        "email": "Email",
        "password": "Password",
        "welcome": "Welcome to Your Dashboard",
        "create_project": "Create New Project",
        "project_name": "Project Name",
        "create": "Create",
        "projects": "Your Projects",
        "open_project": "Open Project",
        "logout": "Logout",
        "about": "About",
        "developer": "Developer",
        "theme": "Theme",
        "language": "Language",
        "captcha": "Captcha",
        "captcha_refresh": "Refresh Captcha",
        "old_password": "Old Password",
        "new_password": "New Password",
        "change_password": "Change Password",
        "password_change_required": "Password Change Required",
        "force_change_message": "For security reasons, you must change your password.",
        "otp_code": "2FA Code",
        "setup_2fa": "Setup 2FA",
        "verify_2fa": "Verify 2FA",
        "enable_2fa": "Enable 2FA",
        "scan_qr": "Scan this QR code with your Authenticator App",
        "password_strength": "Password Strength",
        "weak": "Weak",
        "fair": "Fair",
        "good": "Good",
        "strong": "Strong",
        "predict": "Predict",
        "models": "Models",
        "datasets": "Datasets",
        "upload_dataset": "Upload Dataset",
        "train_model": "Train Model",
        "overview": "Overview",
        "data_analysis": "Data Analysis",
        "comparison": "Comparison",
        "result": "Result",
        "close": "Close",
        "error_login": "Failed to login. Please check your credentials.",
        "error_captcha": "Invalid Captcha",
        "error_2fa": "Invalid 2FA Code",
        "error_password": "Password too weak or incorrect old password."
    }
};

// Persian
const fa = {
    translation: {
        "login": "ورود",
        "register": "ثبت نام",
        "email": "ایمیل",
        "password": "رمز عبور",
        "welcome": "به داشبورد خوش آمدید",
        "create_project": "ایجاد پروژه جدید",
        "project_name": "نام پروژه",
        "create": "ایجاد",
        "projects": "پروژه های شما",
        "open_project": "باز کردن پروژه",
        "logout": "خروج",
        "about": "درباره ما",
        "developer": "توسعه دهنده",
        "theme": "تم",
        "language": "زبان",
        "captcha": "کد امنیتی",
        "captcha_refresh": "تغییر کد امنیتی",
        "old_password": "رمز عبور فعلی",
        "new_password": "رمز عبور جدید",
        "change_password": "تغییر رمز عبور",
        "password_change_required": "تغییر رمز عبور الزامی است",
        "force_change_message": "به دلایل امنیتی، باید رمز عبور خود را تغییر دهید.",
        "otp_code": "کد تایید دو مرحله ای",
        "setup_2fa": "تنظیم تایید دو مرحله ای",
        "verify_2fa": "تایید کد",
        "enable_2fa": "فعال سازی تایید دو مرحله ای",
        "scan_qr": "این کد QR را با برنامه Authenticator اسکن کنید",
        "password_strength": "قدرت رمز عبور",
        "weak": "ضعیف",
        "fair": "متوسط",
        "good": "خوب",
        "strong": "قوی",
        "predict": "پیش بینی",
        "models": "مدل ها",
        "datasets": "مجموعه داده ها",
        "upload_dataset": "آپلود داده",
        "train_model": "آموزش مدل",
        "overview": "بررسی اجمالی",
        "data_analysis": "تحلیل داده",
        "comparison": "مقایسه",
        "result": "نتیجه",
        "close": "بستن",
        "error_login": "ورود ناموفق. لطفا اطلاعات خود را بررسی کنید.",
        "error_captcha": "کد امنیتی نامعتبر است",
        "error_2fa": "کد تایید دو مرحله ای نامعتبر است",
        "error_password": "رمز عبور ضعیف است یا رمز قبلی اشتباه است."
    }
};

i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: en,
      fa: fa
    },
    lng: "en",
    fallbackLng: "en",
    interpolation: {
      escapeValue: false
    }
  });

export default i18n;
