import React, { useState, useEffect } from 'react';
import api from '../services/api';
import { Container, Typography, TextField, Button, Paper, Box, Modal, Alert } from '@mui/material';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import Logo from './Logo';
import zxcvbn from 'zxcvbn';
import '../App.css'; // Ensure we have CSS for blur

const Login = () => {
  const { t } = useTranslation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [captchaText, setCaptchaText] = useState('');
  const [captchaImage, setCaptchaImage] = useState('');
  const [captchaSessionId, setCaptchaSessionId] = useState('');
  const [otpCode, setOtpCode] = useState('');

  // States for different steps
  const [step, setStep] = useState('login'); // login, 2fa, change_password
  const [error, setError] = useState('');

  // Password Change State
  const [newPassword, setNewPassword] = useState('');
  const [passwordStrength, setPasswordStrength] = useState(0);

  const navigate = useNavigate();

  useEffect(() => {
    fetchCaptcha();
  }, []);

  const fetchCaptcha = async () => {
    try {
      const response = await api.get('/captcha', { responseType: 'blob' });
      setCaptchaSessionId(response.headers['x-captcha-session-id']);
      setCaptchaImage(URL.createObjectURL(response.data));
    } catch (err) {
      console.error('Failed to fetch captcha', err);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);

      const response = await api.post('/token', formData, {
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-Captcha-Session-Id': captchaSessionId,
            'X-Captcha-Text': captchaText,
            'X-OTP-Code': otpCode
        },
      });

      const data = response.data;

      if (data.require_password_change) {
          // Store token temporarily to allow password change request
          localStorage.setItem('temp_token', data.access_token);
          setStep('change_password');
          return;
      }

      localStorage.setItem('token', data.access_token);
      navigate('/dashboard');

    } catch (err) {
      if (err.response && err.response.status === 401 && err.response.data.detail === "2FA Required") {
          setStep('2fa');
          setError('');
      } else {
          setError(t('error_login'));
          fetchCaptcha(); // Refresh captcha on failure
      }
    }
  };

  const handleChangePassword = async (e) => {
      e.preventDefault();
      try {
          const tempToken = localStorage.getItem('temp_token');
          await api.post('/auth/change-password', {
              old_password: password,
              new_password: newPassword
          }, {
              headers: { Authorization: `Bearer ${tempToken}` }
          });

          // After success, force re-login or just go to dashboard?
          // Usually re-login is safer but token is valid.
          // Let's update the stored token to be the real one.
          localStorage.setItem('token', tempToken);
          localStorage.removeItem('temp_token');
          navigate('/dashboard');

      } catch(err) {
          setError(t('error_password'));
      }
  };

  const handlePasswordStrength = (e) => {
      const val = e.target.value;
      setNewPassword(val);
      const result = zxcvbn(val);
      setPasswordStrength(result.score);
  };

  const strengthLabel = [t('weak'), t('weak'), t('fair'), t('good'), t('strong')];
  const strengthColor = ['error', 'error', 'warning', 'success', 'success'];

  return (
    <div className="login-background">
        <Container maxWidth="sm" sx={{ mt: 8, position: 'relative', zIndex: 2 }}>
        <Paper sx={{ p: 4, borderRadius: 2, backdropFilter: 'blur(10px)', backgroundColor: 'rgba(255,255,255,0.8)' }}>
            <Box mb={3}>
                <Logo size="large" />
            </Box>
            <Typography variant="h4" align="center" gutterBottom>{t('login')}</Typography>

            {step === 'login' && (
                <form onSubmit={handleLogin}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                        label={t('email')}
                        type="email"
                        variant="outlined"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                        fullWidth
                    />
                    <TextField
                        label={t('password')}
                        type="password"
                        variant="outlined"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                        fullWidth
                    />

                    {/* Captcha */}
                    <Box display="flex" gap={1}>
                         <img src={captchaImage} alt="captcha" style={{ height: 56, borderRadius: 4 }} />
                         <Button onClick={fetchCaptcha} variant="outlined" size="small">{t('captcha_refresh')}</Button>
                    </Box>
                    <TextField
                        label={t('captcha')}
                        variant="outlined"
                        value={captchaText}
                        onChange={(e) => setCaptchaText(e.target.value)}
                        required
                        fullWidth
                    />

                    <Button type="submit" variant="contained" size="large" fullWidth>{t('login')}</Button>
                    <Box textAlign="center">
                        <RouterLink to="/register">{t('register')}</RouterLink>
                    </Box>
                </Box>
                </form>
            )}

            {step === '2fa' && (
                 <form onSubmit={handleLogin}>
                     <Typography variant="body1" gutterBottom>{t('otp_code')}</Typography>
                     <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                         <TextField
                             label="Code"
                             value={otpCode}
                             onChange={(e) => setOtpCode(e.target.value)}
                             required
                             fullWidth
                         />
                         <Button type="submit" variant="contained" fullWidth>{t('verify_2fa')}</Button>
                     </Box>
                 </form>
            )}

            {step === 'change_password' && (
                <form onSubmit={handleChangePassword}>
                    <Alert severity="warning" sx={{ mb: 2 }}>{t('force_change_message')}</Alert>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <TextField
                            label={t('new_password')}
                            type="password"
                            value={newPassword}
                            onChange={handlePasswordStrength}
                            required
                            fullWidth
                        />
                        {/* Strength Meter */}
                        <Box>
                            <Typography variant="caption">{t('password_strength')}: </Typography>
                            <Typography variant="caption" color={`${strengthColor[passwordStrength]}.main`} fontWeight="bold">
                                {strengthLabel[passwordStrength]}
                            </Typography>
                            <div style={{ height: 5, width: '100%', backgroundColor: '#e0e0e0', borderRadius: 2 }}>
                                <div style={{
                                    height: '100%',
                                    width: `${(passwordStrength + 1) * 20}%`,
                                    backgroundColor: passwordStrength < 2 ? 'red' : passwordStrength === 2 ? 'orange' : 'green',
                                    borderRadius: 2,
                                    transition: 'width 0.3s'
                                }} />
                            </div>
                             <Typography variant="caption" display="block">
                                 Requires: Min 8 chars, letters & numbers.
                             </Typography>
                        </Box>

                        <Button type="submit" variant="contained" fullWidth disabled={passwordStrength < 2}>{t('change_password')}</Button>
                    </Box>
                </form>
            )}

            {error && <Typography color="error" align="center" sx={{ mt: 2 }}>{error}</Typography>}
        </Paper>
        </Container>
    </div>
  );
};

export default Login;
